"""
Mispricing scan: pra cada série, compara fair value (BS com IV da SVI) vs mid de mercado.

Output: lista ordenada por |edge| desc. Cada row tem:
    - symbol, underlying, side, strike, expiration
    - iv_empírica (do mercado)
    - iv_svi (interpolada pela surface)
    - fair_value (preço BS com iv_svi)
    - mid (mercado)
    - edge_abs (fair - mid, em R$)
    - edge_pct (edge_abs / mid, em %)

Edge positivo: tu acha que tá CARO (mercado > teu fair) → considera VENDER
Edge negativo: tu acha que tá BARATO (mercado < teu fair) → considera COMPRAR
"""
from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
from math import log
from pathlib import Path
from typing import Literal

import numpy as np

from src.bs import price
from src.svi import SVIParams, calibrate_svi


@dataclass
class MispricingRow:
    symbol: str
    underlying: str
    side: Literal["call", "put"]
    strike: float
    expiration: str
    bid: float
    ask: float
    mid: float
    iv_emp: float        # IV invertida do mercado
    iv_svi: float        # IV interpolada pela SVI
    fair_value: float    # preço BS com iv_svi
    edge_abs: float      # fair - mid (em R$)
    edge_pct: float      # edge_abs / mid (em %)
    recommendation: Literal["buy", "sell", "skip", "skip-warn"]
    spot: float
    risk_free: float
    dividend_yield: float
    t_years: float


def _iv_from_mid(mid: float, spot: float, strike: float, t: float,
                 r: float, q: float, side: str) -> float:
    """IV empírica via Newton impl_vol. Retorna 0 se falhar."""
    from src.bs import implied_vol
    try:
        return implied_vol(mid, spot, strike, t, r, q, side)
    except Exception:
        return 0.0


def scan_chain(csv_path: Path, edge_threshold_pct: float = 5.0,
               moneyness_min: float = 0.85, moneyness_max: float = 1.15):
    """
    Lê chain CSV, calibra SVI, gera mispricing rows.

    Returns: (list[MispricingRow], SVIParams) ou (None, None) se chain inválida.
    """
    rows = list(csv.DictReader(csv_path.open()))
    if len(rows) < 5:
        return None, None

    spot = float(rows[0]["spot"])
    r = float(rows[0]["risk_free"])
    q = float(rows[0]["dividend_yield"])
    t = float(rows[0]["t_years"])
    underlying = rows[0]["underlying"]

    forward = spot * np.exp((r - q) * t)

    # primeira passada: IV empírica bruta
    raw = []
    for row in rows:
        strike = float(row["strike"])
        bid = float(row["bid"])
        ask = float(row["ask"])
        mid = (bid + ask) / 2
        if mid <= 0:
            continue
        moneyness = strike / forward
        if moneyness < moneyness_min or moneyness > moneyness_max:
            continue
        side = row["side"]
        iv = _iv_from_mid(mid, spot, strike, t, r, q, side)
        if iv <= 0.10 or iv > 1.50:
            continue
        raw.append({"row": row, "iv": iv, "moneyness": moneyness})

    if len(raw) < 5:
        return None, None

    # ATM IV
    atm_ivs = [d["iv"] for d in raw if 0.95 <= d["moneyness"] <= 1.05]
    atm_iv = float(np.median(atm_ivs)) if atm_ivs else 0.40

    # segunda passada: descarta IVs outliers (>1.6× ATM)
    strikes_for_fit = []
    ivs_for_fit = []
    for d in raw:
        if d["iv"] > atm_iv * 1.6:
            continue
        strikes_for_fit.append(float(d["row"]["strike"]))
        ivs_for_fit.append(d["iv"])

    if len(strikes_for_fit) < 5:
        return None, None

    # calibra SVI
    svi = calibrate_svi(strikes_for_fit, ivs_for_fit, t, forward=forward)

    # detecta SVI degenerada (params no bound = fit ruim, edge não confiável)
    svi_degenerate = (
        svi.b >= 4.99           # bound superior de b
        or abs(svi.rho) >= 0.99  # bound de rho
        or svi.sigma <= 1e-3     # limite inferior de sigma (colapsou)
    )
    if svi_degenerate:
        # marca edge como skip mesmo se magnitude grande
        svi_warn = True
    else:
        svi_warn = False

    # terceira passada: pra cada série (incluindo as filtradas), calcula fair vs mid
    results = []
    for d in raw:
        row = d["row"]
        strike = float(row["strike"])
        bid = float(row["bid"])
        ask = float(row["ask"])
        mid = (bid + ask) / 2
        side = row["side"]

        # IV SVI interpolada
        k = log(strike / forward)
        iv_svi = svi.iv(k, t)

        # Fair value com IV SVI
        fair = price(spot, strike, t, r, q, iv_svi, side)

        edge_abs = fair - mid
        edge_pct = (edge_abs / mid * 100) if mid > 0 else 0.0

        if edge_pct > edge_threshold_pct:
            # fair > market → mercado barato → COMPRAR
            rec = "buy"
        elif edge_pct < -edge_threshold_pct:
            # fair < market → mercado caro → VENDER
            rec = "sell"
        else:
            rec = "skip"

        # se SVI degenerada, força skip independente da magnitude
        if svi_warn and rec != "skip":
            rec = "skip-warn"

        # edge > 25% é estatisticamente improvável — desconfiar da calibração SVI
        # (pode ser ruído do bid/ask spread, smile superestimado, etc)
        if abs(edge_pct) > 25.0 and rec in ("buy", "sell"):
            rec = "skip-warn"

        results.append(MispricingRow(
            symbol=row["symbol"],
            underlying=underlying,
            side=side,
            strike=strike,
            expiration=row["expiration"],
            bid=bid, ask=ask, mid=mid,
            iv_emp=d["iv"], iv_svi=iv_svi,
            fair_value=fair,
            edge_abs=edge_abs,
            edge_pct=edge_pct,
            recommendation=rec,
            spot=spot, risk_free=r, dividend_yield=q, t_years=t,
        ))

    return results, svi


def save_mispricing(rows: list[MispricingRow], path: Path) -> None:
    """Salva mispricing rows em CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))


def main():
    """Varre todas as chains em data/*/ e gera mispricing."""
    root = Path(__file__).parent.parent
    data = root / "data"
    out = root / "models" / "scan"
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_svi = {}

    for under_dir in sorted(data.iterdir()):
        if not under_dir.is_dir():
            continue
        underlying = under_dir.name
        for csv_path in sorted(under_dir.glob("chain_*.csv")):
            try:
                rows, svi = scan_chain(csv_path)
            except Exception as e:
                print(f"  ✗ {csv_path.name}: {e}")
                continue
            if not rows:
                continue
            # salva por chain
            exp = rows[0].expiration
            out_path = out / f"{underlying}_{exp}.csv"
            save_mispricing(rows, out_path)
            all_results.extend(rows)
            all_svi[f"{underlying}_{exp}"] = svi
            # top 3 edges
            top = sorted(rows, key=lambda r: abs(r.edge_pct), reverse=True)[:3]
            print(f"\n=== {underlying} {exp} ({len(rows)} séries) ===")
            print(f"  SVI: a={svi.a:.4f} b={svi.b:.4f} ρ={svi.rho:.3f} m={svi.m:.3f} σ={svi.sigma:.3f}")
            for r in top:
                print(f"  {'★' if r.recommendation != 'skip' else ' '} "
                      f"{r.symbol:18s} {r.side:4s} K={r.strike:>6.2f} "
                      f"mid={r.mid:6.3f} fair={r.fair_value:6.3f} "
                      f"edge={r.edge_pct:+5.1f}% {r.recommendation}")

    # salva tudo consolidado
    if all_results:
        consolidated = out / "all_mispricing.csv"
        save_mispricing(all_results, consolidated)
        print(f"\n→ {consolidated} ({len(all_results)} rows total)")

        # stats
        edges = [r.edge_pct for r in all_results]
        candidates = [r for r in all_results if r.recommendation != "skip"]
        print(f"\n=== resumo ===")
        print(f"  total séries: {len(all_results)}")
        print(f"  edge médio: {np.mean(edges):+.2f}%")
        print(f"  edge std:   {np.std(edges):.2f}%")
        print(f"  candidatos (|edge|>{5}%): {len(candidates)}")


if __name__ == "__main__":
    main()