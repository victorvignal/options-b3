"""
Edge realized backtest — call only.

A pergunta: SE eu comprar uma call com edge>5% no dia D, qual o PnL em D+1?

Pipeline (sem look-ahead):
    Para cada dia D na janela:
        1. Calibra SVI usando chain_call_only de D-1 (calls europeias, moneyness 0.95-1.20)
        2. Calcula fair value de cada call em D usando IV SVI + spot/r/q de D
        3. Compara fair vs mid de D → edge previsto
        4. SE edge > threshold (default 5%), marca como "sinal de compra"
        5. Em D+1: pega mid de mercado da mesma série
        6. PnL bruto = (mid_D+1 - mid_D) / mid_D
        7. PnL líquido = PnL bruto - spread_pct em D (custo de transação)

Sem look-ahead: D-1 calibra modelo, D prevê, D+1 mede realizado.
Brapi EOD: D+1 só é usado se a brapi confirma snapshot daquele dia
(campo `date` da série == data pedida) — senão a brapi retorna stale
e a gente pula pra evitar look-ahead disfarçado.
"""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np

from src.chain_ingest import (
    fetch_chain_at_date,
    fetch_analytics_at_date,
    get_underlying,
    get_risk_free_curve,
    interpolate_risk_free,
    filter_call_liquid,
)
from src.svi import SVIParams, calibrate_svi
from src.bs import price as bs_price


ROOT = Path(__file__).parent.parent
EDGES_DIR = ROOT / "models" / "edges"
EDGES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EdgeTrade:
    """Um trade simulado: compra no dia D com edge>threshold, fecha em D+1."""
    signal_date: str
    close_date: str
    underlying: str
    expiration: str
    symbol_d: str
    strike: float
    side: Literal["call"]
    spot_d: float
    spot_d1: float

    mid_d: float
    mid_d1: float
    bid_d: float
    ask_d: float

    iv_pred_d: float
    iv_real_d: float
    fair_d: float

    edge_pct: float
    spread_pct: float

    pnl_bruto_pct: float
    pnl_liquido_pct: float


@dataclass
class EdgeRunConfig:
    underlying: str
    expiration: str
    n_days: int = 60
    edge_threshold_pct: float = 5.0
    moneyness_min: float = 0.95
    moneyness_max: float = 1.20
    oi_min: int = 100
    max_spread_pct: float = 30.0   # descarta séries com spread > X% (mercado seco)


def _trading_days(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _calibrate_svi_on_calls(chain_d: list[dict], analytics_d: list[dict],
                             spot: float, r: float, q: float, t: float,
                             moneyness_min: float, moneyness_max: float,
                             oi_min: int) -> SVIParams | None:
    filtered = filter_call_liquid(chain_d, spot,
                                  moneyness_min=moneyness_min,
                                  moneyness_max=moneyness_max,
                                  oi_min=oi_min)
    iv_brapi: dict[float, float] = {}
    for a in analytics_d:
        if a.get("side") != "call":
            continue
        st = a.get("strike")
        iv = a.get("impliedVolatility")
        if st is None or iv is None or iv <= 0:
            continue
        if a.get("confidence") == "none":
            continue
        iv_brapi[float(st)] = iv

    forward = spot * np.exp((r - q) * t)
    strikes, ivs = [], []
    for s in filtered:
        strike = float(s.get("strike") or 0)
        if strike not in iv_brapi:
            continue
        iv = iv_brapi[strike]
        if iv <= 0.10 or iv > 1.50:
            continue
        strikes.append(strike)
        ivs.append(iv)

    if len(strikes) < 5:
        return None
    return calibrate_svi(strikes, ivs, t, forward=forward)


def _find_next_real_snapshot(underlying: str, expiration: str,
                              start_d: date, token: str,
                              max_offset: int = 7) -> tuple[date, list[dict], float] | None:
    """
    Acha próximo dia útil >= start_d+1 onde brapi tem snapshot REAL
    (campo `date` da série == data pedida). Retorna (data, chain, spot).
    Se não achar dentro de max_offset dias, retorna None.
    """
    for offset in range(1, max_offset + 1):
        candidate = start_d + timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue
        cand_str = candidate.isoformat()
        try:
            cand_chain = fetch_chain_at_date(underlying, expiration, cand_str, token)
        except Exception:
            continue
        if not cand_chain:
            continue
        sample_date = cand_chain[0].get("date")
        if not isinstance(sample_date, int):
            continue
        cand_date = datetime.fromtimestamp(sample_date).date()
        if cand_date != candidate:
            continue  # brapi retornou snapshot stale (EOD do último pregão)
        try:
            u = get_underlying(underlying, token)
            spot = u["spot"]
        except Exception:
            return None
        return (candidate, cand_chain, spot)
    return None


def walk_forward_one_step(cfg: EdgeRunConfig, day_d: date, token: str) -> list[EdgeTrade]:
    day_prev = day_d - timedelta(days=1)
    while day_prev.weekday() >= 5:
        day_prev -= timedelta(days=1)

    dp_str = day_prev.isoformat()
    d_str = day_d.isoformat()

    try:
        chain_prev = fetch_chain_at_date(cfg.underlying, cfg.expiration, dp_str, token)
        analytics_prev = fetch_analytics_at_date(cfg.underlying, cfg.expiration, dp_str, token)
        u_prev = get_underlying(cfg.underlying, token)
        curve = get_risk_free_curve(token)
    except Exception:
        return []

    spot_prev = u_prev["spot"]
    q_prev = u_prev["dividend_yield"]

    exp_dt = datetime.fromisoformat(cfg.expiration).date()
    t_prev = max((exp_dt - day_prev).days / 365.0, 1 / 365.0)
    r_prev = interpolate_risk_free(curve, t_prev)

    svi = _calibrate_svi_on_calls(
        chain_prev, analytics_prev,
        spot_prev, r_prev, q_prev, t_prev,
        cfg.moneyness_min, cfg.moneyness_max, cfg.oi_min,
    )
    if svi is None:
        return []

    try:
        chain_d = fetch_chain_at_date(cfg.underlying, cfg.expiration, d_str, token)
        analytics_d = fetch_analytics_at_date(cfg.underlying, cfg.expiration, d_str, token)
        u_d = get_underlying(cfg.underlying, token)
    except Exception:
        return []

    spot_d = u_d["spot"]
    q_d = u_d["dividend_yield"]
    t_d = max((exp_dt - day_d).days / 365.0, 1 / 365.0)
    r_d = interpolate_risk_free(curve, t_d)
    forward_d = spot_d * np.exp((r_d - q_d) * t_d)

    iv_real_d: dict[float, float] = {}
    for a in analytics_d:
        if a.get("side") != "call":
            continue
        st = a.get("strike")
        iv = a.get("impliedVolatility")
        if st is None or iv is None or iv <= 0:
            continue
        if a.get("confidence") == "none":
            continue
        iv_real_d[float(st)] = iv

    candidates = []
    for s in chain_d:
        if s.get("optionStyle") != "european":
            continue
        if s.get("side") != "call":
            continue
        strike = float(s.get("strike") or 0)
        if strike <= 0 or spot_d <= 0:
            continue
        moneyness = strike / spot_d
        if moneyness < cfg.moneyness_min or moneyness > cfg.moneyness_max:
            continue

        bid = s.get("bid") or 0
        ask = s.get("ask") or 0
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        if mid <= 0:
            continue
        spread_pct = (ask - bid) / mid * 100
        if spread_pct > cfg.max_spread_pct:
            continue
        oi = s.get("openInterest") or 0
        vol = s.get("volume") or 0
        if vol == 0 and oi < cfg.oi_min:
            continue

        iv_r = iv_real_d.get(strike, 0.0)
        k = float(np.log(strike / forward_d))
        iv_pred = svi.iv(k, t_d)
        fair = bs_price(spot_d, strike, t_d, r_d, q_d, iv_pred, "call")
        edge_pct = (fair - mid) / mid * 100

        if edge_pct > cfg.edge_threshold_pct:
            candidates.append((s, strike, mid, bid, ask, iv_pred, iv_r, fair,
                              edge_pct, spread_pct))

    if not candidates:
        return []

    # D+1 com snapshot real (não stale)
    next_data = _find_next_real_snapshot(cfg.underlying, cfg.expiration, day_d, token)
    if next_data is None:
        return []
    dn_date, chain_next, spot_d1 = next_data
    dn_str = dn_date.isoformat()

    mid_d1_by_strike: dict[float, float] = {}
    for s in chain_next:
        if s.get("side") != "call":
            continue
        st = float(s.get("strike") or 0)
        bid = s.get("bid") or 0
        ask = s.get("ask") or 0
        if bid > 0 and ask > 0:
            mid_d1_by_strike[st] = (bid + ask) / 2

    trades = []
    for (s, strike, mid_d, bid_d, ask_d, iv_pred, iv_r, fair_d,
         edge_pct, spread_pct) in candidates:
        mid_d1 = mid_d1_by_strike.get(strike)
        if mid_d1 is None:
            continue
        pnl_bruto = (mid_d1 - mid_d) / mid_d * 100
        pnl_liquido = pnl_bruto - spread_pct
        trades.append(EdgeTrade(
            signal_date=d_str,
            close_date=dn_str,
            underlying=cfg.underlying,
            expiration=cfg.expiration,
            symbol_d=s.get("symbol", ""),
            strike=strike,
            side="call",
            spot_d=spot_d,
            spot_d1=spot_d1,
            mid_d=mid_d,
            mid_d1=mid_d1,
            bid_d=bid_d,
            ask_d=ask_d,
            iv_pred_d=iv_pred,
            iv_real_d=iv_r,
            fair_d=fair_d,
            edge_pct=edge_pct,
            spread_pct=spread_pct,
            pnl_bruto_pct=pnl_bruto,
            pnl_liquido_pct=pnl_liquido,
        ))
    return trades


def run_backtest(cfg: EdgeRunConfig, token: str) -> list[EdgeTrade]:
    today = date.today()
    start = today - timedelta(days=cfg.n_days + 10)
    trading_days = _trading_days(start, today)

    all_trades: list[EdgeTrade] = []
    print(f"=== edge realized {cfg.underlying} {cfg.expiration} ===")
    print(f"  janela: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} pregões)")
    print(f"  threshold: edge>{cfg.edge_threshold_pct}%, call only, "
          f"moneyness [{cfg.moneyness_min}, {cfg.moneyness_max}]")
    print()

    for i, day in enumerate(trading_days):
        if i < 2 or i >= len(trading_days) - 1:
            continue
        d_str = day.isoformat()
        print(f"  [{i}/{len(trading_days)-1}] {d_str}", end=" ... ", flush=True)
        try:
            trades = walk_forward_one_step(cfg, day, token)
            all_trades.extend(trades)
            print(f"{len(trades)} sinais")
        except Exception as e:
            print(f"✗ {e}")

    return all_trades


def compute_pnl_stats(trades: list[EdgeTrade]) -> dict:
    if not trades:
        return {"error": "sem trades"}
    pnls_bruto = np.array([t.pnl_bruto_pct for t in trades])
    pnls_liquido = np.array([t.pnl_liquido_pct for t in trades])
    n = len(trades)
    win_bruto = int(np.sum(pnls_bruto > 0))
    win_liquido = int(np.sum(pnls_liquido > 0))
    return {
        "n_trades": n,
        "pnl_bruto_medio_pct": float(np.mean(pnls_bruto)),
        "pnl_bruto_median_pct": float(np.median(pnls_bruto)),
        "pnl_bruto_total_pct": float(np.sum(pnls_bruto)),
        "pnl_bruto_std_pct": float(np.std(pnls_bruto)),
        "win_rate_bruto_pct": win_bruto / n * 100,
        "pnl_liquido_medio_pct": float(np.mean(pnls_liquido)),
        "pnl_liquido_median_pct": float(np.median(pnls_liquido)),
        "pnl_liquido_total_pct": float(np.sum(pnls_liquido)),
        "win_rate_liquido_pct": win_liquido / n * 100,
        "sharpe_bruto": float(np.mean(pnls_bruto) / np.std(pnls_bruto)) if np.std(pnls_bruto) > 0 else 0.0,
        "max_drawdown_bruto_pct": float(min(0, min(pnls_bruto))),
        "edge_pct_medio": float(np.mean([t.edge_pct for t in trades])),
        "spread_pct_medio": float(np.mean([t.spread_pct for t in trades])),
    }


def save_csv(trades: list[EdgeTrade], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trades:
        path.write_text("")
        return
    fields = list(asdict(trades[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in trades:
            w.writerow(asdict(t))


def main():
    """Variante D: rodar backtest com 3 thresholds de edge (10/15/20%).

    Hipótese: edge>5% gera sinais fracos que perdem. Sinais mais fortes
    (edge>10/15/20%) devem ter PnL realizado melhor.
    """
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        raise SystemExit("BRAPI_TOKEN não definido")

    print("=== edge realized — VARIANTE D: thresholds 10/15/20% ===")
    print("    call only, moneyness 0.95-1.20, spread<30%")
    print()

    bases = [
        EdgeRunConfig("PETR4", "2026-10-16", n_days=60),
        EdgeRunConfig("BBAS3", "2026-10-16", n_days=60),
        EdgeRunConfig("VALE3", "2026-10-16", n_days=60),
    ]
    thresholds = [10.0, 15.0, 20.0]

    summary_rows = []

    for thr in thresholds:
        print()
        print("#" * 70)
        print(f"#  THRESHOLD = edge > {thr}%")
        print("#" * 70)
        all_trades: list[EdgeTrade] = []
        for base in bases:
            cfg = EdgeRunConfig(
                base.underlying, base.expiration, n_days=base.n_days,
                edge_threshold_pct=thr,
                moneyness_min=base.moneyness_min, moneyness_max=base.moneyness_max,
                oi_min=base.oi_min, max_spread_pct=base.max_spread_pct,
            )
            trades = run_backtest(cfg, token)
            all_trades.extend(trades)

        if not all_trades:
            print(f"\n  threshold {thr}: nenhum trade gerado")
            summary_rows.append({"threshold": thr, "n_trades": 0})
            continue

        # CSV separado por threshold: D_10.csv, D_15.csv, D_20.csv
        out_path = EDGES_DIR / f"D_{int(thr)}.csv"
        save_csv(all_trades, out_path)

        stats = compute_pnl_stats(all_trades)
        pnl_med = stats["pnl_liquido_medio_pct"]
        pnl_anual = pnl_med * 240
        pnls = np.array([t.pnl_liquido_pct for t in all_trades])
        std = float(np.std(pnls))
        sharpe_liq = (pnl_med / std) if std > 0 else 0.0
        sharpe_anual = sharpe_liq * np.sqrt(240)

        bate_13 = "SIM" if pnl_anual > 13.0 else "nao"

        print(f"\n=== threshold {thr}% ===")
        print(f"  arquivo              : {out_path}")
        print(f"  n_trades             : {stats['n_trades']}")
        print(f"  pnl_bruto_medio_pct  : {stats['pnl_bruto_medio_pct']:+.2f}%")
        print(f"  pnl_liquido_medio_pct: {pnl_med:+.2f}%")
        print(f"  win_rate_liquido_pct : {stats['win_rate_liquido_pct']:.1f}%")
        print(f"  pnl_anualizado       : {pnl_anual:+.2f}%  (= médio × 240)")
        print(f"  sharpe_liquido       : {sharpe_liq:.2f}")
        print(f"  sharpe_anualizado    : {sharpe_anual:.2f}")
        print(f"  edge_pct_medio       : {stats['edge_pct_medio']:.1f}%")
        print(f"  spread_pct_medio     : {stats['spread_pct_medio']:.1f}%")
        print(f"  bate_13pct?          : {bate_13}")

        summary_rows.append({
            "threshold": thr,
            "n_trades": stats["n_trades"],
            "pnl_liquido_medio_pct": pnl_med,
            "pnl_bruto_medio_pct": stats["pnl_bruto_medio_pct"],
            "win_rate_liquido_pct": stats["win_rate_liquido_pct"],
            "pnl_liquido_anualizado_pct": pnl_anual,
            "sharpe_liquido": sharpe_liq,
            "sharpe_anualizado": sharpe_anual,
            "edge_pct_medio": stats["edge_pct_medio"],
            "spread_pct_medio": stats["spread_pct_medio"],
            "bate_13pct": bate_13,
            "csv": str(out_path),
        })

    # Resumo final
    print()
    print("#" * 70)
    print("#  RESUMO COMPARATIVO — Variante D")
    print("#" * 70)
    best = None
    best_anual = -1e9
    for row in summary_rows:
        if row.get("n_trades", 0) == 0:
            print(f"  threshold {row['threshold']}%: SEM TRADES")
            continue
        print(f"  threshold {row['threshold']}%: "
              f"n={row['n_trades']}, "
              f"pnl_liq_med={row['pnl_liquido_medio_pct']:+.2f}%, "
              f"win={row['win_rate_liquido_pct']:.1f}%, "
              f"anualizado={row['pnl_liquido_anualizado_pct']:+.2f}% "
              f"(bate13={row['bate_13pct']}), "
              f"sharpe_anual={row['sharpe_anualizado']:.2f}")
        if row['pnl_liquido_anualizado_pct'] > best_anual:
            best_anual = row['pnl_liquido_anualizado_pct']
            best = row
    if best is not None:
        print(f"\n  MELHOR THRESHOLD = {best['threshold']}% "
              f"(pnl_liq anualizado {best['pnl_liquido_anualizado_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
