"""
Edge realized backtest — Variante C: SVI rolling 5 dias.

A pergunta: SE eu comprar uma call com edge>5% no dia D, qual o PnL em D+1?

Variante C (vs A base):
    - SVI é calibrada usando MÉDIA ROLANTE de 5 pregões (D-5..D-1)
      em vez de só D-1. Isso dá SVI mais estável e sinal melhor,
      porque calibração em 1 dia só é ruidosa.

Pipeline (sem look-ahead):
    Para cada dia D na janela:
        1. Calibra SVI em cada um dos 5 pregões anteriores (D-5..D-1)
        2. SVI final = média dos params (a, b, rho, m, sigma) das 5 calibrações
        3. Calcula fair value de cada call em D usando IV SVI + spot/r/q de D
        4. Compara fair vs mid de D → edge previsto
        5. SE edge > threshold (default 5%), marca como "sinal de compra"
        6. Em D+1: pega mid de mercado da mesma série
        7. PnL bruto = (mid_D+1 - mid_D) / mid_D
        8. PnL líquido = PnL bruto - spread_pct em D (custo de transação)

Sem look-ahead: D-5..D-1 calibra modelo, D prevê, D+1 mede realizado.
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
    svi_rolling_window: int = 5   # Variante C: SVI em janela rolante de 5 pregões


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


def _calibrate_svi_rolling(
    cfg: EdgeRunConfig,
    day_d: date,
    token: str,
    curve: list[tuple[float, float]],
    cache: dict[tuple[date, str], tuple[list[dict], list[dict], dict, list[dict]]],
) -> tuple[SVIParams | None, int]:
    """
    Variante C: calibra SVI em cada um dos últimos N pregões (D-N..D-1) e
    retorna SVIParams com cada param = MÉDIA das N calibrações.

    Retorna (svi, n_usados) onde n_usados = quantas calibrações foram bem-sucedidas.
    Se TODAS falharem, retorna (None, 0).

    cache: dict chaveado por (date, expiration+underlying) que guarda tupla
    (chain, analytics, underlying_dict, [vazio]) já fetched nesse run,
    para não dobrar chamadas (ex: D-1 pode ser D de outro step).
    """
    exp_dt = datetime.fromisoformat(cfg.expiration).date()

    # Coleta os últimos N pregões anteriores a day_d (D-N..D-1)
    window_days: list[date] = []
    cur = day_d
    while len(window_days) < cfg.svi_rolling_window:
        cur -= timedelta(days=1)
        if cur.weekday() >= 5:
            continue
        window_days.append(cur)
    # agora window_days = [D-N, ..., D-2, D-1]

    calibrated: list[SVIParams] = []
    n_used = 0
    for d_calib in window_days:
        d_str = d_calib.isoformat()
        cache_key = (d_calib, cfg.underlying)
        cached = cache.get(cache_key)
        if cached is None:
            try:
                chain_c = fetch_chain_at_date(cfg.underlying, cfg.expiration, d_str, token)
                analytics_c = fetch_analytics_at_date(cfg.underlying, cfg.expiration, d_str, token)
                u_c = get_underlying(cfg.underlying, token)
            except Exception:
                continue
            cache[cache_key] = (chain_c, analytics_c, u_c, [])
        chain_c, analytics_c, u_c, _ = cache[cache_key]

        if not chain_c or not analytics_c:
            continue

        spot_c = u_c["spot"]
        q_c = u_c["dividend_yield"]
        t_c = max((exp_dt - d_calib).days / 365.0, 1 / 365.0)
        r_c = interpolate_risk_free(curve, t_c)

        svi_i = _calibrate_svi_on_calls(
            chain_c, analytics_c,
            spot_c, r_c, q_c, t_c,
            cfg.moneyness_min, cfg.moneyness_max, cfg.oi_min,
        )
        if svi_i is None:
            continue
        calibrated.append(svi_i)
        n_used += 1

    if not calibrated:
        return (None, 0)

    avg_a = float(np.mean([p.a for p in calibrated]))
    avg_b = float(np.mean([p.b for p in calibrated]))
    avg_rho = float(np.mean([p.rho for p in calibrated]))
    avg_m = float(np.mean([p.m for p in calibrated]))
    avg_sigma = float(np.mean([p.sigma for p in calibrated]))

    # Clamps leves pra manter região válida do SVI
    if avg_a < 0:
        avg_a = 0.0
    if avg_b < 0:
        avg_b = 0.0
    if avg_sigma <= 1e-6:
        avg_sigma = 1e-6
    if avg_rho > 0.999:
        avg_rho = 0.999
    if avg_rho < -0.999:
        avg_rho = -0.999

    return (SVIParams(a=avg_a, b=avg_b, rho=avg_rho, m=avg_m, sigma=avg_sigma), n_used)


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


def walk_forward_one_step(cfg: EdgeRunConfig, day_d: date, token: str,
                          curve: list[tuple[float, float]],
                          svi_cache: dict) -> list[EdgeTrade]:
    d_str = day_d.isoformat()

    try:
        chain_d = fetch_chain_at_date(cfg.underlying, cfg.expiration, d_str, token)
        analytics_d = fetch_analytics_at_date(cfg.underlying, cfg.expiration, d_str, token)
        u_d = get_underlying(cfg.underlying, token)
    except Exception:
        return []

    spot_d = u_d["spot"]
    q_d = u_d["dividend_yield"]
    exp_dt = datetime.fromisoformat(cfg.expiration).date()
    t_d = max((exp_dt - day_d).days / 365.0, 1 / 365.0)
    r_d = interpolate_risk_free(curve, t_d)

    # Variante C: SVI rolling 5 dias
    svi_result = _calibrate_svi_rolling(cfg, day_d, token, curve, svi_cache)
    svi, n_calib_used = svi_result
    if svi is None:
        return []

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

    try:
        curve = get_risk_free_curve(token)
    except Exception:
        curve = []

    # Cache compartilhado por underlying — chaves são (data, underlying)
    # Para não buscar a mesma chain/analytics várias vezes
    svi_cache: dict = {}

    all_trades: list[EdgeTrade] = []
    n_dias_usados = 0
    print(f"=== edge realized {cfg.underlying} {cfg.expiration} (variante C) ===")
    print(f"  janela: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} pregões)")
    print(f"  threshold: edge>{cfg.edge_threshold_pct}%, call only, "
          f"moneyness [{cfg.moneyness_min}, {cfg.moneyness_max}]")
    print(f"  SVI rolling window: {cfg.svi_rolling_window} pregões")
    print()

    # Para variante C, começamos em i >= svi_rolling_window+1 (precisa de N pregões anteriores)
    min_i = max(2, cfg.svi_rolling_window)
    for i, day in enumerate(trading_days):
        if i < min_i or i >= len(trading_days) - 1:
            continue
        d_str = day.isoformat()
        print(f"  [{i}/{len(trading_days)-1}] {d_str}", end=" ... ", flush=True)
        try:
            trades = walk_forward_one_step(cfg, day, token, curve, svi_cache)
            all_trades.extend(trades)
            n_dias_usados += 1
            print(f"{len(trades)} sinais")
        except Exception as e:
            print(f"✗ {e}")

    print(f"  (n_dias_usados para SVI rolling: {n_dias_usados})")
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
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        raise SystemExit("BRAPI_TOKEN não definido")

    print("=== edge realized — Variante C: SVI rolling 5 dias ===")
    print("    call only, moneyness 0.95-1.20")
    print()

    all_trades: list[EdgeTrade] = []
    configs = [
        EdgeRunConfig("PETR4", "2026-10-16", n_days=60,
                      edge_threshold_pct=5.0, max_spread_pct=30.0),
        EdgeRunConfig("BBAS3", "2026-10-16", n_days=60,
                      edge_threshold_pct=5.0, max_spread_pct=30.0),
        EdgeRunConfig("VALE3", "2026-10-16", n_days=60,
                      edge_threshold_pct=5.0, max_spread_pct=30.0),
    ]

    per_underlying_stats = {}
    for cfg in configs:
        trades = run_backtest(cfg, token)
        all_trades.extend(trades)
        if trades:
            stats = compute_pnl_stats(trades)
            per_underlying_stats[cfg.underlying] = stats
            print(f"\n  {cfg.underlying} {cfg.expiration}: {stats['n_trades']} trades")
            print(f"    PnL bruto  médio: {stats['pnl_bruto_medio_pct']:+.2f}%, "
                  f"total: {stats['pnl_bruto_total_pct']:+.1f}%, win: {stats['win_rate_bruto_pct']:.1f}%")
            print(f"    PnL líquido médio: {stats['pnl_liquido_medio_pct']:+.2f}%, "
                  f"total: {stats['pnl_liquido_total_pct']:+.1f}%, win: {stats['win_rate_liquido_pct']:.1f}%")
            print(f"    Sharpe bruto: {stats['sharpe_bruto']:.2f}, "
                  f"spread médio: {stats['spread_pct_medio']:.1f}%")
            print()

    if not all_trades:
        print("Nenhum trade gerado.")
        return

    out_path = EDGES_DIR / "C.csv"
    save_csv(all_trades, out_path)
    print(f"\n→ {out_path} ({len(all_trades)} trades)")

    print(f"\n=== resumo agregado ===")
    stats = compute_pnl_stats(all_trades)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Relatório final variante C
    pnls_liq = np.array([t.pnl_liquido_pct for t in all_trades])
    pnls_liq_aa = float(np.mean(pnls_liq)) * 240
    std = float(np.std(pnls_liq))
    sharpe_aa = float(np.mean(pnls_liq) / std) * np.sqrt(240) if std > 0 else 0.0
    n_per = {u: per_underlying_stats[u]["n_trades"] for u in per_underlying_stats}
    bate = "sim" if pnls_liq_aa > 13 else "não"
    print("\n=== RELATÓRIO VARIANTE C ===")
    print(f"  n_trades_total: {stats['n_trades']}")
    print(f"  n_trades_por_underlying: {n_per}")
    print(f"  pnl_liquido_medio_pct: {stats['pnl_liquido_medio_pct']:+.2f}%")
    print(f"  win_rate_liquido_pct: {stats['win_rate_liquido_pct']:.1f}%")
    print(f"  pnl_liquido_anualizado_pct: {pnls_liq_aa:+.2f}%")
    print(f"  sharpe_anualizado: {sharpe_aa:.2f}")
    print(f"  bate_13pct_aa: {bate}")
    print(f"  spread_medio_pct: {stats['spread_pct_medio']:.1f}%")


if __name__ == "__main__":
    main()
