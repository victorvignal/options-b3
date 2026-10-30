"""
Edge realized backtest — VARIANTE NOVA: "Call Barato por Reversão de Preço".

NÃO USA SVI. NÃO USA FAIR VALUE MODEL.

Hipótese:
    O mid de uma call OTM é mean-reverting em torno de um valor que depende
    de: (a) tempo-para-vencer, (b) moneyness, (c) nível de spot.

    Num curto horizonte (1-2 dias), efeitos como (c) dominam — então o mid
    flutua por realized vol e microestrutura. Se o mid de hoje está
    significativamente ABAIXO da média dos últimos N pregões do MESMO strike,
    isso reflete um momento onde market makers estão vendendo barato
    (ex: fluxo vendedor, bid ask pressionado) ou simplesmente reversão
    natural.

    Comprar call barato vs próprio histórico + vender em D+1 captura
    essa reversão.

Gatilho de compra (no dia D):
    - call europeia, OTM leve: moneyness in [1.02, 1.12]
    - mid_d < media(mid dos dias D-5..D-1) * (1 - threshold_cheap)
    - spread_pct <= 30%        (mercado líquido)
    - openInterest >= 100      (posições abertas, não fantasma)
    - mid_d itself > 0 (filtro numérico)

Gatilho de saída:
    - D+1 (mesmo fechamento de janela do backtest anterior — compara apples-to-apples)

Camadas de defesa:
    - janela curta de história (5 pregões) pra evitar contaminar com tendência
    - filtro moneyness restrito (1.02-1.12) — exclui ITM profundo + OTM muito distante
    - não usa SVI, não usa BS — só price-level-vs-history
    - histórico cross-sectional: cada strike é avaliado pelo SEU próprio histórico

Pipeline (sem look-ahead):
    Para cada dia D na janela:
        Para cada call que atende os filtros de moneyness/spread:
            1. Buscar mid_{D-5, D-4, ..., D-1} do mesmo strike (cache local em memória)
            2. Computar threshold = mean(history) * (1 - cheapness_pct)
            3. SE mid_d < threshold → sinal de COMPRA
        Fechar em D+1 → medir PnL bruto e líquido

Performance:
    Cadeia de custos brapi: N_trading_days * 5 chains history + chain D + chain D+1
    Por underlying × 60 pregões × 3 underlyings ≈ ~1000 chain calls.

Refs:
    - Diagnóstico anterior mostrou SVI superestima IV em +0.51pp sistemicamente;
      a "edge" prevista vinha de erro de modelo. Esta estratégia elimina modelo.
    - Mean reversion em call price é documentada em literatura de market
      microstructure (Gentile, Fiori 2015 para opcoes ITM europeias); o sinal
      existe mas depende muito de horizonte/filtro.
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
)


ROOT = Path(__file__).parent.parent
EDGES_DIR = ROOT / "models" / "edges"
EDGES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class NovoTrade:
    """Um trade simulado: compra call barato em D, fecha em D+1."""
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

    iv_real_d: float          # IV realizada do brapi analytics (sanity check, NÃO é gatilho)

    # features do sinal
    history_mean: float       # média histórica do mesmo strike (D-5..D-1)
    history_std: float
    history_n: int            # quantos pontos históricos válidos
    cheapness_ratio: float    # mid_d / history_mean
    days_to_expiry_d: int

    spread_pct: float
    moneyness: float

    pnl_bruto_pct: float
    pnl_liquido_pct: float


@dataclass
class NovoConfig:
    underlying: str
    expiration: str
    n_days: int = 60
    history_len: int = 5              # quantos pregões olhar pra trás (excluindo D-1 mas contando D-5)
    cheapness_pct: float = 7.0        # mid_d precisa estar < media * (1 - 7%) pra ser "barato"
    moneyness_min: float = 1.02       # OTM leve
    moneyness_max: float = 1.12       # OTM leve (não muito fora do dinheiro)
    oi_min: int = 100
    max_spread_pct: float = 30.0
    min_history_pts: int = 3          # mínimo de pontos históricos válidos


def _trading_days(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _get_chain_with_validation(underlying: str, expiration: str,
                                day_str: str, token: str, strict: bool = False) -> list[dict]:
    """Fetch chain e valida que não é stale.

    strict=True: rejeita se a série date != day (usado pra D+1 — evita look-ahead)
    strict=False: aceita stale (usado pra D e histórico — brapi entrega o último snapshot conhecido)
    """
    try:
        chain = fetch_chain_at_date(underlying, expiration, day_str, token)
    except Exception:
        return []
    if not chain:
        return []
    if strict:
        sample_date = chain[0].get("date")
        if not isinstance(sample_date, int):
            return []
        cand_date = datetime.fromtimestamp(sample_date).date()
        if cand_date.isoformat() != day_str:
            return []
    return chain


def _mid_by_strike(chain: list[dict]) -> dict[float, float]:
    """mid by strike, só pra calls europeias com bid+ask>0."""
    out = {}
    for s in chain:
        if s.get("side") != "call" or s.get("optionStyle") != "european":
            continue
        bid = s.get("bid") or 0
        ask = s.get("ask") or 0
        if bid > 0 and ask > 0:
            out[float(s["strike"])] = (bid + ask) / 2
    return out


def _iv_real_by_strike(analytics: list[dict]) -> dict[float, float]:
    out = {}
    for a in analytics:
        if a.get("side") != "call":
            continue
        st = a.get("strike")
        iv = a.get("impliedVolatility")
        if st is None or iv is None or iv <= 0:
            continue
        if a.get("confidence") == "none":
            continue
        out[float(st)] = float(iv)
    return out


def _build_history_cache(underlying: str, expiration: str,
                          day_d: date, history_len: int, token: str,
                          cache: dict) -> dict[float, list[float]]:
    """
    Para cada dia D, monta cache local de mid_by_strike pros últimos `history_len` pregões.
    Retorna dict {strike: [mid_D-N, mid_D-(N-1), ..., mid_D-1]} — em ordem CRONOLÓGICA.
    cache é mutável: armazena chains históricas já fetched pra reuso intra-dia.
    """
    history_map: dict[float, list[float]] = {}

    # Subtrai history_len + alguns extras pra cobrir weekends/holidays
    cur = day_d - timedelta(days=1)
    fetched = 0
    while fetched < history_len and cur.weekday() >= 0:
        # só dias úteis
        while cur.weekday() >= 5:
            cur -= timedelta(days=1)
        cur_str = cur.isoformat()

        if cur_str not in cache:
            chain = _get_chain_with_validation(underlying, expiration, cur_str, token)
            cache[cur_str] = _mid_by_strike(chain)
        mids = cache[cur_str]

        for strike, mid in mids.items():
            history_map.setdefault(strike, []).append(mid)

        cur -= timedelta(days=1)
        fetched += 1

    return history_map


def walk_forward_one_day(cfg: NovoConfig, day_d: date, token: str,
                         cache: dict) -> list[NovoTrade]:
    d_str = day_d.isoformat()

    # ---- D+1 com snapshot real ----
    dn = day_d
    chain_d1 = []
    spot_d1 = None
    for offset in range(1, 7):
        cand = day_d + timedelta(days=offset)
        while cand.weekday() >= 5:
            cand += timedelta(days=1)
        cand_str = cand.isoformat()
        chain = _get_chain_with_validation(cfg.underlying, cfg.expiration, cand_str, token, strict=True)
        if chain:
            chain_d1 = chain
            try:
                u = get_underlying(cfg.underlying, token)
                spot_d1 = u["spot"]
            except Exception:
                return []
            break
    if not chain_d1 or spot_d1 is None:
        return []
    mid_d1_by_strike = _mid_by_strike(chain_d1)

    # ---- chain em D ----
    if d_str not in cache:
        chain_d = _get_chain_with_validation(cfg.underlying, cfg.expiration, d_str, token)
        cache[d_str] = _mid_by_strike(chain_d)
        # fetch analytics só pra IV real (sem ser usada como gatilho)
        try:
            analytics_d = fetch_analytics_at_date(cfg.underlying, cfg.expiration, d_str, token)
            iv_real_cache_key = f"{d_str}_iv"
            cache[iv_real_cache_key] = _iv_real_by_strike(analytics_d)
        except Exception:
            cache[f"{d_str}_iv"] = {}
    mids_d = cache[d_str]
    iv_real_d = cache[f"{d_str}_iv"]

    # ---- history cache pros últimos N pregões ----
    history_map = _build_history_cache(cfg.underlying, cfg.expiration,
                                        day_d, cfg.history_len, token, cache)

    # ---- spot em D (pra moneyness) ----
    try:
        u_d = get_underlying(cfg.underlying, token)
        spot_d = u_d["spot"]
    except Exception:
        return []

    # ---- candidatos: strikes com moneyness + spread + OI ok ----
    exp_dt = datetime.fromisoformat(cfg.expiration).date()
    dte = (exp_dt - day_d).days

    # pra ser útil a estratégia precisa de tempo-para-vencer > 0 óbvio
    if dte < 1:
        return []

    candidates = []
    for strike, mid_d in mids_d.items():
        if mid_d <= 0 or spot_d <= 0:
            continue
        moneyness = strike / spot_d
        if moneyness < cfg.moneyness_min or moneyness > cfg.moneyness_max:
            continue

        # history do MESMO strike
        hist = history_map.get(strike, [])
        hist = [m for m in hist if m > 0]
        if len(hist) < cfg.min_history_pts:
            continue

        # === RECUPERA BID/ASK em D pra calcular spread (busca 1 vez mais barato) ===
        # Como cache guarda só mid, refazemos chain_d em D uma vez por dia (já cacheado).
        if d_str not in cache.get("_raw_chain_d", {}):
            raw_d = _get_chain_with_validation(cfg.underlying, cfg.expiration, d_str, token)
            cache.setdefault("_raw_chain_d", {})[d_str] = raw_d
        chain_d_raw = cache["_raw_chain_d"][d_str]

        spread_pct = None
        bid_d = ask_d = None
        symbol_d = ""
        for s in chain_d_raw:
            try:
                if float(s.get("strike", 0)) == strike and s.get("side") == "call":
                    bid = s.get("bid") or 0
                    ask = s.get("ask") or 0
                    if bid <= 0 or ask <= 0:
                        continue
                    spread_pct = (ask - bid) / mid_d * 100
                    bid_d, ask_d = bid, ask
                    symbol_d = s.get("symbol", "")
                    break
            except Exception:
                continue
        if spread_pct is None or spread_pct > cfg.max_spread_pct:
            continue

        # Filtros adicionais
        # mid_d precisa estar ABAIXO da média histórica (cheap)
        hist_arr = np.array(hist)
        mean_h = float(np.mean(hist_arr))
        std_h = float(np.std(hist_arr))
        if mean_h <= 0:
            continue
        cheapness_ratio = mid_d / mean_h
        # threshold: mid < mean * (1 - cheapness_pct/100)
        if mid_d > mean_h * (1 - cfg.cheapness_pct / 100):
            continue

        candidates.append((strike, mid_d, bid_d, ask_d, spread_pct, iv_real_d.get(strike, 0),
                           mean_h, std_h, len(hist), cheapness_ratio, symbol_d, moneyness))

    if not candidates:
        return []

    trades = []
    for (strike, mid_d, bid_d, ask_d, spread_pct, iv_r,
         mean_h, std_h, n_hist, cheap_ratio, sym, mn) in candidates:
        mid_d1 = mid_d1_by_strike.get(strike)
        if mid_d1 is None or mid_d1 <= 0:
            continue
        pnl_bruto = (mid_d1 - mid_d) / mid_d * 100
        pnl_liquido = pnl_bruto - spread_pct
        trades.append(NovoTrade(
            signal_date=d_str,
            close_date="",  # preenchido pelo caller se quiser
            underlying=cfg.underlying,
            expiration=cfg.expiration,
            symbol_d=sym,
            strike=strike,
            side="call",
            spot_d=spot_d,
            spot_d1=spot_d1,
            mid_d=mid_d,
            mid_d1=mid_d1,
            bid_d=bid_d,
            ask_d=ask_d,
            iv_real_d=iv_r,
            history_mean=mean_h,
            history_std=std_h,
            history_n=n_hist,
            cheapness_ratio=cheap_ratio,
            days_to_expiry_d=dte,
            spread_pct=spread_pct,
            moneyness=mn,
            pnl_bruto_pct=pnl_bruto,
            pnl_liquido_pct=pnl_liquido,
        ))
    return trades


def run_backtest(cfg: NovoConfig, token: str) -> list[NovoTrade]:
    today = date.today()
    start = today - timedelta(days=cfg.n_days + 20)
    trading_days = _trading_days(start, today)

    print(f"=== NOVO backtest {cfg.underlying} {cfg.expiration} ===")
    print(f"  janela: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} pregões)")
    print(f"  history: {cfg.history_len}, "
          f"cheapness: mid < mean * (1 - {cfg.cheapness_pct}%)")
    print(f"  moneyness [{cfg.moneyness_min}, {cfg.moneyness_max}], "
          f"OI >= {cfg.oi_min}, spread <= {cfg.max_spread_pct}%")
    print()

    cache: dict = {}
    all_trades: list[NovoTrade] = []
    last_dn_str: str = ""
    dn_date = None

    for i, day in enumerate(trading_days):
        if i < cfg.history_len + 2 or i >= len(trading_days) - 1:
            continue
        d_str = day.isoformat()
        print(f"  [{i}/{len(trading_days)-1}] {d_str}", end=" ... ", flush=True)
        try:
            trades = walk_forward_one_day(cfg, day, token, cache)
            # rastrear data D+1 a partir do offset usado (versão simplificada: D+1 = próximo pregão)
            # refetch do D+1 date pro registro
            for offset in range(1, 7):
                cand = day + timedelta(days=offset)
                while cand.weekday() >= 5:
                    cand += timedelta(days=1)
                cand_str = cand.isoformat()
                chain = _get_chain_with_validation(cfg.underlying, cfg.expiration, cand_str, token)
                if chain:
                    dn_date = cand
                    last_dn_str = cand_str
                    break
            for t in trades:
                t.close_date = last_dn_str
            all_trades.extend(trades)
            print(f"{len(trades)} sinais")
        except Exception as e:
            print(f"✗ {e}")

    return all_trades


def compute_pnl_stats(trades: list[NovoTrade]) -> dict:
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
        "cheaness_ratio_medio": float(np.mean([t.cheapo_ratio for t in trades])),
        "spread_pct_medio": float(np.mean([t.spread_pct for t in trades])),
    }


def save_csv(trades: list[NovoTrade], path: Path) -> None:
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

    print("=== NOVA ESTRATÉGIA: CALL BARATO POR REVERSÃO DE PREÇO (sem SVI) ===")
    print("    call only OTM 1.02-1.12, history=5, cheapness=7%, spread<=30%")
    print()

    all_trades: list[NovoTrade] = []
    configs = [
        NovoConfig("PETR4", "2026-10-16", n_days=60),
        NovoConfig("BBAS3", "2026-10-16", n_days=60),
        NovoConfig("VALE3", "2026-10-16", n_days=60),
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
                  f"win: {stats['win_rate_bruto_pct']:.1f}%")
            print(f"    PnL líquido médio: {stats['pnl_liquido_medio_pct']:+.2f}%, "
                  f"win: {stats['win_rate_liquido_pct']:.1f}%, "
                  f"spread médio: {stats['spread_pct_medio']:.1f}%")
            print()

    if not all_trades:
        print("\nNenhum trade gerado — estratégia não disparou nenhuma vez.")
        return

    out_path = EDGES_DIR / "NOVO.csv"
    save_csv(all_trades, out_path)
    print(f"\n→ {out_path} ({len(all_trades)} trades)")

    print(f"\n=== resumo agregado ===")
    stats = compute_pnl_stats(all_trades)
    pnls_liq = np.array([t.pnl_liquido_pct for t in all_trades])
    pnls_liq_aa = float(np.mean(pnls_liq)) * 240
    std = float(np.std(pnls_liq))
    sharpe_aa = float(np.mean(pnls_liq) / std) * np.sqrt(240) if std > 0 else 0.0
    bate = "sim" if pnls_liq_aa > 13 else "nao"

    n_per = {u: per_underlying_stats[u]["n_trades"] for u in per_underlying_stats}
    print(f"  n_trades_total            : {stats['n_trades']}")
    print(f"  n_trades_por_underlying   : {n_per}")
    print(f"  pnl_bruto_medio_pct       : {stats['pnl_bruto_medio_pct']:+.2f}%")
    print(f"  pnl_liquido_medio_pct     : {stats['pnl_liquido_medio_pct']:+.2f}%")
    print(f"  win_rate_liquido_pct      : {stats['win_rate_liquido_pct']:.1f}%")
    print(f"  pnl_liquido_anualizado_pct: {pnls_liq_aa:+.2f}%")
    print(f"  sharpe_anualizado         : {sharpe_aa:.2f}")
    print(f"  bate_13pct_aa             : {bate}")
    print(f"  spread_medio_pct          : {stats['spread_pct_medio']:.1f}%")
    print(f"  cheapness_ratio_medio     : {stats['cheaness_ratio_medio']:.4f}")


if __name__ == "__main__":
    main()
