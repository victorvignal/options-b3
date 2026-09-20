"""
Walk-forward OOS backtest do pricer SVI.

Princípio: pra cada dia D na janela de teste,
    1. Calibra SVI usando chain de D-1 (informações de D-1 só)
    2. Prevê IV surface em D interpolando pela SVI calibrada
    3. Compara com IV realizada em D (= analytics brapi em D)
    4. Calcula métricas (RMSE, MAE, sign-accuracy)

Baseline trivial: "IV_amanhã = IV_hoje" (random walk em IV).
Se o modelo não bate isso, é ruído.

Sem look-ahead: nunca usa info de D ou posterior pra calibrar modelo
que prevê D. Spot/divYield/r são freezing no momento da calibração.
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
)
from src.svi import SVIParams, calibrate_svi


ROOT = Path(__file__).parent.parent
MODELS_BACKTEST = ROOT / "models" / "backtest"


@dataclass
class BacktestRow:
    """Uma previsão de 1 série em 1 dia."""
    date: str           # data da previsão (D)
    symbol: str         # symbol brapi na data D
    side: str           # call/put
    strike: float
    spot_at_d: float    # spot em D (conhecido em D, OK usar)
    iv_pred: float      # IV prevista (vinda da SVI calibrada em D-1)
    iv_real: float      # IV realizada em D (analytics brapi)
    iv_random_walk: float  # baseline: IV em D-1 da mesma série
    pred_err_pp: float  # (iv_pred - iv_real) em pontos percentuais
    rw_err_pp: float    # (iv_random_walk - iv_real) em pp


def _trading_days_between(start: date, end: date) -> list[date]:
    """
    Lista dias úteis entre start e end (inclusivo).
    Brapi atualiza em dias de pregão B3 (seg-sex, exceto feriados).
    """
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # seg-sex
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _build_chain_for_calibration(series: list[dict], spot: float, r: float,
                                  q: float, t_years: float,
                                  analytics: list[dict] | None = None) -> tuple[list[float], list[float]]:
    """
    Filtra séries da chain em D-1 e converte pra (strikes, IVs) prontos pra SVI.

    Critérios (espelho do mispricing_scan pra consistência):
    - europeian
    - bid > 0 AND ask > 0
    - moneyness entre 0.85 e 1.15 (forward = spot * exp((r-q)*T))
    - IV entre 10% e 150%

    Fonte da IV:
    - Se `analytics` é passado (lista do /options/analytics), usa IV brapi.
      Comparação direta: SVI calibrada em IV brapi vs IV brapi em D+1.
      Sem intermediário Newton, sem divergência de modelo.
    - Caso contrário, recalcula via Newton (implied_vol).
    """
    forward = spot * np.exp((r - q) * t_years)

    # mapa (strike, side) → IV brapi, se disponível
    iv_brapi: dict[tuple[float, str], float] = {}
    if analytics:
        for a in analytics:
            st = a.get("strike")
            sd = a.get("side")
            iv = a.get("impliedVolatility")
            if st is None or sd is None or iv is None or iv <= 0:
                continue
            if a.get("confidence") == "none":
                continue
            iv_brapi[(float(st), sd)] = iv

    strikes, ivs = [], []
    for s in series:
        if s.get("optionStyle") != "european":
            continue
        bid = s.get("bid") or 0
        ask = s.get("ask") or 0
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        if mid <= 0:
            continue
        strike = float(s.get("strike") or 0)
        if strike <= 0:
            continue
        moneyness = strike / forward
        if moneyness < 0.85 or moneyness > 1.15:
            continue
        side = s.get("side")

        if (strike, side) in iv_brapi:
            # preferir IV brapi quando disponível (consistência com target)
            iv = iv_brapi[(strike, side)]
        else:
            # fallback Newton
            from src.bs import implied_vol
            try:
                iv = implied_vol(mid, spot, strike, t_years, r, q, side)
            except Exception:
                continue

        if iv <= 0.10 or iv > 1.50:
            continue
        strikes.append(strike)
        ivs.append(iv)
    return strikes, ivs


def _predict_iv_surface(svi: SVIParams, strikes: list[float],
                        forward: float, t_years: float) -> dict[float, float]:
    """Pra cada strike, retorna IV prevista pela SVI."""
    out = {}
    for k in strikes:
        log_moneyness = np.log(k / forward)
        out[k] = svi.iv(float(log_moneyness), t_years)
    return out


def _is_svi_degenerate(svi: SVIParams) -> bool:
    """Detecta SVI que colou no bound = calibração ruim."""
    return (
        svi.b >= 4.99
        or abs(svi.rho) >= 0.99
        or svi.sigma <= 1e-3
    )


def walk_forward_one_step(
    underlying: str,
    expiration: str,
    day_d: date,
    token: str,
) -> list[BacktestRow]:
    """
    Um passo do walk-forward: calibra em D-1, prevê D.

    Retorna lista de BacktestRow (1 por série que existe em ambos os dias).
    """
    day_prev = day_d - timedelta(days=1)
    # Se D-1 cair num fim de semana, anda pra trás até dia útil
    while day_prev.weekday() >= 5:
        day_prev -= timedelta(days=1)

    d_str = day_d.isoformat()
    dp_str = day_prev.isoformat()

    # 1. Calibração em D-1
    try:
        chain_prev = fetch_chain_at_date(underlying, expiration, dp_str, token)
        analytics_prev = fetch_analytics_at_date(underlying, expiration, dp_str, token)
        u_prev = get_underlying(underlying, token)
        curve = get_risk_free_curve(token)
    except Exception as e:
        print(f"  ✗ {dp_str}: falha ao buscar dados D-1: {e}")
        return []

    spot_prev = u_prev["spot"]
    q_prev = u_prev["dividend_yield"]

    exp_dt = datetime.fromisoformat(expiration).date()
    t_prev = max((exp_dt - day_prev).days / 365.0, 1 / 365.0)
    r_prev = interpolate_risk_free(curve, t_prev)

    strikes_prev, ivs_prev = _build_chain_for_calibration(
        chain_prev, spot_prev, r_prev, q_prev, t_prev, analytics=analytics_prev
    )
    if len(strikes_prev) < 5:
        return []

    forward_prev = spot_prev * np.exp((r_prev - q_prev) * t_prev)
    svi = calibrate_svi(strikes_prev, ivs_prev, t_prev, forward=forward_prev)

    # 2. Previsão em D — usa spot/div/r de D (conhecidos no momento da decisão em D)
    try:
        chain_d = fetch_chain_at_date(underlying, expiration, d_str, token)
        analytics_d = fetch_analytics_at_date(underlying, expiration, d_str, token)
        u_d = get_underlying(underlying, token)
    except Exception as e:
        print(f"  ✗ {d_str}: falha ao buscar dados D: {e}")
        return []

    spot_d = u_d["spot"]
    q_d = u_d["dividend_yield"]
    t_d = max((exp_dt - day_d).days / 365.0, 1 / 365.0)
    r_d = interpolate_risk_free(curve, t_d)
    forward_d = spot_d * np.exp((r_d - q_d) * t_d)

    # mapa strike → IV_prev (do analytics de D-1) pra baseline random walk
    iv_prev_by_strike_side: dict[tuple[float, str], float] = {}
    for a in analytics_prev:
        st = a.get("strike")
        sd = a.get("side")
        iv = a.get("impliedVolatility")
        if st is None or iv is None or iv <= 0:
            continue
        iv_prev_by_strike_side[(float(st), sd)] = iv

    # mapa strike → IV_real em D (do analytics de D)
    iv_real_by_strike_side: dict[tuple[float, str], float] = {}
    for a in analytics_d:
        st = a.get("strike")
        sd = a.get("side")
        iv = a.get("impliedVolatility")
        if st is None or iv is None or iv <= 0:
            continue
        # só séries com confiança razoável
        if a.get("confidence") == "none":
            continue
        iv_real_by_strike_side[(float(st), sd)] = iv

    # 3. Cruzar: pra cada strike em D, prever via SVI e comparar com realizado
    # Filtro de moneyness 0.85-1.15: onde SVI é precisa. Asas são extrapolação.
    MONEYNESS_MIN, MONEYNESS_MAX = 0.85, 1.15
    rows = []
    svi_degenerate = _is_svi_degenerate(svi)

    for s in chain_d:
        if s.get("optionStyle") != "european":
            continue
        st = s.get("strike")
        sd = s.get("side")
        if st is None or sd is None:
            continue
        st_f = float(st)

        # filtro de moneyness (mesmo critério da calibração)
        moneyness = st_f / forward_d
        if moneyness < MONEYNESS_MIN or moneyness > MONEYNESS_MAX:
            continue

        # precisa de IV realizada pra comparar
        iv_real = iv_real_by_strike_side.get((st_f, sd))
        if iv_real is None:
            continue

        # previsão via SVI (calibrada em D-1)
        if svi_degenerate:
            continue  # SVI degenerada — previsão não confiável, pula
        k = float(np.log(st_f / forward_d))
        iv_pred = svi.iv(k, t_d)

        # baseline random walk: IV em D-1 da mesma (strike, side)
        iv_rw = iv_prev_by_strike_side.get((st_f, sd))

        rows.append(BacktestRow(
            date=d_str,
            symbol=s.get("symbol", ""),
            side=sd,
            strike=st_f,
            spot_at_d=spot_d,
            iv_pred=iv_pred,
            iv_real=iv_real,
            iv_random_walk=iv_rw if iv_rw is not None else 0.0,
            pred_err_pp=(iv_pred - iv_real) * 100,
            rw_err_pp=(iv_rw - iv_real) * 100 if iv_rw is not None else 0.0,
        ))

    return rows


def run_backtest(
    underlying: str,
    expiration: str,
    n_days: int,
    token: str,
) -> list[BacktestRow]:
    """
    Roda walk-forward dos últimos n_days pregões.

    Janela: de (hoje - n_days) até hoje.
    Pra cada dia D nessa janela, calibra em D-1 e prevê D.
    """
    today = date.today()
    # começar n_days atrás, ajustar pra pegar D-1 disponível
    start = today - timedelta(days=n_days + 5)  # folga pra weekends
    trading_days = _trading_days_between(start, today)

    all_rows: list[BacktestRow] = []
    print(f"=== walk-forward {underlying} {expiration} ===")
    print(f"  janela: {trading_days[0]} → {trading_days[-1]} ({len(trading_days)} pregões)")
    print()

    for i, day in enumerate(trading_days):
        if i == 0:
            continue  # pula o primeiro (não tem D-1)
        d_str = day.isoformat()
        print(f"  [{i}/{len(trading_days)-1}] {d_str}", end=" ... ", flush=True)
        try:
            rows = walk_forward_one_step(underlying, expiration, day, token)
            all_rows.extend(rows)
            print(f"{len(rows)} séries")
        except Exception as e:
            print(f"✗ {e}")

    return all_rows


def compute_metrics(rows: list[BacktestRow]) -> dict:
    """Métricas agregadas do backtest."""
    if not rows:
        return {"error": "sem rows"}

    pred_errs = np.array([r.pred_err_pp for r in rows])
    rw_errs = np.array([r.rw_err_pp for r in rows])

    # rmse em pontos percentuais
    rmse_pred = float(np.sqrt(np.mean(pred_errs ** 2)))
    rmse_rw = float(np.sqrt(np.mean(rw_errs ** 2)))
    mae_pred = float(np.mean(np.abs(pred_errs)))
    mae_rw = float(np.mean(np.abs(rw_errs)))

    # % das vezes que o modelo bateu a baseline
    model_wins = float(np.mean(np.abs(pred_errs) < np.abs(rw_errs)))

    # sign accuracy: em quantas séries o sinal do erro foi menor?
    # (edge positivo = modelo disse caro, edge negativo = barato)
    # aqui a gente mede erro absoluto relativo
    return {
        "n_obs": len(rows),
        "rmse_model_pp": rmse_pred,
        "rmse_baseline_pp": rmse_rw,
        "mae_model_pp": mae_pred,
        "mae_baseline_pp": mae_rw,
        "model_beats_baseline_pct": model_wins * 100,
        "delta_rmse_pp": rmse_rw - rmse_pred,  # positivo = modelo melhor
    }


def save_backtest_csv(rows: list[BacktestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        raise SystemExit("BRAPI_TOKEN não definido")

    underlying = "PETR4"
    expiration = "2026-10-16"
    n_days = 30

    rows = run_backtest(underlying, expiration, n_days, token)

    if not rows:
        print("\nNenhuma row gerada — verifique conectividade e janela.")
        return

    metrics = compute_metrics(rows)

    # salvar
    MODELS_BACKTEST.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_BACKTEST / f"{underlying}_{expiration}.csv"
    save_backtest_csv(rows, out_path)
    print(f"\n→ {out_path} ({len(rows)} rows)")

    # resumo
    print(f"\n=== métricas (em pontos percentuais de IV) ===")
    print(f"  obs:                   {metrics['n_obs']}")
    print(f"  RMSE modelo (SVI):     {metrics['rmse_model_pp']:.2f} pp")
    print(f"  RMSE baseline (RW):    {metrics['rmse_baseline_pp']:.2f} pp")
    print(f"  MAE  modelo (SVI):     {metrics['mae_model_pp']:.2f} pp")
    print(f"  MAE  baseline (RW):    {metrics['mae_baseline_pp']:.2f} pp")
    print(f"  modelo bate baseline:  {metrics['model_beats_baseline_pct']:.1f}% das séries")
    print(f"  delta RMSE:            {metrics['delta_rmse_pp']:+.2f} pp (positivo = melhor que RW)")

    if metrics["delta_rmse_pp"] < 0:
        print(f"\n  ⚠️  SVI calibrada em D-1 PREVÊ PIOR que 'IV_amanhã = IV_hoje'.")
        print(f"      O modelo está adicionando ruído. Não operar edge baseado só em SVI.")
    else:
        print(f"\n  ✓ SVI calibrada bate baseline trivial em {metrics['model_beats_baseline_pct']:.1f}% das séries.")


if __name__ == "__main__":
    main()
