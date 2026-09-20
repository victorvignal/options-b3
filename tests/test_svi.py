"""
Testes SVI.

Validações:
    1. Fit em dados sintéticos (sabemos a resposta)
    2. Fit em dados reais brapi (PETR4 mensal)
    3. Sem-arbabilidade: w(k) ≥ 0 sempre
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.bs import implied_vol, price
from src.svi import SVIParams, calibrate_svi, fit_quality


def test_svi_recovery_synthetic():
    """
    Gera IV empírica a partir de params SVI conhecidos,
    calibra de volta, checa que params batem.
    """
    true_params = SVIParams(a=0.04, b=0.30, rho=-0.40, m=0.0, sigma=0.20)
    t = 0.25  # 3 meses

    strikes = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]) * 100.0
    # forward = 100, então k = log(K/100)
    k = np.log(strikes / 100.0)
    w = np.array([true_params.w(float(kk)) for kk in k])
    iv_emp = np.sqrt(w / t)

    fitted = calibrate_svi(strikes.tolist(), iv_emp.tolist(), t, forward=100.0)

    # Tolerância razoável pra least_squares
    assert abs(fitted.a - true_params.a) < 0.01, f"a: {fitted.a} vs {true_params.a}"
    assert abs(fitted.b - true_params.b) < 0.05, f"b: {fitted.b} vs {true_params.b}"
    assert abs(fitted.rho - true_params.rho) < 0.05, f"rho: {fitted.rho} vs {true_params.rho}"
    print(f"  ✓ recovery: a={fitted.a:.4f} b={fitted.b:.4f} rho={fitted.rho:.4f} m={fitted.m:.4f} sigma={fitted.sigma:.4f}")


def test_svi_no_negative_variance():
    """w(k) deve ser ≥ 0 pra qualquer k (sem variância negativa)."""
    params = SVIParams(a=0.04, b=0.30, rho=-0.40, m=0.0, sigma=0.20)
    ks = np.linspace(-0.5, 0.5, 100)
    for k in ks:
        w = params.w(float(k))
        assert w >= 0, f"w negativa em k={k}: {w}"
    print(f"  ✓ w(k) ≥ 0 em todo range")


def test_svi_real_petr4():
    """
    Calibra SVI em dados reais do chain PETR4 mensal (2026-10-16).
    Filtra outliers de IV (bid stale, OTM profundo) antes de calibrar.
    """
    csv_path = Path(__file__).parent.parent / "data" / "PETR4" / "chain_2026-10-16.csv"
    if not csv_path.exists():
        print("  · pulando (chain não existe)")
        return

    import csv
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < 5:
        print(f"  · pulando ({len(rows)} rows)")
        return

    spot = float(rows[0]["spot"])
    r = float(rows[0]["risk_free"])
    q = float(rows[0]["dividend_yield"])
    t = float(rows[0]["t_years"])

    forward = spot * np.exp((r - q) * t)

    # primeira passada: coleta IVs pra estimar ATM
    raw_data = []
    for row in rows:
        strike = float(row["strike"])
        bid = float(row["bid"])
        ask = float(row["ask"])
        mid = (bid + ask) / 2
        if mid <= 0:
            continue
        side = row["side"]
        moneyness = strike / forward
        if moneyness < 0.85 or moneyness > 1.15:
            continue
        try:
            iv = implied_vol(mid, spot, strike, t, r, q, side)
        except Exception:
            continue
        if iv < 0.15 or iv > 0.80:
            continue
        raw_data.append((strike, iv, side, moneyness))

    if len(raw_data) < 5:
        print(f"  · só {len(raw_data)} strikes válidos")
        return

    # ATM = strikes com moneyness entre 0.95 e 1.05
    atm_ivs = [iv for _, iv, _, m in raw_data if 0.95 <= m <= 1.05]
    atm_iv = float(np.median(atm_ivs)) if atm_ivs else 0.40

    # segunda passada: descarta IVs >1.6× ATM (bid stale em ITM/OTM profundo)
    strikes_raw = []
    ivs_raw = []
    discarded = 0
    for strike, iv, side, moneyness in raw_data:
        if iv > atm_iv * 1.6:
            discarded += 1
            continue
        strikes_raw.append(strike)
        ivs_raw.append(iv)

    print(f"  ATM IV estimado: {atm_iv*100:.1f}%")

    if len(strikes_raw) < 5:
        print(f"  · só {len(strikes_raw)} strikes válidos")
        return

    fitted = calibrate_svi(strikes_raw, ivs_raw, t, forward=forward)
    q_metrics = fit_quality(fitted, strikes_raw, ivs_raw, t, forward=forward)

    print(f"  ✓ {len(strikes_raw)} strikes calibrados ({discarded} descartados):")
    print(f"    params: a={fitted.a:.4f} b={fitted.b:.4f} rho={fitted.rho:.3f} m={fitted.m:.3f} sigma={fitted.sigma:.3f}")
    print(f"    IV RMSE: {q_metrics['iv_rmse_pct']:.2f}% ({q_metrics['iv_rmse']:.4f})")
    print(f"    variância RMSE: {q_metrics['variance_rmse']:.4f}")
    print(f"    spot={spot} forward={forward:.2f}")

    # qualidade razoável: RMSE < 5% em IV (mercado tem ruído, mas filtrado deve caber)
    assert q_metrics["iv_rmse"] < 0.05, f"fit ruim: {q_metrics['iv_rmse']}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passaram")
    exit(failed)