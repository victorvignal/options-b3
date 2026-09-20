"""
Teste rigoroso de call-put parity pra BS vanilla.

Parity: C - P = S·e^(-qT) - K·e^(-rT)

Se isso falha, o pricer está errado.
"""
from math import exp

from src.bs import greeks, implied_vol, price


def call_put_parity(spot, strike, t, r, q, sigma, tol=1e-9):
    """Verifica C - P = S·e^(-qT) - K·e^(-rT)."""
    c = price(spot, strike, t, r, q, sigma, "call")
    p = price(spot, strike, t, r, q, sigma, "put")
    expected = spot * exp(-q * t) - strike * exp(-r * t)
    diff = abs((c - p) - expected)
    assert diff < tol, f"parity falhou: C-P={c-p}, esperado={expected}, diff={diff}"


def test_parity_atm_riskfree_zero():
    """ATM, sem juros, sem dividendo."""
    call_put_parity(spot=100, strike=100, t=1.0, r=0.0, q=0.0, sigma=0.30)


def test_parity_otm_high_vol():
    """OTM, vol alta, juros e dividendo não-zero (PETR4-realista)."""
    call_put_parity(spot=38.50, strike=40.0, t=0.25, r=0.1390, q=0.05, sigma=0.45)


def test_parity_deep_itm():
    """Deep ITM call."""
    call_put_parity(spot=100, strike=70, t=2.0, r=0.10, q=0.03, sigma=0.25)


def test_parity_short_maturity():
    """Vencimento em 1 dia."""
    call_put_parity(spot=38.50, strike=39.0, t=1/365, r=0.1390, q=0.05, sigma=0.40)


def test_intrinsic_at_expiry():
    """Em t=0, preço = intrinsic."""
    assert abs(price(spot=40, strike=35, t=0, r=0.10, q=0.05, sigma=0.5, side="call") - 5) < 1e-9
    assert price(spot=35, strike=40, t=0, r=0.10, q=0.05, sigma=0.5, side="call") == 0.0
    assert abs(price(spot=35, strike=40, t=0, r=0.10, q=0.05, sigma=0.5, side="put") - 5) < 1e-9
    assert price(spot=45, strike=40, t=0, r=0.10, q=0.05, sigma=0.5, side="put") == 0.0


def test_known_value_atm():
    """
    ATM call com σ=20%, T=1y, r=5%, q=0%.
    d1 = (r - q + σ²/2)·T / (σ·√T) = (0.05 + 0.02) / 0.20 = 0.35
    C = S·N(0.35) - K·e^(-0.05)·N(0.15)
    ≈ 100·0.6368 - 95.12·0.5596
    ≈ 63.68 - 53.23 ≈ 10.45
    """
    from math import erf, sqrt
    SQRT_2 = sqrt(2)
    n_cdf = lambda x: 0.5 * (1 + erf(x / SQRT_2))
    # Cálculo analítico direto via BS
    d1 = (0 + (0.05 - 0 + 0.5 * 0.04) * 1.0) / 0.20  # = 0.35
    d2 = d1 - 0.20  # = 0.15
    expected = 100 * n_cdf(d1) - 100 * exp(-0.05) * n_cdf(d2)
    got = price(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
    assert abs(got - expected) < 1e-9, f"ATM call: got={got}, esperado={expected}"


def test_known_value_otm_put():
    """
    Hull (9ª ed) exemplo: S=42, K=40, r=0.10, σ=0.20, T=0.5, q=0.
    Call ≈ 4.76, Put ≈ 0.81.
    """
    call = price(42, 40, 0.5, 0.10, 0.0, 0.20, "call")
    put = price(42, 40, 0.5, 0.10, 0.0, 0.20, "put")
    assert abs(call - 4.76) < 0.01, f"call Hull: {call}"
    assert abs(put - 0.81) < 0.01, f"put Hull: {put}"


def test_greeks_call_atm():
    """ATM call: delta ~0.5, gamma alto, vega > 0."""
    g = greeks(100, 100, 0.5, 0.05, 0.0, 0.25, "call")
    # delta ~ N(0.125) ~ 0.55 com q=0
    assert 0.5 < g.delta < 0.6, f"delta ATM call: {g.delta}"
    assert g.gamma > 0
    assert g.vega > 0
    # theta negativo (time decay)
    assert g.theta < 0


def test_greeks_put_atm():
    """ATM put: delta = -N(-d1), com d1=0.35 → -0.363."""
    g = greeks(100, 100, 0.5, 0.05, 0.0, 0.25, "put")
    # d1 = (r - q + σ²/2)·T / (σ·√T) = (0.05 + 0.03125)·0.5 / (0.25·0.707) = 0.229
    # delta_put = -N(-d1) ≈ -0.41
    assert -0.50 < g.delta < -0.30, f"delta ATM put: {g.delta}"


def test_greeks_call_deep_itm():
    """Deep ITM call: delta → 1."""
    g = greeks(100, 70, 0.5, 0.05, 0.0, 0.25, "call")
    assert g.delta > 0.95, f"delta ITM call: {g.delta}"


def test_implied_vol_roundtrip():
    """Preço → IV → Preço deve dar o mesmo (até tolerância)."""
    market = price(100, 100, 0.5, 0.05, 0.0, 0.30, "call")
    iv = implied_vol(market, 100, 100, 0.5, 0.05, 0.0, "call")
    assert abs(iv - 0.30) < 1e-5, f"IV roundtrip: got={iv}, esperado=0.30"


def test_implied_vol_from_brapi_petr4():
    """
    PETR4 call: spot=38.50, K=39, T=0.08 (~30d), r=13.90%, q=5%, market=1.62
    IV empírica bruta da chain brapi 09-25, strike ATM aproximado.
    """
    market_price = 1.62
    iv = implied_vol(market_price, spot=38.50, strike=39.0, t=0.08,
                     r=0.1390, q=0.05, side="call")
    # IV razoável pra PETR4 ATM 30d: ~30-45%
    assert 0.25 < iv < 0.55, f"IV PETR4 ATM: {iv}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passaram")
    exit(failed)