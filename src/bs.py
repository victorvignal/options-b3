"""
Black-Scholes vanilla para opções europeias.

Sem dividendo discreto, sem exercício antecipado.
Apropriado para ~70% das opções B3 (séries mensais/europeias).

Convenção:
    - spot, strike: em R$
    - t: tempo até vencimento em anos (ACT/365)
    - r: taxa risk-free anual (decimal, ex: 0.1390 = 13.90%)
    - q: dividend yield anual (decimal, ex: 0.05 = 5%)
    - sigma: volatilidade anualizada (decimal, ex: 0.40 = 40%)
"""
from __future__ import annotations
from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

SQRT_2 = sqrt(2.0)
SQRT_2PI = sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    """CDF normal padrão via erf (precisão ~1e-9)."""
    return 0.5 * (1.0 + erf(x / SQRT_2))


def _norm_pdf(x: float) -> float:
    """PDF normal padrão."""
    return exp(-0.5 * x * x) / SQRT_2PI


@dataclass(frozen=True)
class BSResult:
    price: float
    delta: float
    gamma: float
    vega: float       # ∂preço/∂σ (por 1.0 de σ, NÃO por 1%)
    theta: float      # ∂preço/∂t (por ano)
    rho: float        # ∂preço/∂r (por 1.0 de r)


def _d1_d2(spot: float, strike: float, t: float, r: float, q: float, sigma: float):
    """d1 e d2 da fórmula BS."""
    if t <= 0 or sigma <= 0:
        raise ValueError(f"t e sigma devem ser > 0 (t={t}, sigma={sigma})")
    if spot <= 0 or strike <= 0:
        raise ValueError(f"spot e strike devem ser > 0 (spot={spot}, strike={strike})")
    v = sigma * sqrt(t)
    d1 = (log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / v
    d2 = d1 - v
    return d1, d2


def price(spot: float, strike: float, t: float, r: float, q: float,
          sigma: float, side: Literal["call", "put"]) -> float:
    """
    Preço BS vanilla de opção europeia.

    Call: S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
    Put:  K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
    """
    if t <= 0:
        if side == "call":
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)
    d1, d2 = _d1_d2(spot, strike, t, r, q, sigma)
    df_r = exp(-r * t)
    df_q = exp(-q * t)
    if side == "call":
        return spot * df_q * _norm_cdf(d1) - strike * df_r * _norm_cdf(d2)
    else:
        return strike * df_r * _norm_cdf(-d2) - spot * df_q * _norm_cdf(-d1)


def greeks(spot: float, strike: float, t: float, r: float, q: float,
           sigma: float, side: Literal["call", "put"]) -> BSResult:
    """
    Gregas analíticas Black-Scholes.

    Vega e rho por unidade (1.0) do parâmetro, não percentual.
    Theta por ano (dividir por 365 pra ter por dia).
    """
    if t <= 0:
        return BSResult(price=0.0, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)
    d1, d2 = _d1_d2(spot, strike, t, r, q, sigma)
    df_r = exp(-r * t)
    df_q = exp(-q * t)
    pdf_d1 = _norm_pdf(d1)

    if side == "call":
        delta = df_q * _norm_cdf(d1)
    else:
        delta = -df_q * _norm_cdf(-d1)

    gamma = df_q * pdf_d1 / (spot * sigma * sqrt(t))
    vega = spot * df_q * pdf_d1 * sqrt(t)

    common = -(spot * df_q * pdf_d1 * sigma) / (2 * sqrt(t))
    if side == "call":
        theta = common - r * strike * df_r * _norm_cdf(d2) + q * spot * df_q * _norm_cdf(d1)
    else:
        theta = common + r * strike * df_r * _norm_cdf(-d2) - q * spot * df_q * _norm_cdf(-d1)

    if side == "call":
        rho = strike * t * df_r * _norm_cdf(d2)
    else:
        rho = -strike * t * df_r * _norm_cdf(-d2)

    return BSResult(
        price=price(spot, strike, t, r, q, sigma, side),
        delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
    )


def implied_vol(market_price: float, spot: float, strike: float, t: float,
                r: float, q: float, side: Literal["call", "put"],
                tol: float = 1e-7, max_iter: int = 100) -> float:
    """
    Volatilidade implícita via bisseção (robusta, sem chute inicial).

    Retorna sigma tal que BS(spot, strike, t, r, q, sigma, side) == market_price.
    Lança ValueError se preço de mercado estiver fora do range alcançável.
    """
    if market_price <= 0:
        return 0.0

    # Intrinsic e upper bound (quando σ → ∞, preço → S·e^(-qT) pra call)
    if side == "call":
        intrinsic = max(0.0, spot * exp(-q * t) - strike * exp(-r * t))
        upper = spot * exp(-q * t)
    else:
        intrinsic = max(0.0, strike * exp(-r * t) - spot * exp(-q * t))
        upper = strike * exp(-r * t)

    if market_price < intrinsic - tol:
        raise ValueError(f"preço {market_price} abaixo do intrinsic {intrinsic}")

    sigma_low, sigma_high = 1e-6, 5.0
    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        p = price(spot, strike, t, r, q, sigma_mid, side)
        diff = p - market_price
        if abs(diff) < tol:
            return sigma_mid
        if diff > 0:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
        if sigma_high - sigma_low < tol:
            return sigma_mid

    raise ValueError(f"IV não convergiu após {max_iter} iterações (preço={market_price})")