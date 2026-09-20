"""
SVI (Stochastic Volatility Inspired) parametrização da IV surface.

Gatheral 2014: w(k) = a + b·(ρ·(k-m) + √((k-m)² + σ²))

Parâmetros:
    a: ATM variance mínimo (≥ 0)
    b: slope do skew wings (≥ 0)
    ρ: rotação do smile (-1 < ρ < 1)
    m: deslocamento do ATM (qualquer)
    σ: curvatura do smile (> 0)

Input: lista de strikes com IV empírica.
Output: 5 params SVI + função que retorna w(k) → σ²(k)·T → IV(k).

Calibração em variância (não em preço) — variância é o que SVI modela.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float  # 'sigma' do SVI, não confundir com IV

    def w(self, k: float) -> float:
        """
        Total implied variance w(k) = σ²(k) · T.
        k = log(K/F).
        """
        dm = k - self.m
        return self.a + self.b * (self.rho * dm + sqrt(dm * dm + self.sigma * self.sigma))

    def iv(self, k: float, t: float) -> float:
        """IV implícita num dado k e maturity T (decimal, ex: 0.40)."""
        if t <= 0:
            return 0.0
        w = self.w(k)
        if w < 0:
            return 0.0
        return sqrt(w / t)

    def variance(self, k: float, t: float) -> float:
        """Variância total implícita σ²·T num strike k e maturity T."""
        return self.w(k)


def calibrate_svi(
    strikes: list[float],
    ivs: list[float],
    t: float,
    forward: float | None = None,
    x0: list[float] | None = None,
) -> SVIParams:
    """
    Calibra SVI num único slice de maturity.

    strikes: lista de strikes observados
    ivs: lista de IV empírica (decimal, ex: 0.40)
    t: maturity em anos
    forward: forward price. Se None, usa ATM strike como proxy.
    x0: chute inicial [a, b, rho, m, sigma]

    Retorna SVIParams otimizado.
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)

    # Total variance empírica w = σ²·T
    w_emp = ivs * ivs * t

    # log-moneyness k = log(K/F). Se F=None, usa ATM proxy
    if forward is None:
        # ATM = strike mais próximo do preço à vista (passamos forward=None aqui;
        # caller pode passar spot se não tiver futuro)
        # fallback: k centrado em strike médio
        k = np.log(strikes / np.median(strikes))
    else:
        k = np.log(strikes / forward)

    # Chute inicial baseado em ATM variance + skew empírico
    if x0 is None:
        atm_var = float(np.median(w_emp))
        a0 = max(atm_var * 0.5, 0.005)
        b0 = max((w_emp.max() - w_emp.min()) / 4.0, 0.05)
        # skew empírico: regressão linear w vs k
        if len(k) >= 2:
            slope = float(np.polyfit(k, w_emp, 1)[0])
        else:
            slope = 0.0
        rho0 = float(np.clip(slope / max(b0, 0.01), -0.9, 0.9))
        m0 = 0.0
        sigma0 = 0.1
        x0 = [a0, b0, rho0, m0, sigma0]

    def residuals(params):
        a, b, rho, m, sig = params
        dm = k - m
        w_model = a + b * (rho * dm + np.sqrt(dm * dm + sig * sig))
        return w_model - w_emp

    # Bounds:
    #   a ≥ 0 (ATM variance)
    #   b ∈ [0.01, 5] (slope mínimo > 0 evita degeneração)
    #   ρ ∈ (-1, 1) (rotação)
    #   m ∈ [-3, 3] (deslocamento)
    #   σ ≥ 0.05 (curvatura mínima — sem isso SVI colapsa em σ→0 e perde smile)
    lower = [0.0,    0.01, -0.999, -3.0, 0.05]
    upper = [2.0,    5.0,   0.999,  3.0, 3.0]

    result = least_squares(
        residuals, x0, bounds=(lower, upper),
        method="trf", max_nfev=2000, ftol=1e-10, xtol=1e-10,
    )

    a, b, rho, m, sig = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sig)


def fit_quality(params: SVIParams, strikes: list[float], ivs: list[float],
                t: float, forward: float | None = None) -> dict:
    """
    Mede qualidade do fit: RMSE em IV (decimal) e em variância.
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    if forward is None:
        k = np.log(strikes / np.median(strikes))
    else:
        k = np.log(strikes / forward)

    ivs_fitted = np.array([params.iv(float(kk), t) for kk in k])
    iv_rmse = float(np.sqrt(np.mean((ivs_fitted - ivs) ** 2)))

    w_emp = ivs * ivs * t
    w_fitted = np.array([params.w(float(kk)) for kk in k])
    w_rmse = float(np.sqrt(np.mean((w_fitted - w_emp) ** 2)))

    return {
        "iv_rmse": iv_rmse,           # em decimal (0.01 = 1pp)
        "variance_rmse": w_rmse,
        "iv_rmse_pct": iv_rmse * 100, # human-readable
        "n_points": len(strikes),
    }