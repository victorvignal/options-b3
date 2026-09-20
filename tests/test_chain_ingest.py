"""
Testa o chain_ingest num underlying (sandbox PETR4 funciona sem token
pro `/quote`, mas `/options/chain` precisa token). Pra teste real,
seta BRAPI_TOKEN no env.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chain_ingest import (
    get_underlying,
    get_risk_free_curve,
    interpolate_risk_free,
    get_expirations,
    get_chain,
    filter_european,
    filter_liquid,
)


def test_get_underlying_petr4():
    """PETR4 sandbox funciona sem token em /quote."""
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        print("  · pulando (sem BRAPI_TOKEN)")
        return
    u = get_underlying("PETR4", token)
    print(f"  spot={u['spot']} divYield={u['dividend_yield']:.2%} shortName={u['short_name']}")
    assert u["spot"] > 0, "spot deve ser positivo"


def test_risk_free_curve():
    """Curva DI1 retorna lista (t_anos, taxa_decimal)."""
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        print("  · pulando (sem BRAPI_TOKEN)")
        return
    curve = get_risk_free_curve(token)
    print(f"  curva tem {len(curve)} pontos")
    if len(curve) >= 2:
        t_lo, r_lo = curve[0]
        t_hi, r_hi = curve[-1]
        print(f"  short end: T={t_lo:.3f}y r={r_lo:.2%}")
        print(f"  long end:  T={t_hi:.3f}y r={r_hi:.2%}")
        # taxa curta deve ser menor ou igual à longa (curva normal)
        assert r_lo <= r_hi + 0.02, "curva invertida (anomalia) ou dados faltando"


def test_interpolate_risk_free():
    """Interpolação retorna valor entre extremos."""
    curve = [(0.25, 0.13), (1.0, 0.14), (5.0, 0.16)]
    # T=0.5y: entre 13% e 14%
    r = interpolate_risk_free(curve, 0.5)
    assert 0.13 < r < 0.14, f"interpolação errada: {r}"
    # T fora do range: usa extremo
    assert interpolate_risk_free(curve, 0.01) == 0.13
    assert interpolate_risk_free(curve, 10.0) == 0.16


def test_expirations_petr4():
    """Lista vencimentos do PETR4."""
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        print("  · pulando (sem BRAPI_TOKEN)")
        return
    exps = get_expirations("PETR4", token)
    print(f"  primeiros 5 vencimentos: {exps[:5]}")
    assert len(exps) >= 5, "deveria ter pelo menos 5 vencimentos"


def test_chain_petr4_filtered():
    """Chain de 1 vencimento, filtrada europeia+líquida."""
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        print("  · pulando (sem BRAPI_TOKEN)")
        return
    exps = get_expirations("PETR4", token)
    chain = get_chain("PETR4", exps[0], token)
    print(f"  chain total: {len(chain)} séries")
    eur = filter_european(chain)
    print(f"  europeias: {len(eur)}")
    liq = filter_liquid(eur)
    print(f"  europeias + líquidas: {len(liq)}")
    if liq:
        s = liq[0]
        print(f"  exemplo: {s['symbol']} {s['side']} K={s['strike']} bid={s['bid']} ask={s['ask']} OI={s['openInterest']}")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in tests:
        print(f"\n{fn.__name__}:")
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")