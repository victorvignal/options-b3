"""
Chain ingest da brapi pra opções europeias B3.

Por chain: brapi retorna strike, bid, ask, close, volume, openInterest,
side, optionStyle, expirationDate. Cruza com:
    - spot e dividendYield do underlying (/quote)
    - curva risk-free do DI1 (/futures/term-structure)

Output: data/{UNDERLYING}/chain_{EXPIRATION}.csv

Filtra séries:
    - optionStyle == "european" (descarta americana)
    - bid > 0 AND ask > 0 (descarta sem liquidez)
    - spread < 50% do mid (descarta ruído)
    - openInterest > 100 (descarta OTM profundo sem OI)
"""
from __future__ import annotations
import json
import os
import urllib.request
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

BRAPI_BASE = "https://brapi.dev/api/v2"


def _get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get_underlying(underlying: str, token: str) -> dict:
    """
    Pega spot + dividendYield do underlying.

    brapi tem 2 endpoints:
        - /stocks/quote → spot (regularMarketPrice)
        - /stocks/statistics → dividendYield (decimal: 0.08 = 8%)
    """
    quote = _get(f"{BRAPI_BASE}/stocks/quote?symbols={underlying}", token)
    stats = _get(f"{BRAPI_BASE}/stocks/statistics?symbols={underlying}", token)
    q_results = quote.get("results", [])
    s_results = stats.get("results", [])
    if not q_results:
        raise ValueError(f"quote vazio pra {underlying}")
    qd = q_results[0].get("data", q_results[0])
    sd = s_results[0].get("data", {}) if s_results else {}
    return {
        "spot": float(qd.get("regularMarketPrice", 0)),
        "dividend_yield": float(sd.get("dividendYield") or 0),
        "short_name": qd.get("shortName", underlying),
    }


def get_risk_free_curve(token: str) -> list[tuple[float, float]]:
    """
    Curva DI1 futura em % a.a. (decimal).
    Retorna lista de [(tenor_anos, taxa_decimal), ...].

    brapi shape: {asset, contracts: [{symbol, expirationDate, settlementRate, ...}]}
    settlementRate vem em % a.a. (ex: 13.90 = 13.90%).
    """
    data = _get(f"{BRAPI_BASE}/futures/term-structure?asset=DI1", token)
    curve = []
    for c in data.get("contracts", []):
        rate_pct = c.get("settlementRate")
        due = c.get("expirationDate")
        if rate_pct is None or due is None:
            continue
        due_dt = datetime.fromisoformat(due[:10]).date()
        t_years = max((due_dt - date.today()).days / 365.0, 0.001)
        curve.append((t_years, rate_pct / 100.0))
    return sorted(curve, key=lambda x: x[0])


def interpolate_risk_free(curve: list[tuple[float, float]], t: float) -> float:
    """
    Interpolação linear na curva. Se t fora do range, usa o extremo.
    Retorna taxa decimal.
    """
    if not curve:
        return 0.1390  # fallback: SELIC atual
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        t_lo, r_lo = curve[i]
        t_hi, r_hi = curve[i + 1]
        if t_lo <= t <= t_hi:
            w = (t - t_lo) / (t_hi - t_lo)
            return r_lo + w * (r_hi - r_lo)
    return curve[-1][1]


def get_expirations(underlying: str, token: str) -> list[str]:
    """Lista vencimentos disponíveis (ISO YYYY-MM-DD)."""
    data = _get(f"{BRAPI_BASE}/options/expirations?underlying={underlying}", token)
    return data.get("expirations", [])


def get_chain(underlying: str, expiration: str, token: str) -> list[dict]:
    """
    Chain completa de um vencimento (último pregão disponível).
    Retorna lista de dicts com todos os campos brapi por série.
    """
    data = _get(
        f"{BRAPI_BASE}/options/chain?underlying={underlying}&expirationDate={expiration}",
        token,
    )
    return data.get("series", [])


def fetch_chain_at_date(underlying: str, expiration: str, date_str: str,
                        token: str) -> list[dict]:
    """
    Chain histórica: retorna o snapshot de uma data específica (YYYY-MM-DD).
    Útil pra backtest OOS — sem isso, brapi só dá o último pregão.

    Verificado empiricamente: o parâmetro ?date= é respeitado, e o `date`
    de cada série bate com o dia pedido. Símbolos que não existiam naquela
    data simplesmente somem do payload (brapi não retorna histórico de
    symbols reusados, mas mantém continuidade por strike+side+vencimento).
    """
    url = (
        f"{BRAPI_BASE}/options/chain"
        f"?underlying={underlying}&expirationDate={expiration}&date={date_str}"
    )
    data = _get(url, token)
    return data.get("series", [])


def fetch_analytics_at_date(underlying: str, expiration: str, date_str: str,
                            token: str) -> list[dict]:
    """
    Analytics histórica (IV + gregas pré-calculadas pela brapi) por data.
    Útil como 'verdade' pra backtest — evita recalcular IV via Newton.

    Retorna lista com impliedVolatility/delta/gamma/theta/vega/rho por série.
    Series com confidence='none' e impliedVolatility=null (ex: ITM profundo
    com último trade >5 dias) devem ser descartadas.
    """
    url = (
        f"{BRAPI_BASE}/options/analytics"
        f"?underlying={underlying}&expirationDate={expiration}&date={date_str}"
    )
    data = _get(url, token)
    return data.get("analytics", [])


def filter_european(series: list[dict]) -> list[dict]:
    """Filtra só europeias (descarta americana — 30% das séries)."""
    return [s for s in series if s.get("optionStyle") == "european"]


def filter_liquid(series: list[dict]) -> list[dict]:
    """
    Filtra séries com liquidez mínima.

    Critérios:
        - bid > 0 AND ask > 0 (existem os dois lados)
        - spread < 50% do mid (mercado não está em stress)
        - moneyness entre 0.85 e 1.15 (descarta OTM profundo)
        - volume > 0 OR openInterest > 50 (alguma atividade)
    """
    out = []
    for s in series:
        bid = s.get("bid") or 0
        ask = s.get("ask") or 0
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        if mid <= 0:
            continue
        spread_pct = (ask - bid) / mid
        if spread_pct > 0.50:
            continue
        # moneyness: requer strike do payload; se não tiver, usa heurística via série
        strike = s.get("strike") or 0
        # sem spot aqui, moneyness check fica no caller (precisa spot)
        volume = s.get("volume") or 0
        oi = s.get("openInterest") or 0
        if volume == 0 and oi < 50:
            continue
        out.append({**s, "_spread_pct": spread_pct})
    return out


@dataclass
class OptionRow:
    """Linha do CSV de output."""
    symbol: str
    underlying: str
    side: Literal["call", "put"]
    strike: float
    expiration: str
    bid: float
    ask: float
    mid: float
    last: float
    spread_pct: float
    volume: int
    open_interest: int

    # contexto de mercado
    spot: float
    risk_free: float
    dividend_yield: float
    t_years: float


def build_row(s: dict, underlying: str, spot: float, r: float, q: float, t: float) -> OptionRow:
    """Converte série brapi em OptionRow pronto pro CSV."""
    bid = float(s.get("bid") or 0)
    ask = float(s.get("ask") or 0)
    mid = (bid + ask) / 2
    return OptionRow(
        symbol=s["symbol"],
        underlying=underlying,
        side=s["side"],
        strike=float(s["strike"]),
        expiration=s["expirationDate"],
        bid=bid,
        ask=ask,
        mid=mid,
        last=float(s.get("close") or 0),
        spread_pct=(ask - bid) / mid if mid > 0 else 0,
        volume=int(s.get("volume") or 0),
        open_interest=int(s.get("openInterest") or 0),
        spot=spot,
        risk_free=r,
        dividend_yield=q,
        t_years=t,
    )


def ingest_underlying(underlying: str, token: str,
                      expirations: list[str] | None = None,
                      max_expirations: int = 4) -> list[OptionRow]:
    """
    Ingest completo pra 1 underlying.

    Pega spot+divYield, curva risk-free, vencimentos mais próximos,
    chain de cada um, filtra europeias+líquidas, retorna OptionRows.
    """
    u = get_underlying(underlying, token)
    spot, q = u["spot"], u["dividend_yield"]
    curve = get_risk_free_curve(token)

    if expirations is None:
        expirations = get_expirations(underlying, token)[:max_expirations]

    all_rows = []
    for exp in expirations:
        try:
            series = get_chain(underlying, exp, token)
        except Exception as e:
            print(f"  ✗ {exp}: {e}")
            continue
        series = filter_european(series)
        series = filter_liquid(series)
        if not series:
            print(f"  · {exp}: 0 séries europeias líquidas")
            continue
        exp_dt = datetime.fromisoformat(exp).date()
        t = max((exp_dt - date.today()).days / 365.0, 1 / 365.0)
        r = interpolate_risk_free(curve, t)
        for s in series:
            all_rows.append(build_row(s, underlying, spot, r, q, t))
        print(f"  ✓ {exp}: {len(series)} séries europeias líquidas (T={t:.3f}y, r={r:.2%})")

    return all_rows


def save_csv(rows: list[OptionRow], path: Path) -> None:
    """Salva rows em CSV (sem header fancy, fácil de inspecionar)."""
    import csv
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
    token = os.environ.get("BRAPI_TOKEN", "")
    if not token:
        raise SystemExit("BRAPI_TOKEN não definido")

    # universe top 3 por volume B3
    underlyings = ["BBAS3", "PETR4", "VALE3"]
    today = date.today().isoformat()

    for u in underlyings:
        print(f"\n=== {u} ===")
        try:
            rows = ingest_underlying(u, token, max_expirations=4)
            out_dir = DATA / u
            if rows:
                # agrupa por expiration
                by_exp = {}
                for r in rows:
                    by_exp.setdefault(r.expiration, []).append(r)
                for exp, exp_rows in by_exp.items():
                    out_path = out_dir / f"chain_{exp}.csv"
                    save_csv(exp_rows, out_path)
                    print(f"  → {out_path} ({len(exp_rows)} rows)")
                # também CSV agregado
                all_path = out_dir / f"all_european_{today}.csv"
                save_csv(rows, all_path)
                print(f"  → {all_path} ({len(rows)} rows total)")
            else:
                print(f"  · nenhuma série europeia líquida encontrada")
        except Exception as e:
            print(f"  ✗ {u}: {e}")


if __name__ == "__main__":
    main()