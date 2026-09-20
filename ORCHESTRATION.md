# OPTIONS-B3 — orchestration log

Projeto isolado. **NÃO** cruzar com `~/projects/sulfur/` ou `~/projects/sulfur-ml/`.

Objetivo: precificar opções europeias B3 (Black-Scholes + SVI), achar mispricing vs mercado, operar edge.

## Stack escolhida (por que simples)

- **Opções europeias só** — 70% das séries B3 são europeias (mensais). Mais simples, pricer fechado.
- **Black-Scholes vanilla** — fórmula fechada, gregas analíticas, call-put parity rigorosa.
- **SVI (Gatheral 2014)** — parametrização da IV surface observada. 5 params por slice.
- **Streamlit** — dashboard local (localhost:8501). Sem deploy, sem auth.

## Universe inicial

Top 3 opções B3 por volume agregado (5 vencimentos):

| rank | ativo | volume total | justificativa |
|---|---|---|---|
| 1 | **BBAS3** | 63.3M | bancão, IV alto, liquidez forte |
| 2 | **PETR4** | 48.6M | commodity + sandbox brapi (sem token) |
| 3 | **VALE3** | 42.5M | commodity + moeda, skew diferente |

Sandbox: PETR4 funciona sem token pra validar pipeline.

## F0: Setup [status: complete] [started: 2026-09-20]
- Repo ~/projects/options-b3/ criado
- Estrutura: src/, tests/, data/, models/, dashboard/
- .env → symlink ~/projects/sulfur/.env.local (BRAPI_TOKEN)

## F1: Pricer BS + call-put parity [status: in_progress] [started: 2026-09-20]
- src/bs.py: Black-Scholes vanilla (call/put europeus)
- src/greeks.py: gregas analíticas
- tests/test_call_put_parity.py: validação rigorosa

## F2: Chain ingest brapi [status: complete] [started: 2026-09-20]
- src/chain_ingest.py: brapi → CSV por (underlying, expiration)
- endpoints usados: /stocks/quote, /stocks/statistics, /options/expirations,
  /options/chain, /futures/term-structure?asset=DI1
- parâmetros REAIS (não chutados): spot, dividendYield, r (curva DI), t (expiration-today)
- filtro: europeian + bid>0 + ask>0 + spread<50% + (vol>0 ou OI>=50)
- output: 3 underlyings × 2-4 vencimentos = 172 séries europeias líquidas

## F3: SVI calibration [status: complete] [started: 2026-09-20]
- src/svi.py: parametrização Gatheral (a, b, ρ, m, σ)
- calibração via scipy least_squares em variância total
- testes:
  - recovery sintético perfeito (params conhecidos → recovered)
  - sem-arbitragem w(k) ≥ 0
  - PETR4 real 2026-10-16: ATM IV 41.9%, 38 strikes calibrados, RMSE 3.38%
- filter insight: descartar IVs > 1.6× ATM (bid stale em ITM)
- ρ = 0.41 detectado (call skew positivo, coerente com mercado BR)

## F4: Mispricing scan [status: complete] [started: 2026-09-20]
- src/mispricing_scan.py: SVI calibrada → fair value → edge vs mid
- output: models/scan/{underlying}_{exp}.csv + all_mispricing.csv
- thresholds: edge >5% = candidato, >25% = skip-warn (suspeita)
- resultado: 116 séries varridas, 76 candidatos com |edge|>5%
- caveat: σ SVI frequentemente no bound (0.05), indica calibração subótima
  com chains semanais. SVI funciona melhor em mensais com ≥30 strikes.

## F5: Streamlit dashboard [status: complete] [started: 2026-09-20]
- dashboard/app.py: filtros + tabela candidatos + scatter edge vs moneyness
  + histograma edge distribution + IV empírica vs SVI por chain
- auto-refresh visual a cada 15s (streamlit-autorefresh)
- timestamp "última atualização CSVs" no topo (mostra se dados brapi mudaram)
- botão "🔄 atualizar agora" roda chain_ingest + mispricing_scan manual
- roda em localhost:8501 (Network URL: 192.168.68.106:8501)
- comando: `PYTHONPATH=. streamlit run dashboard/app.py`
- nota: brapi atualiza 1x/dia às 19h BRT (EOD). Refresh de 15s só é útil
  durante os primeiros minutos após atualização, ou via botão manual.

## F6: Backtest honesto [status: pending]

## Próximo passo
F1 → F2 (chain ingest real)