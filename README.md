# options-b3

Pricer de opções europeias B3 + mispricing scan.

**Stack:** Black-Scholes vanilla → SVI (parametrização Gatheral) → scan de edge vs mercado.

**Pipeline:**
1. `python -m src.chain_ingest` → brapi → CSVs por underlying+vencimento
2. `python -m src.mispricing_scan` → SVI calibrada → fair value vs mid → edge
3. `streamlit run dashboard/app.py` → UI em `localhost:8501`

**Universo inicial:** BBAS3, PETR4, VALE3 (top 3 opções B3 por volume).
**Foco:** só séries europeias (~70% do mercado B3).

## Estrutura

```
src/
├── bs.py                      # Black-Scholes vanilla + gregas analíticas + IV via bisseção
├── svi.py                     # Parametrização SVI (Gatheral 2014)
├── chain_ingest.py            # brapi → CSV
├── mispricing_scan.py         # SVI calibrada → fair value → edge vs mercado
dashboard/
└── app.py                     # streamlit
tests/
├── test_call_put_parity.py    # 12 testes BS
├── test_chain_ingest.py       # 5 testes brapi
└── test_svi.py                # 3 testes SVI (sintético + real)
```

## Setup

```bash
git clone <repo>
cd options-b3
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# token brapi (Pro Irrestrito). Sem token só PETR4 sandbox funciona.
export BRAPI_TOKEN=...

# gerar dados + scan
PYTHONPATH=. python -m src.chain_ingest
PYTHONPATH=. python -m src.mispricing_scan

# dashboard
PYTHONPATH=. streamlit run dashboard/app.py
```

## Estado atual

F0-F5 completas. Veja `ORCHESTRATION.md` para detalhes.

| F | status | entrega |
|---|---|---|
| F0 setup | ✓ | repo isolado |
| F1 pricer BS | ✓ | 12/12 testes |
| F2 chain ingest | ✓ | 172 séries reais |
| F3 SVI | ✓ | 3/3 testes, RMSE 3.45% |
| F4 mispricing | ✓ | 76 candidatos |
| F5 dashboard | ✓ | streamlit em :8501 |
| F6 backtest | pendente | — |

## Caveats

- SVI σ frequentemente no bound (0.05) com chains semanais — calibração subótima, edge com magnitude alta merece desconfiança
- brapi atualiza 1x/dia às 19h BRT (EOD). Refresh de 15s só é útil via botão manual
- Mercado B3 tem call skew (ρ > 0) em blue chips de commodity — divergente do US put skew