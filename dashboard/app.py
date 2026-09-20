"""
Dashboard local pra visualizar o pricer de opções B3 funcionando.

Roda com: streamlit run dashboard/app.py
Abre em: http://localhost:8501

Lê:
    - data/{underlying}/chain_*.csv (chains ingeridos)
    - models/scan/*.csv (mispricing scan)

Mostra:
    - Resumo geral (chains, séries, candidatos)
    - Tabela de candidatos filtrada (sem skip-warn)
    - Scatter edge vs moneyness por underlying
    - Histograma de edge distribution
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh


ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SCAN = ROOT / "models" / "scan"


@st.cache_data(ttl=15)
def load_all_scan() -> pd.DataFrame:
    """Carrega todos os CSVs de scan consolidados."""
    main = SCAN / "all_mispricing.csv"
    if not main.exists():
        return pd.DataFrame()
    df = pd.read_csv(main)
    return df


@st.cache_data(ttl=15)
def list_chains() -> list[tuple[str, str, Path]]:
    """Lista (underlying, expiration, path) de CSVs de chain."""
    out = []
    if not DATA.exists():
        return out
    for under_dir in sorted(DATA.iterdir()):
        if not under_dir.is_dir():
            continue
        for csv_path in sorted(under_dir.glob("chain_*.csv")):
            exp = csv_path.stem.replace("chain_", "")
            out.append((under_dir.name, exp, csv_path))
    return out


@st.cache_data(ttl=15)
def data_mtime() -> str:
    """Quando os CSVs de scan foram atualizados pela última vez."""
    main = SCAN / "all_mispricing.csv"
    if not main.exists():
        return "nunca"
    import datetime
    ts = datetime.datetime.fromtimestamp(main.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def main():
    st.set_page_config(
        page_title="Options B3 — Pricer",
        layout="wide",
        page_icon="📊",
    )

    # Auto-refresh a cada 15s — reexecuta o script todo (incluindo load_all_scan)
    st_autorefresh(interval=15000, key="autorefresh")

    # Header com timestamp de última atualização
    cols = st.columns([3, 1])
    with cols[0]:
        st.title("📊 Options B3 — pricer & mispricing scan")
    with cols[1]:
        st.caption(f"última atualização CSVs: {data_mtime()}")

    # Botão de refresh manual
    if st.button("🔄 atualizar agora (puxa brapi)"):
        with st.spinner("puxando chain + recalculando SVI..."):
            import subprocess, os
            token = os.environ.get("BRAPI_TOKEN", "")
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            r1 = subprocess.run(["python", "-m", "src.chain_ingest"], capture_output=True, env=env)
            r2 = subprocess.run(["python", "-m", "src.mispricing_scan"], capture_output=True, env=env)
            if r1.returncode == 0 and r2.returncode == 0:
                st.success("atualizado ✓")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"erro: chain={r1.returncode} scan={r2.returncode}")
                st.code(r1.stderr.decode()[-500:] + r2.stderr.decode()[-500:])

    df = load_all_scan()

    # Sidebar
    st.sidebar.header("Filtros")
    if not df.empty:
        all_underlyings = sorted(df["underlying"].unique())
        sel_under = st.sidebar.multiselect("Underlying", all_underlyings, default=all_underlyings)
        all_recs = sorted(df["recommendation"].unique())
        sel_rec = st.sidebar.multiselect("Recomendação", all_recs, default=all_recs)
        min_abs_edge = st.sidebar.slider("Edge mínimo (%)", 0.0, 30.0, 5.0, 1.0)

        df = df[df["underlying"].isin(sel_under)]
        df = df[df["recommendation"].isin(sel_rec)]
        df = df[df["edge_pct"].abs() >= min_abs_edge]

    # Resumo
    st.subheader("Resumo")
    cols = st.columns(4)
    chains = list_chains()
    cols[0].metric("Chains ingeridos", len(chains))
    cols[1].metric("Séries no scan", len(load_all_scan()))
    cols[2].metric("Candidatos (|edge|≥5%)", len(df[df["edge_pct"].abs() >= 5]))
    if not df.empty:
        buys = len(df[df["recommendation"] == "buy"])
        sells = len(df[df["recommendation"] == "sell"])
        cols[3].metric(f"Buy / Sell", f"{buys} / {sells}")

    # Tabela
    st.subheader(f"Candidatos ({len(df)} séries)")
    if df.empty:
        st.info("Nenhum candidato com esses filtros. Rode `python -m src.chain_ingest && python -m src.mispricing_scan` primeiro.")
        return

    # coluna moneyness pra scatter
    df["moneyness"] = df["strike"] / df["spot"]

    show_cols = [
        "symbol", "underlying", "side", "strike", "expiration",
        "mid", "iv_emp", "iv_svi", "fair_value",
        "edge_pct", "edge_abs", "recommendation",
    ]
    show_df = df[show_cols].copy()
    show_df["edge_pct"] = show_df["edge_pct"].round(2)
    show_df["edge_abs"] = show_df["edge_abs"].round(3)
    show_df["iv_emp"] = (show_df["iv_emp"] * 100).round(1)
    show_df["iv_svi"] = (show_df["iv_svi"] * 100).round(1)
    show_df = show_df.rename(columns={
        "iv_emp": "iv_emp(%)", "iv_svi": "iv_svi(%)",
        "edge_pct": "edge(%)", "edge_abs": "edge(R$)",
    })

    st.dataframe(
        show_df.sort_values("edge(%)", key=abs, ascending=False),
        use_container_width=True,
        height=400,
    )

    # Scatter edge vs moneyness
    st.subheader("Edge (%) vs Moneyness")
    fig = px.scatter(
        df, x="moneyness", y="edge_pct",
        color="recommendation",
        symbol="side",
        hover_data=["symbol", "strike", "mid", "fair_value", "edge_pct"],
        color_discrete_map={
            "buy": "#4dbe95", "sell": "#e85d75",
            "skip": "#888", "skip-warn": "#bbb",
        },
    )
    fig.add_hline(y=5, line_dash="dash", line_color="gray", annotation_text="+5%")
    fig.add_hline(y=-5, line_dash="dash", line_color="gray", annotation_text="-5%")
    fig.add_hline(y=0, line_color="white")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Histograma de edge distribution
    st.subheader("Distribuição de edge (%)")
    fig2 = px.histogram(
        df, x="edge_pct", nbins=40,
        color="recommendation",
        color_discrete_map={
            "buy": "#4dbe95", "sell": "#e85d75",
            "skip": "#888", "skip-warn": "#bbb",
        },
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # SVI fit por chain
    st.subheader("IV empírica vs IV SVI por chain")
    chain_choice = st.selectbox(
        "Chain",
        options=sorted(df["expiration"].unique() + "/" + df["underlying"].unique())
        if not df.empty else [],
    )
    if chain_choice:
        parts = chain_choice.split("/")
        if len(parts) == 2:
            exp, under = parts
            chain_df = df[(df["expiration"] == exp) & (df["underlying"] == under)]
            if not chain_df.empty:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=chain_df["moneyness"], y=chain_df["iv_emp"] * 100,
                    mode="markers", name="IV empírica",
                    marker=dict(color="#489ffa", size=8),
                ))
                fig3.add_trace(go.Scatter(
                    x=chain_df["moneyness"], y=chain_df["iv_svi"] * 100,
                    mode="lines", name="IV SVI",
                    line=dict(color="#4dbe95", width=2),
                ))
                fig3.update_layout(
                    xaxis_title="Moneyness (K/Spot)",
                    yaxis_title="IV (%)",
                    height=350,
                )
                st.plotly_chart(fig3, use_container_width=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Como rodar:**")
    st.sidebar.code("python -m src.chain_ingest")
    st.sidebar.code("python -m src.mispricing_scan")
    st.sidebar.code("streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()