"""Smoke test isolado: tabs funcionam?"""
import streamlit as st

st.set_page_config(page_title="Test Tabs", layout="wide")
st.title("Test Tabs")

t1, t2 = st.tabs(["Tab A", "Tab B"])
with t1:
    st.write("conteúdo da tab A")
with t2:
    st.write("conteúdo da tab B")

st.sidebar.markdown("---")
st.sidebar.markdown("**Sidebar depois das tabs**")
