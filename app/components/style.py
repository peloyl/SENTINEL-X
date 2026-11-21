
import streamlit as st
def apply_style():
    st.markdown("""<style>
    .stApp{background:#0b0f14;color:#e6edf3} section[data-testid=stSidebar]{background:#0f1620}
    h1,h2,h3{color:#e6edf3;letter-spacing:.02em} .stMetric{background:#111a26;border:1px solid #1f2d3d;border-radius:8px;padding:8px}
    div[data-testid=stExpander]{background:#111a26;border:1px solid #1f2d3d}
    </style>""", unsafe_allow_html=True)
