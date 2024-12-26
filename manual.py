from pathlib import Path
import streamlit as st

st.title(':material/menu_book: User Manual')
st.divider()

st.markdown(Path('./chatgpt.md').read_text(), unsafe_allow_html=True)