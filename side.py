# file: side.py
import streamlit as st

def sidebar():
    with st.sidebar:
        st.markdown("<h2 style='color:#F55F0E;'>Research Buddy</h2>", unsafe_allow_html=True)
        st.write("Upload a research doc and ask questions.")
        api = st.text_input("OpenAI API Key (sk-...)", type="password", key="OPENAI_API_KEY")
        if api:
            st.session_state["OPENAI_API_KEY"] = api
        courier = st.text_input("Courier AUTH_TOKEN (optional for email)", type="password", key="COURIER_AUTH")
        if courier:
            st.session_state["COURIER_AUTH"] = courier
        st.markdown("---")
