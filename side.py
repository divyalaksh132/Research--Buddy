import streamlit as st

def sidebar():
    with st.sidebar:
        st.header("🔑 API Settings")

        api = st.text_input("OpenAI API Key", type="password")
        if api:
            st.session_state["OPENAI_API_KEY"] = api

        courier = st.text_input("Courier AUTH Token (optional)", type="password")
        if courier:
            st.session_state["COURIER_AUTH"] = courier

        st.markdown("---")
        st.write("Upload a document in the main window.")
