# file: main.py
import os
import streamlit as st
from side import sidebar
from tools import (
    embed_docs,
    get_answer,
    parse_docx,
    parse_pdf,
    parse_txt,
    search_docs,
    text_to_docs,
)
import emaill

st.set_page_config(page_title="Research Buddy", page_icon="📖")
sidebar()

col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists("iconn.png"):
        st.image("iconn.png", width=64)
with col2:
    st.markdown("<h1 style='margin-bottom:-5%;'>Research<span style='color:#F55F0E;'> Buddy</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='padding-bottom:10%'>~Effortless Happy Research</p>", unsafe_allow_html=True)

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

uploaded_file = st.file_uploader("Upload a pdf, docx, or txt file", type=["pdf", "docx", "txt"])

inx = None
data = None
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        data = parse_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".docx"):
        data = parse_docx(uploaded_file)
    elif uploaded_file.name.lower().endswith(".txt"):
        data = parse_txt(uploaded_file)
    else:
        st.error("File type not supported!")
    docs = text_to_docs(data)
    with st.spinner("Generating embeddings (using OpenAI)..."):
        try:
            inx = embed_docs(docs, st.session_state.get("OPENAI_API_KEY"))
            st.success("Indexing finished.")
        except Exception as e:
            st.error(f"Embedding error: {e}")

ques = st.text_area("Ask your question about the document")
if st.button("Submit"):
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Set your OpenAI API key in the sidebar first.")
    elif not inx:
        st.error("Please upload a document first.")
    elif not ques or len(ques.strip()) == 0:
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching and generating answer..."):
            sources = search_docs(inx, ques, top_k=5)
            try:
                answer = get_answer(sources, ques, st.session_state.get("OPENAI_API_KEY"))
                st.markdown("#### Answer")
                st.markdown(answer["text"])
                st.markdown("#### Sources")
                if not sources:
                    st.write("No relevant sources found.")
                else:
                    for s in sources:
                        st.markdown(f"- Source `{s['meta']}` — {s['text'][:300]}...")
            except Exception as e:
                st.error(f"OpenAI completion error: {e}")

st.markdown("---")
st.info("To email the Answer, enter your email id and click Send.")
email = st.text_input("Enter your email id to receive the answer")
if st.button("Send"):
    # courier auth token should be stored in Streamlit secrets or enter manually in sidebar
    auth = st.secrets.get("AUTH_TOKEN", None) or st.session_state.get("COURIER_AUTH", "")
    if not auth:
        st.error("Courier AUTH_TOKEN missing. Put it in Streamlit Secrets (AUTH_TOKEN) or enter in sidebar.")
    else:
        try:
            # what to send — try to read last answer stored
            last_ans = st.session_state.get("last_answer", "")
            if not last_ans:
                st.error("No answer to send yet. Submit a question first.")
            else:
                success = emaill.send_email(email, last_ans, auth)
                if success:
                    st.success("Mail Sent Successfully!")
                else:
                    st.error("Sending failed.")
        except Exception as e:
            st.error(f"Error sending email: {e}")
