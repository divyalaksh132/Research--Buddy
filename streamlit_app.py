import os
import streamlit as st
from side import sidebar
from tools import (
    parse_pdf,
    parse_docx,
    parse_txt,
    text_to_docs,
    embed_docs,
    search_docs,
    get_answer
)
import emaill

st.set_page_config(page_title="Research Buddy", page_icon="📖")

# Sidebar (OpenAI key + optional Courier token)
sidebar()

st.title("📖 Research Buddy")
st.write("Upload a file and ask questions!")

uploaded_file = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])

index = None
docs = None

if uploaded_file:
    ext = uploaded_file.name.lower()

    if ext.endswith(".pdf"):
        data = parse_pdf(uploaded_file)
    elif ext.endswith(".docx"):
        data = parse_docx(uploaded_file)
    elif ext.endswith(".txt"):
        data = parse_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    docs = text_to_docs(data)

    with st.spinner("Generating embeddings…"):
        try:
            index = embed_docs(docs, st.session_state["OPENAI_API_KEY"])
            st.success("Document indexed!")
        except Exception as e:
            st.error(f"Embedding error: {e}")

question = st.text_area("Ask a question about the document")

if st.button("Submit"):
    if not index:
        st.error("Upload a document first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Thinking…"):
            sources = search_docs(index, question, top_k=5)
            answer = get_answer(sources, question, st.session_state["OPENAI_API_KEY"])

        st.subheader("Answer")
        st.write(answer["text"])

        st.subheader("Sources")
        if sources:
            for s in sources:
                st.markdown(f"- **{s['meta']}** → {s['text'][:250]}…")
        else:
            st.write("No relevant sources found.")

# EMAIL SECTION
st.markdown("---")
st.write("### Email the answer")

email = st.text_input("Enter your email")

if st.button("Send Email"):
    if "last_answer" not in st.session_state:
        st.error("Ask a question before sending an email.")
    else:
        token = st.session_state.get("COURIER_AUTH") or st.secrets.get("AUTH_TOKEN")

        if not token:
            st.error("Courier AUTH_TOKEN missing.")
        else:
            try:
                emaill.send_email(email, st.session_state["last_answer"], token)
                st.success("Email sent!")
            except Exception as e:
                st.error(f"Email error: {e}")
