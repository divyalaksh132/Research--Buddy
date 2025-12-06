import re
from io import BytesIO
from typing import List, Dict
import numpy as np
import docx2txt
from pypdf import PdfReader
import openai
import streamlit as st


# ---------- FILE PARSING ---------- #

def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    pages = []
    for p in pdf.pages:
        text = p.extract_text() or ""
        text = text.replace("\n", " ")
        pages.append(text)
    return pages

def parse_docx(file: BytesIO) -> List[str]:
    text = docx2txt.process(file)
    return [text.replace("\n", " ")]

def parse_txt(file: BytesIO) -> List[str]:
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except:
        text = raw.decode("latin-1")
    return [text.replace("\n", " ")]


# ---------- CHUNKING ---------- #

def text_to_docs(pages: List[str]) -> List[Dict]:
    docs = []
    for i, page in enumerate(pages):
        start = 0
        while start < len(page):
            chunk = page[start:start+800]
            docs.append({
                "text": chunk,
                "meta": f"page-{i+1}-chunk-{start//800}"
            })
            start += 800
    return docs


# ---------- EMBEDDINGS + SEARCH ---------- #

def embed_docs(docs: List[Dict], api_key: str):
    if not api_key:
        raise ValueError("Missing OpenAI API key.")

    openai.api_key = api_key
    texts = [d["text"] for d in docs]

    res = openai.Embedding.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = np.array([x["embedding"] for x in res["data"]])

    return {"docs": docs, "emb": embeddings}


def search_docs(index, query: str, top_k=5):
    openai.api_key = st.session_state["OPENAI_API_KEY"]

    q_emb = openai.Embedding.create(
        model="text-embedding-3-small",
        input=[query]
    )["data"][0]["embedding"]

    q_emb = np.array(q_emb)

    A = index["emb"]
    scores = A @ q_emb / (np.linalg.norm(A, axis=1) * np.linalg.norm(q_emb))

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_idx:
        results.append({
            "text": index["docs"][i]["text"],
            "meta": index["docs"][i]["meta"],
            "score": float(scores[i])
        })

    return results


# ---------- ANSWER GENERATION ---------- #

def get_answer(sources: List[Dict], question: str, api_key: str):
    openai.api_key = api_key

    context = "\n\n".join(
        f"[{s['meta']}] {s['text']}" for s in sources
    )

    prompt = (
        "Use the following document excerpts to answer the question.\n"
        "If uncertain, say 'I do not know'.\n\n"
        f"QUESTION: {question}\n\n"
        f"DOCUMENTS:\n{context}\n\n"
        "FINAL ANSWER:"
    )

    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",   # use gpt-3.5-turbo if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    ans = res["choices"][0]["message"]["content"]
    st.session_state["last_answer"] = ans

    return {"text": ans}
