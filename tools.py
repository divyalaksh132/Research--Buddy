# file: tools.py
import re
from io import BytesIO
from typing import List, Dict
import docx2txt
import numpy as np
import requests
from pypdf import PdfReader
import openai
import streamlit as st

def parse_docx(file: BytesIO) -> List[str]:
    text = docx2txt.process(file)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return [text]

def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def parse_txt(file: BytesIO) -> List[str]:
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1")
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return [text]

def text_to_docs(texts: List[str]) -> List[Dict]:
    docs = []
    for i, page in enumerate(texts):
        if not page:
            continue
        # split into chunks ~800 chars
        start = 0
        while start < len(page):
            chunk = page[start : start + 800]
            docs.append({"text": chunk, "meta": f"page-{i+1}-chunk-{start//800}"})
            start += 800
    return docs

def _get_openai_client(api_key: str):
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing")
    openai.api_key = api_key
    return openai

def _embed_texts(openai_client, texts: List[str]) -> List[List[float]]:
    # uses OpenAI embeddings endpoint
    res = openai_client.Embedding.create(input=texts, model="text-embedding-3-small")
    return [r["embedding"] for r in res["data"]]

def embed_docs(docs: List[Dict], api_key: str):
    client = _get_openai_client(api_key)
    texts = [d["text"] for d in docs]
    embeddings = _embed_texts(client, texts)
    # store vectors on the index (just a list)
    index = {"docs": docs, "embeddings": np.array(embeddings)}
    st.session_state["__last_index__"] = index
    return index

def search_docs(index, query: str, top_k: int = 3):
    client = _get_openai_client(st.session_state.get("OPENAI_API_KEY") or "")
    q_emb = _embed_texts(client, [query])[0]
    vecs = index["embeddings"]
    dots = np.dot(vecs, np.array(q_emb))
    norms = np.linalg.norm(vecs, axis=1) * (np.linalg.norm(q_emb) + 1e-12)
    sims = dots / norms
    top_idx = list(np.argsort(-sims)[:top_k])
    results = []
    for i in top_idx:
        results.append({"text": index["docs"][i]["text"], "meta": index["docs"][i]["meta"], "score": float(sims[i])})
    return results

def get_answer(sources: List[Dict], query: str, api_key: str) -> Dict:
    client = _get_openai_client(api_key)
    # Build context from sources (concatenate top 5)
    context = "\n\n".join([f"[SOURCE: {s['meta']}]\n{s['text']}" for s in sources])
    prompt = (
        f"Use the following document excerpts as references. If you cannot answer, say 'I do not know'.\n\n"
        f"QUESTION: {query}\n\nDOCUMENTS:\n{context}\n\nFINAL ANSWER:"
    )
    completion = client.ChatCompletion.create(
        model="gpt-4o-mini",  # if not available for you, change to "gpt-4o" or "gpt-3.5-turbo"
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    text = completion["choices"][0]["message"]["content"].strip()
    st.session_state["last_answer"] = text
    return {"text": text}
