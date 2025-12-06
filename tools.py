# tools.py — pure-python vector math (no numpy) + OpenAI calls
import re
from io import BytesIO
from typing import List, Dict
import docx2txt
from pypdf import PdfReader
import openai
import streamlit as st
import math

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

# ---------- VECTOR HELPERS (pure python) ---------- #

def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

def norm(a: List[float]) -> float:
    return math.sqrt(sum(x*x for x in a)) + 1e-12

# ---------- EMBEDDINGS + SEARCH ---------- #

def embed_docs(docs: List[Dict], api_key: str):
    if not api_key:
        raise ValueError("Missing OpenAI API key.")
    openai.api_key = api_key
    texts = [d["text"] for d in docs]
    res = openai.Embedding.create(model="text-embedding-3-small", input=texts)
    embeddings = [item["embedding"] for item in res["data"]]
    return {"docs": docs, "emb": embeddings}

def search_docs(index, query: str, top_k=5):
    api_key = st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in session_state")
    openai.api_key = api_key
    q_emb = openai.Embedding.create(model="text-embedding-3-small", input=[query])["data"][0]["embedding"]
    scores = []
    for emb in index["emb"]:
        s = dot(emb, q_emb) / (norm(emb) * norm(q_emb))
        scores.append(s)
    # get top_k indices
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in idxs:
        results.append({
            "text": index["docs"][i]["text"],
            "meta": index["docs"][i]["meta"],
            "score": float(scores[i])
        })
    return results

# ---------- ANSWER ---------- #

def get_answer(sources: List[Dict], question: str, api_key: str):
    openai.api_key = api_key
    context = "\n\n".join(f"[{s['meta']}] {s['text']}" for s in sources)
    prompt = (
        "Use the following document excerpts to answer the question.\n"
        "If uncertain, say 'I do not know'.\n\n"
        f"QUESTION: {question}\n\n"
        f"DOCUMENTS:\n{context}\n\n"
        "FINAL ANSWER:"
    )
    # Use ChatCompletion fallback (gpt-3.5-turbo) — safest for most users
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        ans = res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # fallback to completions (older API) if necessary
        comp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=512, temperature=0)
        ans = comp["choices"][0]["text"].strip()
    st.session_state["last_answer"] = ans
    return {"text": ans}
