import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------- Loaders ----------------------

@st.cache_resource(show_spinner=True)
def load_sections(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        sections = json.load(f)
    return sections

@st.cache_resource(show_spinner=True)
def embed_sections(sections, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    corpus = [
        f"{sec['section_number']}. {sec['title']} {sec['content']}"
        for sec in sections
    ]
    embeddings = model.encode(corpus, show_progress_bar=False, normalize_embeddings=True)
    return model, np.array(embeddings)

@st.cache_resource(show_spinner=True)
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine similarity
    index.add(embeddings)
    return index

@st.cache_resource(show_spinner=True)
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ---------------------- Search Logic ----------------------

def search_sections(query, model, index, sections, summarizer, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_emb), top_k)
    results = []

    for idx, score in zip(I[0], D[0]):
        sec = sections[idx]
        raw_content = sec["content"]
        try:
            summary = summarizer(raw_content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        except Exception:
            summary = raw_content[:300] + "..." if len(raw_content) > 300 else raw_content

        results.append({
            "section_number": sec["section_number"],
            "title": sec["title"],
            "summary": summary,
            "score": float(score),
        })
    return results

# ---------------------- Streamlit App ----------------------

st.set_page_config(page_title="IPC Section Retriever", layout="wide")
st.title("🔍 Legal Section Finder from FIR-like Descriptions")

st.markdown("Enter a short incident description or FIR-like story. This app will retrieve the most relevant Indian Penal Code (IPC) sections.")

query = st.text_area("✍️ Enter FIR Description", height=200, placeholder="Example: A man broke into a house at night and stole jewellery and cash...")

if st.button("Find Relevant IPC Sections") and query.strip():
    with st.spinner("Searching and Summarizing..."):
        json_file = "all_sections_1_to_358.json"
        sections = load_sections(json_file)
        model, embeddings = embed_sections(sections)
        index = build_faiss_index(embeddings)
        summarizer = load_summarizer()
        results = search_sections(query, model, index, sections, summarizer, top_k=5)

    st.subheader("📜 Top 5 IPC Sections Relevant to Your Query")
    for i, res in enumerate(results, 1):
        st.markdown(f"### {i}. Section {res['section_number']}: {res['title']}")
        st.markdown(f"**Score:** `{res['score']:.4f}`")
        st.markdown(f"**Summary:** {res['summary']}")
        st.markdown("---")

elif query.strip() == "":
    st.info("Please enter an FIR or incident to begin searching.")
