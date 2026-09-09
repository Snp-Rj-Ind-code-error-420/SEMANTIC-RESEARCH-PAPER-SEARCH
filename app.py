import streamlit as st

import pandas as pd
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

@st.cache_data
def load_papers():
    dataset = load_dataset(
        'CShorten/ML-ArXiv-Papers',
        split="train"
    )
    return dataset

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_index():

    index_path = hf_hub_download(
        repo_id="lensy402/paper-search-index",
        filename="paper_faiss.index",
        repo_type="dataset"
    )

    return faiss.read_index(index_path)

dataset = load_papers()
df=pd.DataFrame(dataset)
index = load_index()
embedding_model = load_embedding_model()

def search_paper(query, k=5):

    # Create query embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []
    for score, idx in zip(scores[0], indices[0]):

        # FAISS can return -1 if there aren't enough results
        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"]
        })

    return results



st.set_page_config(
page_title="SEMANTIC RESEARCH PAPER SEARCH",
 page_icon=":spiral_notepad:",
  layout="centered",
   initial_sidebar_state=None,
    menu_items=None
	)
st.title(body="SEMANTIC RESEARCH PAPER SEARCH 📜")
query=st.text_input(label="input_for_res",
 		label_visibility="hidden",
  
           placeholder="ENTER YOUR TOPIC OF RESEARCH 📜"
           )
if st.button("🔍 SEARCH"):

    if not query.strip():

        st.warning("Please enter a research topic.")

    else:

        with st.spinner("Searching papers..."):

            results = search_paper(query, k=5)

        st.subheader("🔎 Search Results")

        for i, result in enumerate(results, start=1):

            st.markdown(
                f"### {i}. {result['title']}"
            )

            st.caption(
                f"Similarity Score: {result['score']:.4f}"
            )

            st.write(
                result["abstract"][:500] + "..."
            )

            st.divider()