import streamlit as st

import pandas as pd
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer,CrossEncoder
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
def load_reranker_model():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
reranker=load_reranker_model()

def search_paper(query, k=50, q=10):

    # -----------------------------------------
    # 1. Create query embedding
    # -----------------------------------------
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # -----------------------------------------
    # 2. FAISS retrieves candidate papers
    # -----------------------------------------
    scores, indices = index.search(
        query_embedding,
        k
    )

    # -----------------------------------------
    # 3. Create query-paper pairs
    # -----------------------------------------
    pairs = []
    valid_indices = []

    for idx in indices[0]:

        if idx == -1:
            continue

        paper_text = (
            str(df.iloc[idx]["title"])
            + " "
            + str(df.iloc[idx]["abstract"])
        )

        pairs.append([
            query,
            paper_text
        ])

        valid_indices.append(idx)

    # -----------------------------------------
    # 4. CrossEncoder reranking
    # -----------------------------------------
    rerank_scores = reranker.predict(pairs)

    # -----------------------------------------
    # 5. Sort by reranker score
    # -----------------------------------------
    ranked = sorted(
        zip(valid_indices, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------------------
    # 6. Return final top-q results
    # -----------------------------------------
    results = []

    for idx, score in ranked[:q]:

        results.append({
            "index": int(idx),
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