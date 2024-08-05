import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import load_data, prepare_faiss_index
from utils.search_utils import create_faiss_index, save_faiss_index, load_faiss_index, search_similar_docs, encode_query

@st.cache_data
def load_and_process_data():
    df = load_data("data/nuri_des.csv")
    embeddings, doc_ids, titles = prepare_faiss_index(df)
    return df, embeddings, doc_ids, titles

@st.cache_resource
def get_faiss_index(embeddings):
    index = create_faiss_index(embeddings)
    save_faiss_index(index, "data/faiss_index.bin")
    return index

def display_results(results: pd.DataFrame, top_k: int):
    for _, row in results.iterrows():
        st.write(f"제목: {row['title']}")
        st.write(f"초록: {row['abstract.Element:Text'][:200]}...")
        st.write("---")

def main():
    st.title("AI 논문 검색 엔진")
    
    df, embeddings, doc_ids, titles = load_and_process_data()
    index = get_faiss_index(embeddings)
    
    st.write(f"총 {len(doc_ids)}개의 문서가 로드되었습니다.")
    st.write(f"임베딩 차원: {embeddings.shape[1]}")

    st.sidebar.title("검색 옵션")
    search_type = st.sidebar.radio("검색 유형", ("키워드", "의미론적"))
    top_k = st.sidebar.slider("표시할 결과 수", 1, 20, 5)

    query = st.text_input("검색어를 입력하세요")

    if st.button("검색"):
        if query:
            if search_type == "키워드":
                results = df[df['title'].str.contains(query, case=False) | df['abstract.Element:Text'].str.contains(query, case=False)]
                display_results(results.head(top_k), top_k)
            else:
                query_vector = encode_query(query, embeddings.shape[1])
                st.write(f"쿼리 벡터 차원: {query_vector.shape[1]}")
                distances, indices = search_similar_docs(index, query_vector, top_k)
                results = df.iloc[indices[0]]
                display_results(results, top_k)
        else:
            st.warning("검색어를 입력해주세요.")

if __name__ == "__main__":
    main()