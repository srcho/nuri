# src/app.py
import streamlit as st
import json
import faiss
import numpy as np
from transformers import BertModel, BertTokenizer
import torch

def load_papers(file_path):
    with open(file_path, 'r') as f:
        papers = json.load(f)
    return papers

def load_faiss_index(index_file):
    return faiss.read_index(index_file)

def search(query, papers, index, model, tokenizer):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    D, I = index.search(np.array([query_embedding]), k=5)
    results = [papers[i] for i in I[0]]
    return results

def main():
    st.title("AI-Powered Paper Search Engine")

    query = st.text_input("Enter your search query:")
    
    if query:
        papers = load_papers("data/processed_papers.json")
        index = load_faiss_index("data/faiss_index.index")

        model_name = 'beomi/kcbert-base'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        results = search(query, papers, index, model, tokenizer)
        
        st.write(f"Results for: {query}")
        for i, result in enumerate(results):
            st.write(f"**Result {i+1}: {result['title']}**")
            st.write(f"**Authors:** {result['authors']}")
            st.write(f"**Abstract:** {result['abstract']}")
            st.write("---")

if __name__ == "__main__":
    main()
