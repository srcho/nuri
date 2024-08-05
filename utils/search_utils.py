import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def save_faiss_index(index: faiss.IndexFlatL2, file_path: str):
    faiss.write_index(index, file_path)

def load_faiss_index(file_path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(file_path)

def search_similar_docs(index: faiss.IndexFlatL2, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

def encode_query(query: str, embedding_dim: int) -> np.ndarray:
    query_embedding = model.encode([query])
    padding = np.zeros((1, embedding_dim - query_embedding.shape[1]))
    return np.concatenate((query_embedding, padding), axis=1)