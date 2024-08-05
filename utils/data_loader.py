import pandas as pd
import numpy as np
from typing import List, Tuple

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.fillna('')
    columns_to_use = ['node_id', 'title', 'abstract.Element:Text', 'title_embedding', 'abstract_embedding']
    df = df[columns_to_use]
    return df

def prepare_faiss_index(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    title_embeddings = np.array([eval(emb) for emb in df['title_embedding']])
    abstract_embeddings = np.array([eval(emb) for emb in df['abstract_embedding']])
    combined_embeddings = np.concatenate((title_embeddings, abstract_embeddings), axis=1)
    doc_ids = df['node_id'].tolist()
    titles = df['title'].tolist()
    return combined_embeddings, doc_ids, titles