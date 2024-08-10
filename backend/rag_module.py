import os
import shutil
import hashlib
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm
import json
import math
import logging
import numpy as np
from kiwipiepy import Kiwi
from sklearn.preprocessing import MinMaxScaler
from langchain_community.retrievers import BM25Retriever

load_dotenv()

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.kiwi = Kiwi()
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.load_data()
        self.setup_retrieval()

    def tokenize_text(self, text):
        tokens = self.kiwi.tokenize(text)
        return " ".join([token.form for token in tokens])

    def load_data(self):
        df = pd.read_excel('../data/nuri_mod2.xlsx')
        logger.info(f"Loaded {len(df)} rows from Excel file")
        
        self.documents = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading documents"):
            if pd.notna(row['title']) and pd.notna(row['abstract.Element:Text']):
                processed_title = self.tokenize_text(row['title'])
                processed_abstract = self.tokenize_text(row['abstract.Element:Text'])
                original_content = f"{row['title']} {row['abstract.Element:Text']}"
                processed_content = f"{processed_title} {processed_abstract}"
                self.documents.append(
                    Document(
                        page_content=processed_content,
                        metadata={
                            "title": row['title'],
                            "authors": row['authors'] if pd.notna(row['authors']) else "정보 없음",
                            "url": row['URL'] if pd.notna(row['URL']) else "정보 없음",
                            "abstract": row['abstract.Element:Text'],
                            "original_content": original_content
                        }
                    )
                )
        logger.info(f"Created {len(self.documents)} valid documents")

    def calculate_directory_hash(self, directory):
        sha256_hash = hashlib.sha256()
        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def setup_retrieval(self):
        index_dir = "faiss_index"
        hash_file = "faiss_index.hash"

        if os.path.exists(index_dir) and os.path.isdir(index_dir) and os.path.exists(hash_file):
            logger.info("Existing FAISS index and hash file found.")
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()

            current_hash = self.calculate_directory_hash(index_dir)

            if current_hash == stored_hash:
                logger.info("Hash matches, loading existing FAISS index.")
                self.vectorstore = FAISS.load_local(index_dir, self.embeddings, allow_dangerous_deserialization=True)
            else:
                logger.warning("Hash mismatch, recreating FAISS index.")
                self.create_new_index(index_dir, hash_file)
        else:
            logger.info("No valid FAISS index found, creating new one.")
            self.create_new_index(index_dir, hash_file)
        
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 5  # 상위 5개 결과 반환

        self.retriever = self.create_ensemble_retriever()

    def create_new_index(self, index_dir, hash_file):
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.vectorstore.save_local(index_dir)
        
        directory_hash = self.calculate_directory_hash(index_dir)
        with open(hash_file, 'w') as f:
            f.write(directory_hash)

        logger.info(f"New FAISS index created and hash saved: {directory_hash}")

    def create_ensemble_retriever(self):
        def ensemble_retrieve(query):
            # BM25 검색 수행
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
            
            # FAISS 검색 수행
            faiss_results = self.vectorstore.similarity_search_with_score(query, k=20)
            
            # 결과 결합 및 중복 제거
            combined_results = {}
            for doc in bm25_results:
                combined_results[doc.page_content] = {'doc': doc, 'bm25_score': 1, 'faiss_score': 1}
            
            for doc, faiss_score in faiss_results:
                if doc.page_content in combined_results:
                    combined_results[doc.page_content]['faiss_score'] = faiss_score
                else:
                    combined_results[doc.page_content] = {'doc': doc, 'bm25_score': 0, 'faiss_score': faiss_score}
            
            # 점수 정규화 및 최종 점수 계산
            bm25_scores = [result['bm25_score'] for result in combined_results.values()]
            faiss_scores = [result['faiss_score'] for result in combined_results.values()]
            
            # MinMaxScaler를 사용하여 점수 정규화
            bm25_scaler = MinMaxScaler()
            faiss_scaler = MinMaxScaler()
            
            bm25_normalized = bm25_scaler.fit_transform(np.array(bm25_scores).reshape(-1, 1)).flatten()
            faiss_normalized = 1 - faiss_scaler.fit_transform(np.array(faiss_scores).reshape(-1, 1)).flatten()
            
            # 점수 결합
            final_results = []
            for i, (key, result) in enumerate(combined_results.items()):
                bm25_score = bm25_normalized[i]
                faiss_score = faiss_normalized[i]
                
                # 로그 스케일링 적용
                log_bm25 = np.log1p(bm25_score)
                log_faiss = np.log1p(faiss_score)
                
                # 기하 평균 계산
                geometric_mean = np.sqrt(log_bm25 * log_faiss)
                
                # TF-IDF 가중치 적용 (예시)
                tfidf_weight = self.calculate_tfidf_weight(result['doc'], query)
                
                # 최종 점수 계산
                final_score = geometric_mean * (1 + tfidf_weight)
                
                final_results.append((result['doc'], final_score))
            
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            return final_results[:10]  # 상위 10개 결과 반환
        
        return ensemble_retrieve

    def calculate_tfidf_weight(self, doc, query):
        # TF-IDF 가중치 계산 로직 구현
        # 이 부분은 실제 TF-IDF 구현에 따라 달라질 수 있습니다
        # 간단한 예시:
        query_terms = set(query.lower().split())
        doc_terms = set(doc.page_content.lower().split())
        common_terms = query_terms.intersection(doc_terms)
        return len(common_terms) / len(query_terms)

    def answer_question(self, question):
        try:
            similar_docs = self.retriever(question)
            sources = []
            
            for doc, similarity in similar_docs:
                metadata = self.safe_serialize_metadata(doc.metadata)
                metadata['similarity'] = round(similarity, 2)  # 백분율로 변환하고 소수점 둘째 자리까지 표시
                sources.append(metadata)

            logger.info(f"Found {len(sources)} relevant documents")

            if not sources:
                return {
                    "answer": "관련된 정보를 찾을 수 없습니다.",
                    "sources": []
                }

            abstracts = [source['abstract'] for source in sources]
            prompt = PromptTemplate(
                input_variables=["question", "abstracts"],
                template="""다음은 사용자의 질문과 관련된 논문 초록입니다. 이 정보만을 바탕으로 사용자의 질문에 답변해주세요.
                
                - 제공된 논문 정보에 없는 내용은 절대 포함하지 마세요. 
                - 추측하거나 부정확한 정보를 제공하지 마세요.
                - 주어진 논문 정보로 질문에대한 정보가 없어 답하기 어렵다면, "nodata"라고 대답해주세요.
                
                질문: {question}

                관련 논문 초록:
                {abstracts}

                답변:"""
            )

            ai_response = self.llm.invoke(prompt.format(question=question, abstracts='\n'.join(abstracts)))
            answer = ai_response.content
            
            logger.info(f"GPT query successful. Response length: {len(answer)}")
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            raise

    @staticmethod
    def safe_serialize_metadata(metadata):
        return {
            "title": RAGSystem.safe_serialize_value(metadata.get('title')),
            "authors": RAGSystem.safe_serialize_value(metadata.get('authors')),
            "url": RAGSystem.safe_serialize_value(metadata.get('url')),
            "abstract": RAGSystem.safe_serialize_value(metadata.get('abstract'))
        }

    @staticmethod
    def safe_serialize_value(value):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "정보 없음"
        try:
            json.dumps(value)
            return value if value != "" else "정보 없음"
        except (TypeError, OverflowError, ValueError):
            return str(value)

print("Initializing RAG system...")
rag_system = RAGSystem()
print("RAG system initialized and ready for queries.")