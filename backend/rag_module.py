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
from transformers import BertModel, BertTokenizer
from collections import Counter
import torch

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# KoBERT 임베딩 클래스 정의
class KoBERTEmbeddings:
    def __init__(self):
        # KoBERT 토크나이저와 모델 로드
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        self.model = BertModel.from_pretrained("monologg/kobert")

    def __call__(self, texts):
        return self.embed_documents(texts)

    def embed_documents(self, texts):
        # 텍스트를 토큰화하고 임베딩 생성
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output.last_hidden_state[:, 0, :].numpy()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# RAG 시스템 클래스 정의
class RAGSystem:
    def __init__(self):
        # Kiwi 토크나이저, KoBERT 임베딩, ChatGPT 모델 초기화
        self.kiwi = Kiwi()
        self.embeddings = KoBERTEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.load_data()
        self.setup_retrieval()

    def tokenize_text(self, text):
        # 텍스트를 토큰화하고 불필요한 태그 제거
        tokens = self.kiwi.tokenize(text)
        return " ".join([token.form for token in tokens if token.tag not in ['SF', 'SP', 'SS', 'SE']])

    def remove_stopwords(self, text):
        # 불용어 제거
        stopwords = set(['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로'])
        return ' '.join([word for word in text.split() if word not in stopwords])

    def load_data(self):
        # 데이터 로드 및 전처리
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '../data/nuri_mod3.xlsx')
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} rows from Excel file")
        
        self.documents = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading documents"):
            if pd.notna(row['title']) and pd.notna(row['abstract.Element:Text']):
                processed_title = self.tokenize_text(row['title'])
                processed_abstract = self.tokenize_text(row['abstract.Element:Text'])
                
                processed_keywords = ""
                if 'keywords' in row and pd.notna(row['keywords']):
                    processed_keywords = self.tokenize_text(row['keywords'])
                
                processed_content = f"{processed_title} {processed_abstract} {processed_keywords}"
                processed_content = self.remove_stopwords(processed_content)
                self.documents.append(
                    Document(
                        page_content=processed_content,
                        metadata={
                            "title": row['title'],
                            "authors": row['authors'] if pd.notna(row['authors']) else "정보 없음",
                            "url": row['URL'] if pd.notna(row['URL']) else "정보 없음",
                            "abstract": row['abstract.Element:Text'],
                            "processed_content": processed_content
                        }
                    )
                )
        logger.info(f"Created {len(self.documents)} valid documents")

    def calculate_directory_hash(self, directory):
        # 디렉토리의 해시값 계산
        sha256_hash = hashlib.sha256()
        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def setup_retrieval(self):
        # FAISS 인덱스 디렉토리와 해시 파일 경로 설정
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
        self.bm25_retriever.k = 10

        self.retriever = self.create_ensemble_retriever()

    def create_new_index(self, index_dir, hash_file):
        # FAISS 인덱스 생성 및 저장
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.vectorstore.save_local(index_dir)
        
        directory_hash = self.calculate_directory_hash(index_dir)
        with open(hash_file, 'w') as f:
            f.write(directory_hash)

        logger.info(f"New FAISS index created and hash saved: {directory_hash}")

    def expand_query(self, query):
        # 쿼리 확장
        tokens = self.kiwi.tokenize(query)
        content_words = [token.form for token in tokens if token.tag.startswith('N') or token.tag.startswith('V')]
        top_words = [word for word, _ in Counter(content_words).most_common(3)]
        expanded_query = query + " " + " ".join(top_words)
        return expanded_query

    def calculate_semantic_similarity(self, query, results):
        # 의미 유사도 계산
        query_embedding = self.embeddings.embed_query(query)
        semantic_results = []
        for result in results:
            try:
                if isinstance(result, Document):
                    doc = result
                    score = 1.0  # 기본 점수
                elif isinstance(result, tuple):
                    if len(result) == 2:
                        doc, score = result
                    else:
                        logger.warning(f"Unexpected tuple length in result: {len(result)}")
                        continue
                elif isinstance(result, dict) and 'document' in result and 'score' in result:
                    doc = result['document']
                    score = result['score']
                else:
                    logger.warning(f"Unexpected result type: {type(result)}")
                    continue
                
                doc_embedding = self.embeddings.embed_query(doc.page_content)
                semantic_score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                semantic_results.append({'doc': doc, 'score': semantic_score * score})
            except Exception as e:
                logger.error(f"Error processing result in calculate_semantic_similarity: {e}")
                continue
        return semantic_results

    def create_ensemble_retriever(self):
        # 앙상블 검색 함수 생성
        def ensemble_retrieve(query):
            logger.info(f"Original query: {query}")
            processed_query = self.tokenize_text(query)
            logger.info(f"Processed query: {processed_query}")
            expanded_query = self.expand_query(processed_query)
            logger.info(f"Expanded query: {expanded_query}")
            
            bm25_results = self.bm25_retriever.invoke(expanded_query)
            logger.info(f"Number of BM25 results: {len(bm25_results)}")
            
            try:
                faiss_results = self.vectorstore.similarity_search_with_score(expanded_query, k=20)
                logger.info(f"Number of FAISS results: {len(faiss_results)}")
            except Exception as e:
                logger.error(f"Error in FAISS search: {e}")
                faiss_results = []
            
            semantic_results = self.calculate_semantic_similarity(expanded_query, faiss_results)
            logger.info(f"Number of semantic results: {len(semantic_results)}")
            
            combined_results = {}
            
            for doc in bm25_results:
                key = doc.page_content
                if key not in combined_results:
                    combined_results[key] = {'doc': doc, 'score': 1}
                else:
                    combined_results[key]['score'] += 1
            
            for result in semantic_results:
                key = result['doc'].page_content
                if key not in combined_results:
                    combined_results[key] = result
                else:
                    combined_results[key]['score'] += result['score']

            final_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)
            logger.info(f"Final number of results: {len(final_results)}")
            return final_results[:10]
        
        return ensemble_retrieve

    def calculate_tfidf_weight(self, doc, query):
        # TF-IDF 가중치 계산
        query_terms = set(query.split())
        doc_terms = set(doc.page_content.split())
        common_terms = query_terms.intersection(doc_terms)
        tf = len(common_terms) / len(doc_terms)
        idf = math.log(len(self.documents) / (1 + sum(1 for d in self.documents if any(term in d.page_content for term in query_terms))))
        return tf * idf

    def calculate_dynamic_threshold(self, similar_docs):
        # 동적 임계값 계산
        if not similar_docs:
            return 0.75
        scores = [result['score'] for result in similar_docs]
        return max(0.5, np.mean(scores) - np.std(scores))

    def answer_question(self, question):
        try:
            # 질문에 대한 유사 문서 검색
            similar_docs = self.retriever(question)
            logger.info(f"Retrieved {len(similar_docs)} documents")
            
            sources = []            
            # 동적 임계값 계산
            similarity_threshold = self.calculate_dynamic_threshold(similar_docs)
            for result in similar_docs:
                if isinstance(result, dict) and 'doc' in result and 'score' in result:
                    doc = result['doc']
                    similarity = result['score']
                    if similarity >= similarity_threshold:
                        metadata = self.safe_serialize_metadata(doc.metadata)
                        metadata['similarity'] = similarity
                        sources.append(metadata)
                else:
                    logger.warning(f"Unexpected result format: {result}")

            logger.info(f"Found {len(sources)} relevant documents with similarity >= {similarity_threshold}")

            if not sources:
                return {
                    "answer": "관련된 정보를 찾을 수 없습니다.",
                    "sources": []
                }

            abstracts = [source['abstract'] for source in sources]
            logger.info(f"Number of abstracts: {len(abstracts)}")
            logger.info(f"First abstract: {abstracts[0][:100]}...")

            prompt = PromptTemplate(
                input_variables=["question", "abstracts"],
                template="""You are a top-notch data analytics expert. The information below is an abstract of an article relevant to your question. Please use only this information to answer your question.
                - Do not include anything that is not in the article abstract information provided. 
                - Don't speculate or provide inaccurate information.
                - If you don't have enough information to answer the question based on the article abstract information provided, answer ‘nodata’.
                - Refer to the article abstract information to provide information about the question.
                - Please provide text only.
                - Must Include article references for each sentence. e.g., [6], [7&9]
                - Indicate source articles for each summary sentence strictly based on a generated article number. e.g., [6], [7&9]
                - Need to be very accurate. Ensure accuracy of key points.
                - Summarize the key contents. Include enough detail to convey the main ideas effectively.
                - Use bulletpoints to structure the answer like following example.
                - answer in Korean.
                
                Question: {question}

                Related paper abstracts:
                {abstracts}
                """
            )

            formatted_prompt = prompt.format(question=question, abstracts='\n'.join(abstracts))
            logger.info(f"Formatted prompt: {formatted_prompt[:500]}...")

            ai_response = self.llm.invoke(formatted_prompt)
            
            logger.info(f"Raw AI response: {ai_response}")
            
            answer = ai_response.content
            logger.info(f"AI response content: {answer}")
            
            logger.info(f"GPT query successful. Response length: {len(answer)}")
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            logger.error(f"Error occurred at line: {e.__traceback__.tb_lineno}")
            raise

    @staticmethod
    def safe_serialize_metadata(metadata):
        # 메타데이터 안전하게 직렬화
        return {
            "title": RAGSystem.safe_serialize_value(metadata.get('title')),
            "authors": RAGSystem.safe_serialize_value(metadata.get('authors')),
            "url": RAGSystem.safe_serialize_value(metadata.get('url')),
            "abstract": RAGSystem.safe_serialize_value(metadata.get('abstract'))
        }

    @staticmethod
    def safe_serialize_value(value):
        # 값 안전하게 직렬화
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "정보 없음"
        try:
            json.dumps(value)
            return value if value != "" else "정보 없음"
            
        except (TypeError, OverflowError, ValueError):
                return str(value)

# RAG 시스템 초기화
print("Initializing RAG system...")
rag_system = RAGSystem()
print("RAG system initialized and ready for queries.")