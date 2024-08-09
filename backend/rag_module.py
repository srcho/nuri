import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm
import json
import math
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.load_data()
        self.setup_retrieval()

    def load_data(self):
        df = pd.read_excel('../data/nuri_mod.xlsx')
        self.documents = [
            Document(
                page_content=f"{row['title']} {row['abstract.Element:Text']}",
                metadata={
                    "title": row['title'],
                    "authors": row['authors'],
                    "url": row['URL'],
                    "abstract": row['abstract.Element:Text']
                }
            ) for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading documents")
        ]

    def setup_retrieval(self):
        if os.path.exists("faiss_index"):
            self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            self.vectorstore.save_local("faiss_index")
        
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", k=5)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def answer_question(self, question):
        try:
            result = self.qa_chain({"query": question})
            sources = [self.safe_serialize_metadata(doc.metadata) for doc in result['source_documents']]
            
            abstracts = [source['abstract'] for source in sources[:5]]
            prompt = f"""다음은 사용자의 질문과 관련된 논문 초록입니다. 이 정보를 바탕으로 사용자의 질문에 답변해주세요.

질문: {question}

관련 논문 초록:
{' '.join(abstracts)}

답변:"""

            answer = self.llm.predict(prompt)
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
            return str(value)
        try:
            json.dumps(value)
            return value
        except (TypeError, OverflowError, ValueError):
            return str(value)

print("Initializing RAG system...")
rag_system = RAGSystem()
print("RAG system initialized and ready for queries.")