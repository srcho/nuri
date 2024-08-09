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

    def query(self, question):
        result = self.qa_chain({"query": question})
        return {
            "answer": result['result'],
            "sources": [
                self.safe_serialize_metadata(doc.metadata)
                for doc in result['source_documents']
            ]
        }

    def gpt_query(self, prompt):
        try:
            response = self.llm.predict(prompt)
            logger.info(f"GPT query successful. Response length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Error in gpt_query: {str(e)}")
            raise

    def generate_summary(self, abstracts):
        try:
            prompt = PromptTemplate.from_template(
                """다음은 최대 5개의 논문 초록입니다. 각 초록을 요약하고, 요약된 내용을 기반으로 종합적인 답변을 작성해 주세요.
                요약은 번호를 매겨 작성하고, 각 요약의 출처를 각주로 표시해 주세요.
                마지막에는 전체 내용을 한 문장으로 정리한 결론을 작성해 주세요.

                초록:
                {abstracts}

                요약 및 종합적인 답변:
                """
            )
            
            formatted_abstracts = "\n".join([f"{i+1}. {abstract}" for i, abstract in enumerate(abstracts)])
            final_prompt = prompt.format(abstracts=formatted_abstracts)
            
            response = self.llm.predict(final_prompt)
            logger.info(f"Summary generated successfully. Length: {len(response)}")
            return response
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            raise

    def answer_question(self, question):
        # RAG 시스템을 사용하여 관련 문서 검색
        search_result = self.query(question)
        
        # 검색된 문서의 초록을 사용하여 프롬프트 생성
        abstracts = [source['abstract'] for source in search_result['sources'][:5]]
        prompt = f"""다음은 사용자의 질문과 관련된 논문 초록입니다. 이 정보를 바탕으로 사용자의 질문에 답변해주세요.

질문: {question}

관련 논문 초록:
{' '.join(abstracts)}

답변:"""

        # GPT를 사용하여 답변 생성
        answer = self.gpt_query(prompt)
        
        return {
            "answer": answer,
            "sources": search_result['sources']
        }

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