import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
from langchain_core.prompts import PromptTemplate
import os

# API 키 정보 로드
load_dotenv()

logging.langsmith("NURI", True)

# 엑셀 파일 불러오기
file_path = 'nuri_mod.xlsx'
df = pd.read_excel(file_path)

# 필요한 컬럼 추출
df_subset = df[['node_id', 'title', 'abstract.Element:Text', 'URL', 'authors']]

# 결합된 텍스트 컬럼 생성
df_subset['combined_text'] = df_subset['title'] + " " + df_subset['abstract.Element:Text']

# Document 객체 생성 시 메타데이터 추가
documents = [Document(page_content=row['combined_text'], metadata={
    'title': row['title'],
    'node_id': row['node_id'],
    'abstract': row['abstract.Element:Text'],
    'url': row['URL'],
    'authors': ', '.join(str(row['authors']).split(',')) if not pd.isna(row['authors']) else ''
}) for index, row in df_subset.iterrows()]

embedding_model = OpenAIEmbeddings()

# 벡터스토어가 이미 저장되어 있는 경우 불러오기, 없으면 생성 후 저장
index_file = "faiss_index"
if os.path.exists(index_file):
    vectorstore = FAISS.load_local(index_file, embeddings=embedding_model, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)
    vectorstore.save_local(index_file)

# 리트리버 생성
retriever = vectorstore.as_retriever(search_type="similarity", k=10)

prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요. Don't narrate the answer, just answer the question. Let's think step-by-step.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 체인을 생성합니다.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class StreamChain:
    def __init__(self, chain):
        self.chain = chain

    def stream(self, query):
        response = self.chain.stream(query)
        complete_response = ""
        for token in response:
            print(token, end="", flush=True)
            complete_response += token
        return complete_response

    def get_top_n(self, query, n=6):
        results = retriever.get_relevant_documents(query)
        return results[:n]

# 생성자에 chain 을 매개변수로 전달하여 chain 객체를 생성합니다.
chain = StreamChain(rag_chain)