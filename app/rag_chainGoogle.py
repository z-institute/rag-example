import os
from operator import itemgetter
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain


PG_COLLECTION_NAME = "pdf_rag"

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = FAISS.load_local("../all_faiss/ai-gf/faiss_index", embeddings, allow_dangerous_deserialization=True)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
        model="gemini-pro", convert_system_message_to_human=True, streaming=True
    )


class RagInput(TypedDict):
    question: str


final_chain = load_qa_chain(llm, chain_type="stuff", prompt=ANSWER_PROMPT)
