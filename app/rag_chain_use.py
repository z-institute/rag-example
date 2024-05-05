import os
from operator import itemgetter
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


load_dotenv()
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=os.environ.get("HF_TOKEN"), model_name="sentence-transformers/all-MiniLM-l6-v2"
# )
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = os.environ.get("OPENAI_API_KEY"))

# kitty  = (1,10,0,1,1,3)
# cat = (1,10,0,1,2,3)
# dog = (1,10,0,0,0,2)

# vector_store = FAISS.load_local("faiss_index", embeddings)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

# template = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
llm = ChatOpenAI()

class RagInput(TypedDict):
    question: str

ANSWER_PROMPT = ChatPromptTemplate.from_template(prompt_template)

# print(ANSWER_PROMPT)

final_chain = (
        {
            "context": itemgetter("question") | vector_store.as_retriever(),
            "question": itemgetter("question")
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
).with_types(input_type=RagInput)

ticket_name = "甲子"
ai_response = final_chain.invoke({"question": 
"""請回答""" + ticket_name + """籤的內容，格式為：
籤詩：
白話內容：
凡事
作事
家事
家運
婚姻
求兒
六甲
求財
功名
歲君
治病
出外
經商
來人
行舟
移居
失物
求雨
官事
六畜
耕作
築室
墳墓
討海
作塭
魚苗
月令
尋人
遠信"""
, 
"input_documents": vector_store.similarity_search(ticket_name)}
)

print(ai_response)

