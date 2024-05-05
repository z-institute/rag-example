from flask import Flask, request, jsonify
import os
from operator import itemgetter
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

llm = ChatOpenAI()

class RagInput(TypedDict):
    question: str

ANSWER_PROMPT = ChatPromptTemplate.from_template(prompt_template)

final_chain = (
        {
            "context": itemgetter("question") | vector_store.as_retriever(),
            "question": itemgetter("question")
        }
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
).with_types(input_type=RagInput)

@app.route('/get_response', methods=['POST'])
def get_response():
    ticket_name = request.json.get('ticket_name')
    if not ticket_name:
        return jsonify({"error": "Ticket name is required"}), 400
    
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
    遠信""",
    "input_documents": vector_store.similarity_search(ticket_name)})
    
    return ai_response

if __name__ == '__main__':
    app.run(debug=True, port=9901)
