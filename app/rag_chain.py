import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, answer i don't know if the answer is not in the context\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

class RagInput(TypedDict):
    question: str

llm = ChatOpenAI()
# llm = ChatGoogleGenerativeAI(
#         model="gemini-pro", convert_system_message_to_human=True, streaming=True
#     )

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

# lang chain
# output = module1 | module2 | module3 | module4 
