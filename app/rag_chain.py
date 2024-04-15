import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key = os.environ.get("OPENAI_API_KEY"))

# kitty  = (1,10,0,1,1,3)
# cat = (1,10,0,1,2,3)
# dog = (1,10,0,0,0,2)

# vector_store = FAISS.load_local("faiss_index", embeddings)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
llm = ChatOpenAI()

final_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
