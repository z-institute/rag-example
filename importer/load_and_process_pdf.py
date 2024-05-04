import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

import nltk

load_dotenv()

nltk.download('punkt')

loader = DirectoryLoader(
    os.path.relpath("source_docs"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
)
docs = loader.load()

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ.get("HF_TOKEN"), model_name="sentence-transformers/all-MiniLM-l6-v2"
)

html_header_splits = ""
for doc in docs:
    for txt in doc.page_content:
        html_header_splits += txt

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
chunks = text_splitter.split_text(html_header_splits)

# text_splitter = SemanticChunker(
#     embeddings, breakpoint_threshold_type="percentile"
# )

# text_splitter = SemanticChunker(
#     embeddings
# )
# print(text_splitter)
# chunks = text_splitter.split_documents(docs)
# print(len(chunks))

db = FAISS.from_texts(chunks, embeddings)
db.save_local("faiss_index")
