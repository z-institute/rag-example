import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import nltk

load_dotenv()

nltk.download('punkt')

loader = DirectoryLoader(
    os.path.relpath("source_docs"),
    glob="**/*.docx",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredWordDocumentLoader,
)
docs = loader.load()

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', )


db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")
