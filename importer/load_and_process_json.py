import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import nltk

load_dotenv()

nltk.download('punkt')

file_path='source_docs/token.json'
# data = json.loads(Path(file_path).read_text())

loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',
    text_content=False)

data = loader.load()

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

db = FAISS.from_documents(data, embeddings)
db.save_local("faiss_index")
