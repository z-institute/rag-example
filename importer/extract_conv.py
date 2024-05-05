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

html_header_splits = ""
for doc in docs:
    for txt in doc.page_content:
        html_header_splits += txt

lines = html_header_splits.split("。")
for line in lines:
    print(line)
    print('\n================================\n')
# print(chunks[1])
# char_name = "王間"
# for txt in chunks:
#     if char_name in txt:
#         print(txt)


