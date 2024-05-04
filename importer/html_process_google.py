import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


# import error handling
import sys
here = os.path.dirname(__file__)

sys.path.append(os.path.join(here, '..'))

from importer.config import EMBEDDING_MODEL 

import nltk
nltk.download('punkt')

load_dotenv()
notion_path = "source_docs/notion_page_content.html"
loader = UnstructuredHTMLLoader(notion_path)
docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# html_header_splits = []
# for doc in docs:
#     html_header_split = html_splitter.split_text(doc.page_content)
#     html_header_splits.append(html_header_split)
html_header_splits = ""
for doc in docs:
    for txt in html_splitter.split_text(doc.page_content):
        html_header_splits += txt.page_content


text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
chunks = text_splitter.split_text(html_header_splits)
print(chunks[0])
print('\n================================\n')
print('\n================================\n')
print(chunks[2])
print('\n================================\n')
print(chunks[3])
db = FAISS.from_texts(chunks, embeddings)
db.save_local("faiss_index")
