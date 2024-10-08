import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import nltk

load_dotenv()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

loader = DirectoryLoader(
    os.path.relpath("source_docs"),
    glob="**/*.md",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredMarkdownLoader,
)
docs = loader.load()


# dog cat car
# dog: (0, 1, 2)
# cat: (0, 1, 3)
# car: (2, 10, 100)

# input: taiwan -> (111, 1000, 100)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', )

headers_to_split_on = [
    ("#", "Header 1")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = markdown_splitter.split_text(docs[0].page_content)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
