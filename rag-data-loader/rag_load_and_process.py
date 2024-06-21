import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, PDFPlumberLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
connection_string = os.getenv('POSTGRES_CONNECTION_STRING')
loader = DirectoryLoader(
    os.path.abspath("../pdf-documents"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=PDFPlumberLoader, #UnstructuredPDFLoader,
)
docs = loader.load()

embeddings = OpenAIEmbeddings()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

PGVector.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="db64", 
    connection_string=connection_string,
    #"postgresql+psycopg://postgres@localhost:5432/database164",
    pre_delete_collection=True,#delete all above
)