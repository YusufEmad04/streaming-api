from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings
import os
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
from enum import Enum
# from langchain.agents.react.base import DocstoreExplorer
# from langchain.agents import initialize_agent, Tool
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore


# yusuf.emad.pinecone email
# pinecone.init(api_key="267b5ad1-21ec-46eb-be4f-1c9100136e2d", environment="us-east-1")
# os.environ["PINECONE_API_KEY"] =
def syllabus_vectorstore():
    return Pinecone.from_existing_index(index_name="agent", embedding=OpenAIEmbeddings())

def syllabus_vectorstore_parent():
    vectorstore = Pinecone.from_existing_index(index_name="creativity",
                                               embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
    store = LocalFileStore("split_parents")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=660, chunk_overlap=70, length_function=lambda x: len(
        tiktoken.encoding_for_model("gpt-3.5-turbo").encode(x)))
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=190, chunk_overlap=0, length_function=lambda x: len(
        tiktoken.encoding_for_model("gpt-3.5-turbo").encode(x)))

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever

syllabus_vectorstore_parent()