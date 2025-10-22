import os
from typing import List

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
import getpass


# load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

# Initialize embeddings
embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize LangChain Chroma vector store with Chroma Cloud credentials
vector_store = Chroma(
    collection_name=os.getenv("CHROMA_COLLECTION"),
    embedding_function=embedding_function,
    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)


def add_documents(documents: List[Document]) -> None:

    if not documents:
        print("Warning: No documents provided to add.")
        return

    try:
        vector_store.add_documents(documents=documents)
        print(f"Successfully added {len(documents)} documents.")
    except Exception as e:
        print(f"Error adding documents: {e}")


def get_retriever(k_results: int = 5) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_kwargs={"k": k_results})


def delete_documents(source_file: str) -> None:
    """
    Removes all documents associated with a specific file from the vector store.
    
    Args:
        source_file: The name of the file to unindex.
    """
    try:
        # Delete documents where the 'source' metadata field matches the specified filename
        vector_store.delete(where={"source": source_file})
        print(f"Successfully removed all documents from {source_file}.")
    except Exception as e:
        print(f"Error removing documents: {e}")
