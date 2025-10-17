import os
from typing import List

from dotenv import load_dotenv
import chromadb

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


load_dotenv()

# Initialize embeddings
embedding_function = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

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


def delete_documents(filename: str) -> None:
    if not filename:
        print("Error: filename must be provided.")
        return

    try:
        # To delete, we need the collection object. We can get it from the client.
        collection = vector_store.get_collection(name=os.getenv("CHROMA_COLLECTION"))

        # Find the IDs of documents to delete.
        ids_to_delete = collection.get(where={"source": filename})["ids"]

        if not ids_to_delete:
            print(f"No documents found for filename '{filename}'. Nothing to delete.")
            return

        # Use the collected IDs to perform the deletion.
        collection.delete(ids=ids_to_delete)
        print(
            f"Successfully deleted {len(ids_to_delete)} documents for filename: '{filename}'."
        )
    except Exception as e:
        print(
            f"An error occurred while deleting documents for filename '{filename}': {e}"
        )


def get_retriever(k_results: int = 5) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_kwargs={"k": k_results})
