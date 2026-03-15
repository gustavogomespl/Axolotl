import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings


class VectorStoreManager:
    """Manage ChromaDB collections for RAG.

    Each skill or document set maps to a separate collection.
    Supports cross-collection search.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.client = chromadb.HttpClient(
            host=host or settings.chroma_host,
            port=port or settings.chroma_port,
        )
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")

    def get_or_create_collection(self, name: str) -> Chroma:
        return Chroma(
            client=self.client,
            collection_name=name,
            embedding_function=self.embeddings,
        )

    def get_retriever(self, collection_name: str, k: int = 5):
        collection = self.get_or_create_collection(collection_name)
        return collection.as_retriever(search_kwargs={"k": k})

    def add_documents(self, collection_name: str, documents: list[Document]) -> list[str]:
        """Add documents to a collection and return their IDs."""
        collection = self.get_or_create_collection(collection_name)
        return collection.add_documents(documents)

    def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection."""
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass  # Collection may not exist

    def cross_collection_search(
        self,
        query: str,
        collections: list[str],
        k: int = 5,
    ) -> list[Document]:
        """Search across multiple collections and return merged results."""
        all_results: list[tuple[Document, float]] = []

        for coll_name in collections:
            try:
                collection = self.get_or_create_collection(coll_name)
                results = collection.similarity_search_with_score(query, k=k)
                all_results.extend(results)
            except Exception:
                continue

        # Sort by score (lower = more similar for distance metrics)
        all_results.sort(key=lambda x: x[1])
        return [doc for doc, _score in all_results[:k]]

    def list_collections(self) -> list[str]:
        """List all collection names."""
        collections = self.client.list_collections()
        return [c.name for c in collections]
