from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstores.faiss_store import FAISSStore
from src.models.document import Document


class RAGRetriever:

    def __init__(
        self,
        embedder: EmbeddingManager,
        vector_store: FAISSStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> list[Document]:

        query_embedding = self.embedder.embed_query(
            query
        )

        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k
        )