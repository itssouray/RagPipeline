from sentence_transformers import SentenceTransformer
from src.models.document import Document
import numpy as np

class EmbeddingManager:

    def __init__(self,model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name
        )

    def embed_documents(self,documents: list[Document]) -> np.ndarray:

        texts = [
            doc.content
            for doc in documents
        ]

        return self.model.encode(
            texts,
            convert_to_numpy=True
        )


    def embed_query(self,query: str) -> np.ndarray:

        return self.model.encode(
            query,
            convert_to_numpy=True
        )