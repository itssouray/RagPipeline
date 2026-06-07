import faiss
import pickle
import numpy as np

from src.models.document import Document


class FAISSStore:

    def __init__(self, dimension: int):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(dimension)

        self.documents: list[Document] = []

    def add_documents(self,documents: list[Document],embeddings: np.ndarray):

        self.index.add(
            embeddings.astype("float32")
        )

        self.documents.extend(documents)


    def similarity_search(self,query_embedding: np.ndarray,k: int = 5) -> list[Document]:

        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype("float32"),k)

        results = []

        for idx in indices[0]:

            if idx != -1:
                results.append(
                    self.documents[idx]
                )

        return results

    def save(self,index_path: str,metadata_path: str):

        faiss.write_index(self.index,index_path
        )

        with open(metadata_path, "wb") as f:
            pickle.dump(
                self.documents,
                f
            )

    def load(
        self,
        index_path: str,
        metadata_path: str
    ):

        self.index = faiss.read_index(
            index_path
        )

        with open(metadata_path, "rb") as f:
            self.documents = pickle.load(f)