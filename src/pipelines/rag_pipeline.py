from langchain_community.document_loaders import bigquery
from src.loader.pdf_loader import PDFLoader
from src.chunkers.recursive_chunker import RecursiveChunker
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstores.faiss_store import FAISSStore
from src.retrievers.rag_retriever import RAGRetriever
from src.generators.response_generator import ResponseGenerator
from src.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    OPENAI_MODEL,
    TEMPERATURE
)




class RAGPipeline:

    def __init__(self):

        self.loader = PDFLoader()

        self.chunker = RecursiveChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        self.embedder = EmbeddingManager(model_name=EMBEDDING_MODEL)

        self.vector_store = None

        self.retriever = None

        self.generator = ResponseGenerator(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE
        )

    def ingest(
        self,
        pdf_path: str
    ):

        documents = self.loader.load(
            pdf_path
        )

        chunks = self.chunker.split(
            documents
        )

        embeddings = self.embedder.embed_documents(
            chunks
        )

        self.vector_store = FAISSStore(
            dimension=embeddings.shape[1]
        )

        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        self.retriever = RAGRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store
        )

        print(
            f"Ingestion complete. "
            f"{len(chunks)} chunks stored."
        )


    def search(self,query: str,k: int = TOP_K):

        return self.retriever.retrieve(
            query=query,
            k=k
        )


    def save_index(self):
        self.vector_store.save(index_path=FAISS_INDEX_PATH,metadata_path=FAISS_METADATA_PATH)
        print("Index saved successfully")

    
    def load_index(self):

        self.vector_store = FAISSStore(dimension=384)
        self.vector_store.load(index_path=FAISS_INDEX_PATH,metadata_path=FAISS_METADATA_PATH)
        self.retriever = RAGRetriever(embedder=self.embedder,vector_store=self.vector_store)

        print(f"Loaded {len(self.vector_store.documents)} documents")

    
    
    def ask(self,query: str) -> str:

        documents = self.search(query=query)

        return self.generator.generate(
            query=query,
            documents=documents
        )