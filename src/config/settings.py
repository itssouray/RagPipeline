from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"
TEMPERATURE = 0

# Data

PDF_PATH = "data/raw/transformer.pdf"


# Chunking

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# Embeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Retrieval

TOP_K = 3


# Vector Store

FAISS_INDEX_PATH = "vector_store/index.faiss"
FAISS_METADATA_PATH = "vector_store/documents.pkl"