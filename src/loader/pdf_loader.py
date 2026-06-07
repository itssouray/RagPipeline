from langchain_community.document_loaders import PyMuPDFLoader
from  src.models.document import Document


class PDFLoader:

    def load(self, file_path: str) -> list[Document]:

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        return [
            Document(
                content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in documents
        ]