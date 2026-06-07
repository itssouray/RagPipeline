from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models.document import Document


class RecursiveChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(
        self,
        documents: list[Document]
    ) -> list[Document]:

        chunks = []

        for document in documents:

            split_texts = self.text_splitter.split_text(
                document.content
            )


            for idx, chunk in enumerate(split_texts):

                metadata = document.metadata.copy()

                metadata["chunk_id"] = idx

                chunks.append(
                    Document(
                    content=chunk,
                    metadata=metadata
                    )
                )

        return chunks