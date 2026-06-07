from langchain_openai import ChatOpenAI
from src.models.document import Document


class ResponseGenerator:

    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0
    ):

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature
        )

    def generate(
        self,
        query: str,
        documents: list[Document]
    ) -> str:

        context = "\n\n".join(
            doc.content
            for doc in documents
        )

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

Context:
{context}

Question:
{query}
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content