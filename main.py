from src.pipelines.rag_pipeline import RAGPipeline
from src.config.settings import PDF_PATH



def main():

    pipeline = RAGPipeline()
    # pipeline.ingest(PDF_PATH)
    pipeline.load_index()


    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break
        answer = pipeline.ask(query)
        print(f"\nAssistant: {answer}")

    


if __name__ == "__main__":
    main()