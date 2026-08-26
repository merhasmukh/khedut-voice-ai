import asyncio
from rag.ingestion import ingest_knowledge_base_directory

async def main():
    print("Starting ingestion...")
    results = await ingest_knowledge_base_directory("knowledge_base")
    print("Ingestion results:", results)

if __name__ == "__main__":
    asyncio.run(main())
