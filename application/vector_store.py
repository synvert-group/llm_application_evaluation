import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm


class VectorStore:
    def __init__(self, collection_name: str, reset_collection: bool = False):
        """
        Initialize the vector store with OpenAI embeddings.

        Args:
            collection_name: Name of the collection to create/use
            openai_api_key: OpenAI API key (optional if set in environment)
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db", is_persistent=True))

        # Delete existing collection if reset_collection is True
        if reset_collection:
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except ValueError:
                # Collection doesn't exist, ignore the error
                pass

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text using OpenAI."""
        response = self.openai_client.embeddings.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding

    def add_documents(self, texts: List[str]) -> List[str]:
        """
        Add multiple documents to the vector store.

        Args:
            texts: List of document texts

        Returns:
            List of generated document IDs
        """
        ids = [str(i) for i in range(len(texts))]
        embeddings = []

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_text = {executor.submit(self._generate_embedding, text): text for text in texts}

            # Use tqdm to show progress
            with tqdm(total=len(texts), desc="Generating Embeddings", unit="text") as pbar:
                for future in as_completed(future_to_text):
                    embeddings.append(future.result())
                    pbar.update(1)

        self.collection.add(embeddings=embeddings, documents=texts, ids=ids)

    def search(self, query: str, n_results: int = 25) -> dict:
        """
        Search for similar documents.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results
        """
        query_embedding = self._generate_embedding(query)

        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)

        result_list = [
            {"text": doc, "distance": dist} for doc, dist in zip(results["documents"][0], results["distances"][0])
        ]

        result_list.sort(key=lambda x: x["distance"])
        return result_list

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.chroma_client.delete_collection(self.collection.name)
