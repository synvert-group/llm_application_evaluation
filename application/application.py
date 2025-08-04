import os
import time
from typing import List, Tuple

import openai

from application.docling_parser import DocParser
from application.reranker import Reranker
from application.vector_store import VectorStore


class Application:

    def __init__(self, reset_collection: bool = False):

        # Init key components
        self.llm_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = "gpt-4o-mini"
        self.temperature = 0.0

        # LLM Application components
        self.vector_store = VectorStore(collection_name="llm_vector_store", reset_collection=reset_collection)
        self.reranker = Reranker()
        self.doc_parser = DocParser()

        self.system_message = """You are a helpful assistant analyzing annual reports of companies and answering questions about them.
        Answer the following questions as short and correct as possible.
        If you cant answer or missing any information, return "no answer".
        """

        if reset_collection:
            print("Resetting the vector store.")
            chunked_outputs = self.doc_parser.parse()
            print(f"Planning to add {len(chunked_outputs)} documents to the vector store.")
            self.vector_store.add_documents(chunked_outputs)
            print("Finished adding documents to the vector store.")

    def ask_llm(self, query: str) -> Tuple[str, List[str], float]:
        """
        Ask the LLM a question.

        Args:
            query: Question to ask the LLM

        Returns:
            Answer to the question, ["empty"], elapsed_time
        """
        start_time = time.time()
        chat_completion = self.llm_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_message,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            model=self.model,
            temperature=self.temperature,
        )
        elapsed_time = time.time() - start_time
        return chat_completion.choices[0].message.content, ["empty"], elapsed_time

    def ask_llm_with_rag(self, query: str) -> Tuple[str, List[str], float]:
        """
        Ask the LLM a question using RAG.

        Args:
            query: Question to ask the LLM

        Returns:
            Answer to the question, docs, elapsed_time
        """
        start_time = time.time()
        documents = self.vector_store.search(query, n_results=5)
        document_strings = [
            f"{i+1}. Text: {doc['text']}\n Distance: {doc['distance']}" for i, doc in enumerate(documents)
        ]
        documents_str = "\n\n".join(document_strings)
        chat_completion = self.llm_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_message
                    + f"\n You will get some contexts that you can use to answer the question. A lower distance means a higher relevance for the answer.",
                },
                {
                    "role": "user",
                    "content": query + f" \n Context: { '\n'.join(documents_str) }",
                },
            ],
            model=self.model,
            temperature=self.temperature,
        )
        elapsed_time = time.time() - start_time
        return chat_completion.choices[0].message.content, [doc["text"] for doc in documents], elapsed_time

    def ask_llm_with_rag_and_rerank(self, query: str) -> Tuple[str, List[str], float]:
        """
        Ask the LLM a question using RAG and retrieval.

        Args:
            query: Question to ask the LLM

        Returns:
            Answer to the question, reranked_docs_raw, elapsed_time
        """
        start_time = time.time()
        documents = self.vector_store.search(query, n_results=50)
        reranked_docs_parsed, reranked_docs_raw = self.reranker.rerank(query, documents, top_k=5)
        chat_completion = self.llm_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_message
                    + f"\n You will get some texts that you can use to answer the question. A higher score means a higher relevance for the answer. A lower distance means a higher relevance for the answer.",
                },
                {
                    "role": "user",
                    "content": query + f" \n Context: { '\n'.join(reranked_docs_parsed) }",
                },
            ],
            model=self.model,
            temperature=self.temperature,
        )
        elapsed_time = time.time() - start_time
        return chat_completion.choices[0].message.content, reranked_docs_raw, elapsed_time
