from typing import Callable

from .store import EmbeddingStore
from .models import Document
from typing import Any



class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def ingest_docs(self, documents: list[Document], chunker: Any = None) -> None:
        """
        RAG Ingestion Pipeline:
        1. Chunk documents (if chunker is provided).
        2. Add documents to the embedding store.
        """
        if chunker is None:
            self.store.add_documents(documents)
            return
            
        chunked_docs = []
        for doc in documents:
            chunks = chunker.chunk(doc.content)
            for i, chunk_text in enumerate(chunks):
                meta = doc.metadata.copy()
                # Track original document ID if needed
                meta["parent_id"] = doc.id 
                chunked_docs.append(
                    Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=chunk_text,
                        metadata=meta,
                    )
                )
        self.store.add_documents(chunked_docs)

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        RAG workflow:
        1. Retrieve top-k relevant chunks.
        2. Build a context-rich prompt.
        3. Call the LLM (llm_fn).
        """
        results = self.store.search(question, top_k=top_k)
        
        context_parts = []
        for i, res in enumerate(results):
            context_parts.append(f"Chunk {i+1}:\n{res['content']}")
            
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant. Answer the user question concisely using ONLY the provided context.
If the answer is not in the context, say "I don't know based on the provided documents."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""
        
        return self.llm_fn(prompt)
