from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        metadata = doc.metadata.copy()
        # Ensure doc_id is in metadata for deletion
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
            
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored_results = []
        
        for rec in records:
            score = _dot(query_embedding, rec["embedding"])
            scored_results.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score
            })
            
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.
        """
        if self._use_chroma:
            ids = []
            contents = []
            metadatas = []
            embeddings = []
            for doc in docs:
                chunk_id = f"{doc.id}_{self._next_index}"
                self._next_index += 1
                ids.append(chunk_id)
                contents.append(doc.content)
                
                meta = doc.metadata.copy()
                meta["doc_id"] = doc.id
                metadatas.append(meta)
                
                embeddings.append(self._embedding_fn(doc.content))
                
            self._collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k
            )
            # Reformat to match common output
            formatted = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            for i in range(len(ids)):
                formatted.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    # Invert distance for a simple "score" if needed
                    "score": 1.0 - distances[i] if i < len(distances) else 0.0
                })
            return formatted
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.
        """
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter
            )
            # Format same as search
            formatted = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            for i in range(len(ids)):
                formatted.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "score": 1.0 - distances[i] if i < len(distances) else 0.0
                })
            return formatted
        else:
            filtered_records = []
            for rec in self._store:
                match = True
                for k, v in metadata_filter.items():
                    if rec["metadata"].get(k) != v:
                        match = False
                        break
                if match:
                    filtered_records.append(rec)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        """
        if self._use_chroma:
            count_before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < count_before
        else:
            initial_count = len(self._store)
            self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_count
