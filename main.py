from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# Automatically find all .txt and .md files in the data directory
data_path = Path("data/group_data")
if data_path.exists():
    SAMPLE_FILES = [str(p) for p in data_path.glob("*") if p.suffix.lower() in {".txt", ".md"}]
    SAMPLE_FILES.sort()
else:
    SAMPLE_FILES = []


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            continue

        if not path.exists() or not path.is_file():
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={
                    "source": str(path), 
                    "category": path.stem.split("_")[1].capitalize() if "_" in path.stem else "General"
                },
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


from openai import OpenAI

# Load system prompt from file
SYSTEM_PROMPT_PATH = Path("system_prompt.txt")
SYSTEM_PROMT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if SYSTEM_PROMPT_PATH.exists() else "You are a helpful assistant."

def openai_llm(prompt: str) -> str:
    client = OpenAI() # Automatically reads OPENAI_API_KEY from .env
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMT},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    load_dotenv(override=False)

    print("=== Knowledge Base Loading ===")
    raw_docs = load_documents_from_files(files)
    if not raw_docs:
        print("\nNo valid input files were found in 'data/' folder.")
        return 1

    print(f"Loaded {len(raw_docs)} documents.")

    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    agent = KnowledgeBaseAgent(store=store, llm_fn=openai_llm)

    from src.chunking import RecursiveChunker
    chunker = RecursiveChunker(chunk_size=500)
    
    print("Chunking and indexing documents...")
    agent.ingest_docs(raw_docs, chunker=chunker)
    print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore.")

    # Initial query if provided via command line
    if question:
        print(f"\nSearching for: {question}")
        print(f"Answer:\n{agent.answer(question, top_k=3)}")
        return 0

    # Interactive loop
    print("\n" + "="*40)
    print("WELCOME TO PERSONAL RAG CHAT")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*40)

    while True:
        try:
            query = input("\n[You]: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not query:
                continue
            
            print("[Agent]: Typing...", end="\r")
            answer = agent.answer(query, top_k=3)
            print(f"[Agent]: {answer}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())

