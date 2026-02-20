# rag/ingest.py
# Load documents into ChromaDB knowledge base

import os
import uuid
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever


class DocumentIngester:
    """
    Loads documents from files into ChromaDB.
    Run once during setup or when new docs are added.
    Supports PDF, HTML, TXT, MD files.
    """

    def __init__(self, retriever: RAGRetriever, embedder: BGEEmbedder):
        self.retriever = retriever
        self.embedder = embedder

    def ingest_file(self, filepath: str, topic_area: str, source: str):
        """Ingest a single file into ChromaDB knowledge base."""
        text = self._extract_text(filepath)
        chunks = self._chunk_text(text)

        if not chunks:
            print(f"No chunks extracted from {filepath}")
            return

        embeddings = self.embedder.embed_documents(chunks)
        metadatas = [
            {
                "source": source,
                "topic_area": topic_area,
                "filepath": filepath,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]

        self.retriever.knowledge_collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Ingested {len(chunks)} chunks from {source}")

    def ingest_directory(self, directory: str, topic_area: str):
        """Ingest all supported files in a directory."""
        supported = [".pdf", ".html", ".htm", ".txt", ".md"]
        for filepath in Path(directory).rglob("*"):
            if filepath.suffix.lower() in supported:
                self.ingest_file(
                    str(filepath),
                    topic_area=topic_area,
                    source=filepath.name
                )

    def _extract_text(self, filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(filepath)
            return " ".join(
                page.extract_text() for page in reader.pages
                if page.extract_text()
            )
        elif ext in [".html", ".htm"]:
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator=" ", strip=True)
        elif ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print(f"Unsupported file type: {ext}")
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + RAG_CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
            i += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
        return chunks
