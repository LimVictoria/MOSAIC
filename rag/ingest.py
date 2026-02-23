# rag/ingest.py
# Incremental document ingestion into Pinecone
# Only processes NEW files not already in Pinecone

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
    Incremental ingestion into Pinecone.

    On every run:
    - Checks which source files are already in Pinecone
    - Only processes NEW files not yet ingested
    - Never re-embeds documents already uploaded
    - Tracks ingestion via source filename in metadata

    Run anytime — safe to run on app startup since
    it skips already-ingested files instantly.
    """

    def __init__(self, retriever: RAGRetriever, embedder: BGEEmbedder):
        self.retriever = retriever
        self.embedder  = embedder

    def ingest_file(
        self,
        filepath: str,
        topic_area: str,
        source: str,
        namespace: str = "knowledge_base",
        force: bool = False
    ):
        """
        Ingest a single file into Pinecone.

        Args:
            filepath:   path to the document
            topic_area: e.g. "python_data_science", "statistics"
            source:     unique filename used for deduplication
            namespace:  "knowledge_base" or "assessment_bank"
            force:      if True, re-ingest even if already in Pinecone
        """
        # Check if already ingested
        if not force:
            already_ingested = self.retriever.get_ingested_sources(namespace)
            if source in already_ingested:
                print(f"Skipping (already ingested): {source}")
                return

        print(f"Ingesting: {source}")
        text   = self._extract_text(filepath)
        chunks = self._chunk_text(text)

        if not chunks:
            print(f"No chunks extracted from {filepath}")
            return

        # Embed in batches to avoid memory issues
        batch_size = 100
        total_uploaded = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings   = self.embedder.embed_documents(batch_chunks)

            vectors = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                vectors.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "text":        chunk,
                        "source":      source,
                        "topic_area":  topic_area,
                        "filepath":    filepath,
                        "chunk_index": i + j
                    }
                })

            self.retriever.index.upsert(vectors=vectors, namespace=namespace)
            total_uploaded += len(vectors)
            print(f"  Uploaded {total_uploaded}/{len(chunks)} chunks...")

        print(f"Done: {source} — {len(chunks)} chunks in namespace '{namespace}'")
    def ingest_text(
        self,
        text: str,
        topic_area: str,
        source: str,
        namespace: str = "knowledge_base",
        force: bool = False
    ):
        """
        Ingest pre-extracted text directly into Pinecone.
        Used for file types like .ipynb where text extraction
        happens outside this class.
        """
        if not force:
            already_ingested = self.retriever.get_ingested_sources(namespace)
            if source in already_ingested:
                print(f"Skipping (already ingested): {source}")
                return

        print(f"Ingesting: {source}")
        chunks = self._chunk_text(text)

        if not chunks:
            print(f"No chunks extracted from {source}")
            return

        batch_size     = 100
        total_uploaded = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings   = self.embedder.embed_documents(batch_chunks)

            vectors = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                vectors.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "text":        chunk,
                        "source":      source,
                        "topic_area":  topic_area,
                        "chunk_index": i + j
                    }
                })

            self.retriever.index.upsert(vectors=vectors, namespace=namespace)
            total_uploaded += len(vectors)
            print(f"  Uploaded {total_uploaded}/{len(chunks)} chunks...")

        print(f"Done: {source} — {len(chunks)} chunks in namespace '{namespace}'")
    def ingest_directory(
        self,
        directory: str,
        topic_area: str,
        namespace: str = "knowledge_base",
        force: bool = False
    ):
        """
        Incrementally ingest all supported files in a directory.
        Only processes files not already in Pinecone.
        """
        supported = [".pdf", ".html", ".htm", ".txt", ".md"]
        files     = list(Path(directory).rglob("*"))
        files     = [f for f in files if f.suffix.lower() in supported]

        if not files:
            print(f"No supported files found in {directory}")
            return

        already_ingested = self.retriever.get_ingested_sources(namespace)
        new_files        = [f for f in files if f.name not in already_ingested]

        print(f"Found {len(files)} files, {len(new_files)} new to ingest")

        for filepath in new_files:
            self.ingest_file(
                str(filepath),
                topic_area=topic_area,
                source=filepath.name,
                namespace=namespace,
                force=force
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
        words  = text.split()
        chunks = []
        i      = 0
        while i < len(words):
            chunk = " ".join(words[i:i + RAG_CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
            i += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
        return chunks
