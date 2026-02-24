# rag/ingest.py
# Incremental document ingestion into Pinecone
# Section-aware chunking — detects headers and tags chunks with correct topic_area
# No need to split physical PDFs — one file can cover multiple topics

import os
import re
import uuid
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever

# ── Section header → curriculum Topic mapping ─────────────────────────────────
# Keywords detected in section headers map to exact curriculum Topic names
# Add more keywords as you add new documents

SECTION_TOPIC_MAPPING = {
    # Reading Structured Files
    "reading structured":   "Reading Structured Files",
    "read file":            "Reading Structured Files",
    "read csv":             "Reading Structured Files",
    "read excel":           "Reading Structured Files",
    "read json":            "Reading Structured Files",
    "loading data":         "Reading Structured Files",
    "load data":            "Reading Structured Files",
    "import data":          "Reading Structured Files",
    "structured file":      "Reading Structured Files",
    "file format":          "Reading Structured Files",
    "csv":                  "Reading Structured Files",
    "excel":                "Reading Structured Files",
    "parquet":              "Reading Structured Files",

    # Structured Data Types
    "data type":            "Structured Data Types",
    "dataframe":            "Structured Data Types",
    "data structure":       "Structured Data Types",
    "vector":               "Structured Data Types",
    "array":                "Structured Data Types",
    "tensor":               "Structured Data Types",
    "series":               "Structured Data Types",
    "sql":                  "Structured Data Types",

    # Exploratory Data Analysis
    "exploratory":          "Exploratory Data Analysis",
    "eda":                  "Exploratory Data Analysis",
    "data analysis":        "Exploratory Data Analysis",
    "descriptive statistic":"Exploratory Data Analysis",
    "correlation":          "Exploratory Data Analysis",
    "outlier":              "Exploratory Data Analysis",
    "missing value":        "Exploratory Data Analysis",
    "data quality":         "Exploratory Data Analysis",
    "distribution":         "Exploratory Data Analysis",
    "stationarity":         "Exploratory Data Analysis",
    "class imbalance":      "Exploratory Data Analysis",

    # Data Visualization
    "visualization":        "Data Visualization",
    "visualisation":        "Data Visualization",
    "plotting":             "Data Visualization",
    "bar chart":            "Data Visualization",
    "box plot":             "Data Visualization",
    "heatmap":              "Data Visualization",
    "matplotlib":           "Data Visualization",
    "seaborn":              "Data Visualization",
    "pareto":               "Data Visualization",

    # Imputation Techniques
    "imputation":           "Imputation Techniques",
    "impute":               "Imputation Techniques",
    "missing data":         "Imputation Techniques",
    "forward fill":         "Imputation Techniques",
    "backward fill":        "Imputation Techniques",
    "interpolation":        "Imputation Techniques",
    "kalman":               "Imputation Techniques",
    "median fill":          "Imputation Techniques",
    "mean fill":            "Imputation Techniques",

    # Data Augmentation
    "augmentation":         "Data Augmentation",
    "smote":                "Data Augmentation",
    "oversampling":         "Data Augmentation",
    "undersampling":        "Data Augmentation",
    "adasyn":               "Data Augmentation",
    "imbalanced":           "Data Augmentation",

    # Feature Reduction
    "feature reduction":    "Feature Reduction",
    "dimensionality":       "Feature Reduction",
    "pca":                  "Feature Reduction",
    "principal component":  "Feature Reduction",
    "feature selection":    "Feature Reduction",

    # Business Metrics
    "business metric":      "Business Metrics",
    "kpi":                  "Business Metrics",
    "churn":                "Business Metrics",
    "run rate":             "Business Metrics",
    "forecasting":          "Business Metrics",
    "year over year":       "Business Metrics",
    "turnover":             "Business Metrics",

    # Preprocessing Summary
    "preprocessing":        "Preprocessing Summary",
    "pre-processing":       "Preprocessing Summary",
    "data pipeline":        "Preprocessing Summary",
    "data preparation":     "Preprocessing Summary",
    "feature engineering":  "Preprocessing Summary",

    # ML Frameworks
    "machine learning":     "ML Frameworks",
    "pytorch":              "ML Frameworks",
    "tensorflow":           "ML Frameworks",
    "keras":                "ML Frameworks",
    "scikit":               "ML Frameworks",
    "sklearn":              "ML Frameworks",
    "ml framework":         "ML Frameworks",
    "deep learning":        "ML Frameworks",
    "neural network":       "ML Frameworks",

    # Python for Data Science
    "python":               "Python for Data Science",
    "pandas":               "Python for Data Science",
    "numpy":                "Python for Data Science",
    "data science":         "Python for Data Science",
}

# Header patterns to detect section boundaries in PDFs
HEADER_PATTERNS = [
    r"^(UNIT\s+[IVX0-9]+[\.\:\s])",          # UNIT I, UNIT II, UNIT 1
    r"^(Chapter\s+\d+[\.\:\s])",              # Chapter 1
    r"^(\d+\.\d*\s+[A-Z][a-zA-Z\s]+)",       # 1.1 Topic Name
    r"^([A-Z][A-Z\s]{4,}$)",                  # ALL CAPS HEADER
    r"^([A-Z][a-zA-Z\s]+:$)",                 # Title Case Header:
]


class DocumentIngester:
    """
    Section-aware incremental ingestion into Pinecone.

    Instead of blind word-count chunking, this detects section headers
    and tags each chunk with the correct curriculum topic_area.

    One PDF can cover multiple topics — each section gets tagged correctly.
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
        Ingest a single file into Pinecone with section-aware topic tagging.
        topic_area is used as fallback if no section header matches.
        """
        if not force:
            already_ingested = self.retriever.get_ingested_sources(namespace)
            if source in already_ingested:
                print(f"Skipping (already ingested): {source}")
                return

        print(f"Ingesting: {source}")
        text = self._extract_text(filepath)

        # Use section-aware chunking
        chunks_with_topics = self._chunk_by_section(text, fallback_topic=topic_area)

        if not chunks_with_topics:
            print(f"No chunks extracted from {filepath}")
            return

        batch_size     = 100
        total_uploaded = 0

        for i in range(0, len(chunks_with_topics), batch_size):
            batch = chunks_with_topics[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.embed_documents(texts)

            vectors = []
            for j, (chunk_data, embedding) in enumerate(zip(batch, embeddings)):
                vectors.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "text":        chunk_data["text"],
                        "source":      source,
                        "topic_area":  chunk_data["topic_area"],
                        "section":     chunk_data["section"],
                        "filepath":    filepath,
                        "chunk_index": i + j
                    }
                })

            self.retriever.index.upsert(vectors=vectors, namespace=namespace)
            total_uploaded += len(vectors)
            print(f"  Uploaded {total_uploaded}/{len(chunks_with_topics)} chunks...")

        # Show topic distribution
        from collections import Counter
        topic_counts = Counter(c["topic_area"] for c in chunks_with_topics)
        print(f"Done: {source} — {len(chunks_with_topics)} chunks")
        for topic, count in topic_counts.most_common():
            print(f"  {topic}: {count} chunks")

    def ingest_text(
        self,
        text: str,
        topic_area: str,
        source: str,
        namespace: str = "knowledge_base",
        force: bool = False
    ):
        """Ingest pre-extracted text (e.g. from .ipynb) with section-aware chunking."""
        if not force:
            already_ingested = self.retriever.get_ingested_sources(namespace)
            if source in already_ingested:
                print(f"Skipping (already ingested): {source}")
                return

        print(f"Ingesting: {source}")
        chunks_with_topics = self._chunk_by_section(text, fallback_topic=topic_area)

        if not chunks_with_topics:
            print(f"No chunks extracted from {source}")
            return

        batch_size     = 100
        total_uploaded = 0

        for i in range(0, len(chunks_with_topics), batch_size):
            batch      = chunks_with_topics[i:i + batch_size]
            texts      = [c["text"] for c in batch]
            embeddings = self.embedder.embed_documents(texts)

            vectors = []
            for j, (chunk_data, embedding) in enumerate(zip(batch, embeddings)):
                vectors.append({
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": {
                        "text":        chunk_data["text"],
                        "source":      source,
                        "topic_area":  chunk_data["topic_area"],
                        "section":     chunk_data["section"],
                        "chunk_index": i + j
                    }
                })

            self.retriever.index.upsert(vectors=vectors, namespace=namespace)
            total_uploaded += len(vectors)

        print(f"Done: {source} — {len(chunks_with_topics)} chunks")

    def ingest_directory(
        self,
        directory: str,
        topic_area: str,
        namespace: str = "knowledge_base",
        force: bool = False
    ):
        """Incrementally ingest all supported files in a directory."""
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

    # ── Section-aware chunking ─────────────────────────────────────────────────

    def _chunk_by_section(self, text: str, fallback_topic: str) -> list[dict]:
        """
        Split text into sections based on header detection.
        Each chunk is tagged with the curriculum topic matching its section.

        Returns list of dicts:
        [
            {
                "text": "...",
                "topic_area": "Exploratory Data Analysis",
                "section": "1.2 EDA Techniques"
            },
            ...
        ]
        """
        lines    = text.split("\n")
        sections = []
        current_section_header = "Introduction"
        current_lines          = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append(line)
                continue

            # Check if this line is a section header
            if self._is_header(stripped):
                # Save previous section if it has content
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append({
                            "header": current_section_header,
                            "text":   section_text
                        })
                # Start new section
                current_section_header = stripped
                current_lines          = []
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append({
                    "header": current_section_header,
                    "text":   section_text
                })

        # If no sections detected, fall back to word-count chunking
        if len(sections) <= 1:
            return self._chunk_by_words(text, fallback_topic)

        # Convert sections to chunks with topic metadata
        chunks_with_topics = []
        for section in sections:
            topic = self._map_header_to_topic(section["header"], fallback_topic)

            # Split large sections into smaller word-count chunks
            words = section["text"].split()
            if len(words) <= RAG_CHUNK_SIZE:
                chunks_with_topics.append({
                    "text":       section["text"],
                    "topic_area": topic,
                    "section":    section["header"]
                })
            else:
                # Large section — split into word-count chunks, same topic tag
                i = 0
                while i < len(words):
                    chunk = " ".join(words[i:i + RAG_CHUNK_SIZE])
                    if chunk.strip():
                        chunks_with_topics.append({
                            "text":       chunk,
                            "topic_area": topic,
                            "section":    section["header"]
                        })
                    i += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP

        return chunks_with_topics

    def _is_header(self, line: str) -> bool:
        """Detect if a line is a section header."""
        for pattern in HEADER_PATTERNS:
            if re.match(pattern, line):
                return True
        return False

    def _map_header_to_topic(self, header: str, fallback: str) -> str:
        """
        Map a section header to the nearest curriculum Topic name.
        Falls back to the file-level topic if no match found.
        """
        header_lower = header.lower()
        for keyword, topic in SECTION_TOPIC_MAPPING.items():
            if keyword in header_lower:
                return topic
        return fallback

    def _chunk_by_words(self, text: str, topic_area: str) -> list[dict]:
        """Fallback: simple word-count chunking when no headers detected."""
        words  = text.split()
        chunks = []
        i      = 0
        while i < len(words):
            chunk = " ".join(words[i:i + RAG_CHUNK_SIZE])
            if chunk.strip():
                chunks.append({
                    "text":       chunk,
                    "topic_area": topic_area,
                    "section":    "general"
                })
            i += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
        return chunks

    def _extract_text(self, filepath: str) -> str:
        ext = Path(filepath).suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(filepath)
            return "\n".join(
                page.extract_text() for page in reader.pages
                if page.extract_text()
            )
        elif ext in [".html", ".htm"]:
            with open(filepath, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n", strip=True)
        elif ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print(f"Unsupported file type: {ext}")
            return ""
