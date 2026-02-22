# rag/fetch_docs.py
# Downloads free AI/ML textbooks and runs incremental ingestion into Pinecone
# Run once: python rag/fetch_docs.py
# Safe to re-run — skips already ingested files

import os
import requests
from pathlib import Path
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever
from rag.ingest import DocumentIngester

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

# ── Documents to download ────────────────────────────────────────────────────
# All free and legally available
DOCUMENTS = [
    # Python Data Science Handbook — Jake VanderPlas
    {
        "url":        "https://jakevdp.github.io/PythonDataScienceHandbook/",
        "filename":   "python_data_science_handbook.html",
        "topic_area": "python_data_science",
        "namespace":  "knowledge_base",
        "description": "Python Data Science Handbook — NumPy, Pandas, Matplotlib, Scikit-Learn"
    },
    # Think Stats — Allen Downey
    {
        "url":        "https://greenteapress.com/thinkstats2/html/index.html",
        "filename":   "think_stats.html",
        "topic_area": "statistics_probability",
        "namespace":  "knowledge_base",
        "description": "Think Stats — Statistics and Probability for Data Science"
    },
    # Think Stats Chapter on EDA
    {
        "url":        "https://greenteapress.com/thinkstats2/html/thinkstats2002.html",
        "filename":   "think_stats_eda.html",
        "topic_area": "data_wrangling_eda",
        "namespace":  "knowledge_base",
        "description": "Think Stats — Exploratory Data Analysis chapter"
    },
    # Pandas documentation intro
    {
        "url":        "https://pandas.pydata.org/docs/user_guide/10min.html",
        "filename":   "pandas_10min.html",
        "topic_area": "python_data_science",
        "namespace":  "knowledge_base",
        "description": "10 Minutes to Pandas — official Pandas tutorial"
    },
    # NumPy quickstart
    {
        "url":        "https://numpy.org/doc/stable/user/quickstart.html",
        "filename":   "numpy_quickstart.html",
        "topic_area": "python_data_science",
        "namespace":  "knowledge_base",
        "description": "NumPy Quickstart Tutorial — official NumPy docs"
    },
    # Matplotlib tutorials
    {
        "url":        "https://matplotlib.org/stable/tutorials/introductory/usage.html",
        "filename":   "matplotlib_intro.html",
        "topic_area": "python_data_science",
        "namespace":  "knowledge_base",
        "description": "Matplotlib Usage Guide — official intro tutorial"
    },
]


def download_document(url: str, filepath: Path) -> bool:
    """Download a document from URL and save to filepath."""
    if filepath.exists():
        print(f"Already downloaded: {filepath.name}")
        return True
    try:
        print(f"Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0 MOSAICurriculum/1.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        filepath.write_bytes(response.content)
        print(f"Saved: {filepath.name} ({len(response.content) // 1024}KB)")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def run_ingestion():
    """Download all documents and incrementally ingest into Pinecone."""
    print("=" * 60)
    print("MOSAIC Curriculum — Document Ingestion")
    print("=" * 60)

    # Initialise components
    print("\nInitialising embedder and Pinecone...")
    embedder  = BGEEmbedder()
    retriever = RAGRetriever(embedder)
    ingester  = DocumentIngester(retriever, embedder)

    print(f"\nProcessing {len(DOCUMENTS)} documents...")
    print("-" * 60)

    success_count = 0
    skip_count    = 0

    for doc in DOCUMENTS:
        filepath = DOCS_DIR / doc["filename"]
        print(f"\n📄 {doc['description']}")

        # Download if not already saved locally
        downloaded = download_document(doc["url"], filepath)
        if not downloaded:
            print(f"Skipping ingestion — download failed")
            continue

        # Check if already in Pinecone
        already = retriever.get_ingested_sources(doc["namespace"])
        if doc["filename"] in already:
            print(f"Already in Pinecone — skipping ingestion")
            skip_count += 1
            continue

        # Ingest into Pinecone
        ingester.ingest_file(
            filepath=str(filepath),
            topic_area=doc["topic_area"],
            source=doc["filename"],
            namespace=doc["namespace"]
        )
        success_count += 1

    print("\n" + "=" * 60)
    print(f"Ingestion complete:")
    print(f"  New documents ingested : {success_count}")
    print(f"  Already in Pinecone    : {skip_count}")
    print(f"  Total documents        : {len(DOCUMENTS)}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
