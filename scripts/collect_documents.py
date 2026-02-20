# scripts/collect_documents.py
# Download free AI/ML documents and ingest into ChromaDB
# Run once before building the KG

import os
import requests
import arxiv
from pathlib import Path
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever
from rag.ingest import DocumentIngester

DATA_DIR = Path("./data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_arxiv_papers():
    """Download key ML/AI papers from ArXiv."""
    print("Downloading ArXiv papers...")

    papers = [
        "1706.03762",  # Attention Is All You Need (Transformer)
        "1810.04805",  # BERT
        "2005.14165",  # GPT-3
        "1512.03385",  # ResNet
        "1301.3666",   # Word2Vec
        "2106.09685",  # LoRA
        "2005.11401",  # RAG paper
    ]

    client = arxiv.Client()
    folder = DATA_DIR / "arxiv"
    folder.mkdir(exist_ok=True)

    for paper_id in papers:
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper  = next(client.results(search))
            out    = folder / f"{paper_id.replace('/', '_')}.pdf"
            if not out.exists():
                paper.download_pdf(filename=str(out))
                print(f"  Downloaded: {paper.title}")
            else:
                print(f"  Already exists: {paper.title}")
        except Exception as e:
            print(f"  Failed {paper_id}: {e}")


def download_web_docs():
    """Download official documentation pages."""
    print("Downloading documentation...")

    docs = [
        {
            "url":    "https://pytorch.org/docs/stable/nn.html",
            "name":   "pytorch_nn.html",
            "topic":  "deep_learning"
        },
        {
            "url":    "https://scikit-learn.org/stable/supervised_learning.html",
            "name":   "sklearn_supervised.html",
            "topic":  "classical_ml"
        },
        {
            "url":    "https://huggingface.co/docs/transformers/index",
            "name":   "hf_transformers.html",
            "topic":  "llm_engineering"
        },
    ]

    folder = DATA_DIR / "docs"
    folder.mkdir(exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (educational bot)"}

    for doc in docs:
        out = folder / doc["name"]
        if out.exists():
            print(f"  Already exists: {doc['name']}")
            continue
        try:
            r = requests.get(doc["url"], headers=headers, timeout=15)
            out.write_text(r.text, encoding="utf-8")
            print(f"  Downloaded: {doc['name']}")
        except Exception as e:
            print(f"  Failed {doc['name']}: {e}")


def ingest_all():
    """Ingest all downloaded documents into ChromaDB."""
    print("\nIngesting documents into ChromaDB...")

    embedder  = BGEEmbedder()
    retriever = RAGRetriever(embedder)
    ingester  = DocumentIngester(retriever, embedder)

    topic_map = {
        "arxiv":    "deep_learning",
        "docs":     "general",
    }

    for subfolder, topic in topic_map.items():
        folder = DATA_DIR / subfolder
        if folder.exists():
            ingester.ingest_folder(str(folder), topic_area=topic)

    print("Ingestion complete.")


if __name__ == "__main__":
    download_arxiv_papers()
    download_web_docs()
    ingest_all()
    print("\nAll documents collected and ingested.")
    print("You can now run: python scripts/build_kg.py")
