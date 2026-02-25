# rag/fetch_docs.py
# Downloads docs from TWO sources then ingests into Pinecone:
#   1. HuggingFace dataset repo (primary — lvictoria/mosaicdocs)
#   2. Local docs/ folder (secondary — any files already there)
# Safe to run on every app startup — skips already-ingested files

import json
import os
from pathlib import Path
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever
from rag.ingest import DocumentIngester

DOCS_DIR = Path("docs")

SUPPORTED_FORMATS = [".pdf", ".html", ".htm", ".txt", ".md", ".ipynb"]

# ── Secret helper ─────────────────────────────────────────────────────────────
def _get_secret(key, default=""):
    try:
        import streamlit as st
        return st.secrets.get(key, default) or os.getenv(key, default)
    except Exception:
        return os.getenv(key, default)

HF_REPO_ID = "lvictoria/mosaicdocs"
HF_TOKEN   = _get_secret("HF_TOKEN", "")

# ── Topic mapping ─────────────────────────────────────────────────────────────
# Must match EXACT curriculum Topic names in Neo4j
TOPIC_MAPPING = {
    # Reading Structured Files
    "excel":       "Reading Structured Files",
    "csv":         "Reading Structured Files",
    "json":        "Reading Structured Files",
    "parquet":     "Reading Structured Files",
    "read":        "Reading Structured Files",
    "file":        "Reading Structured Files",
    "structured":  "Reading Structured Files",

    # Structured Data Types
    "dataframe":   "Structured Data Types",
    "datatype":    "Structured Data Types",
    "data_type":   "Structured Data Types",
    "vector":      "Structured Data Types",
    "array":       "Structured Data Types",
    "tensor":      "Structured Data Types",
    "sql":         "Structured Data Types",

    # Exploratory Data Analysis
    "eda":         "Exploratory Data Analysis",
    "exploratory": "Exploratory Data Analysis",
    "wrangling":   "Exploratory Data Analysis",
    "cleaning":    "Exploratory Data Analysis",
    "outlier":     "Exploratory Data Analysis",
    "correlation": "Exploratory Data Analysis",
    "missing":     "Exploratory Data Analysis",

    # Data Visualization
    "visual":      "Data Visualization",
    "plot":        "Data Visualization",
    "chart":       "Data Visualization",
    "matplotlib":  "Data Visualization",
    "seaborn":     "Data Visualization",
    "heatmap":     "Data Visualization",

    # Imputation Techniques
    "imputation":  "Imputation Techniques",
    "impute":      "Imputation Techniques",
    "fill":        "Imputation Techniques",
    "interpolat":  "Imputation Techniques",
    "kalman":      "Imputation Techniques",

    # Data Augmentation
    "augment":     "Data Augmentation",
    "smote":       "Data Augmentation",
    "oversample":  "Data Augmentation",
    "imbalance":   "Data Augmentation",

    # Feature Reduction
    "feature":     "Feature Reduction",
    "pca":         "Feature Reduction",
    "reduction":   "Feature Reduction",
    "dimension":   "Feature Reduction",

    # Business Metrics
    "metric":      "Business Metrics",
    "kpi":         "Business Metrics",
    "churn":       "Business Metrics",
    "forecast":    "Business Metrics",
    "business":    "Business Metrics",

    # Preprocessing Summary
    "preprocess":  "Preprocessing Summary",
    "pipeline":    "Preprocessing Summary",
    "transform":   "Preprocessing Summary",

    # ML Frameworks
    "pytorch":     "ML Frameworks",
    "tensorflow":  "ML Frameworks",
    "keras":       "ML Frameworks",
    "sklearn":     "ML Frameworks",
    "ml":          "ML Frameworks",
    "machine":     "ML Frameworks",
    "model":       "ML Frameworks",

    # Python for Data Science (catch-all)
    "python":      "Python for Data Science",
    "pandas":      "Python for Data Science",
    "numpy":       "Python for Data Science",
    "stats":       "Python for Data Science",
    "statistic":   "Python for Data Science",
    "fods":        "Python for Data Science",
}


def get_topic_area(filename: str) -> str:
    """Map filename keywords to curriculum Topic name."""
    name_lower = filename.lower()
    for keyword, topic in TOPIC_MAPPING.items():
        if keyword in name_lower:
            return topic
    return "Python for Data Science"


def extract_text_from_ipynb(filepath: str) -> str:
    """Extract text and code from Jupyter notebook cells."""
    with open(filepath, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    text = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            text += "".join(cell["source"]) + "\n\n"
        elif cell["cell_type"] == "code":
            text += "```python\n" + "".join(cell["source"]) + "\n```\n\n"
    return text


# ── Source 1: HuggingFace ─────────────────────────────────────────────────────

def download_from_huggingface():
    """
    Download all files from HuggingFace dataset repo into docs/.
    Only downloads files not already present locally.
    Requires HF_TOKEN in Streamlit secrets for private repos.
    Skips silently if HF_TOKEN not set.
    """
    if not HF_TOKEN:
        print("No HF_TOKEN set — skipping HuggingFace download")
        return

    print(f"Checking HuggingFace repo: {HF_REPO_ID}...")
    DOCS_DIR.mkdir(exist_ok=True)

    try:
        from huggingface_hub import list_repo_files, hf_hub_download

        # List all files in the HF repo
        all_hf_files = list(list_repo_files(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        ))

        # Filter to supported formats only
        doc_files = [
            f for f in all_hf_files
            if Path(f).suffix.lower() in SUPPORTED_FORMATS
        ]

        if not doc_files:
            print("No supported files found in HuggingFace repo")
            return

        print(f"Found {len(doc_files)} file(s) in HuggingFace repo")
        downloaded = 0

        for hf_path in doc_files:
            filename  = Path(hf_path).name
            dest_path = DOCS_DIR / filename

            # Skip if already downloaded locally
            if dest_path.exists():
                print(f"  Already local: {filename}")
                continue

            # Download from HuggingFace into docs/
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=str(DOCS_DIR)
            )
            print(f"  Downloaded: {filename}")
            downloaded += 1

        print(f"HuggingFace: {downloaded} new file(s) downloaded")

    except Exception as e:
        print(f"HuggingFace download error (non-fatal): {e}")
        print("Continuing with local docs/ files only...")


# ── Source 2: Local docs/ ─────────────────────────────────────────────────────
# No download needed — files already in docs/ from GitHub repo
# Both HF files and local files are picked up in run_ingestion()


# ── Main ingestion ────────────────────────────────────────────────────────────

def run_ingestion():
    """
    Full pipeline:
    1. Download new files from HuggingFace into docs/
    2. Scan docs/ for ALL files (HF downloaded + local)
    3. Ingest new files into Pinecone — skip already-ingested
    """
    print("=" * 60)
    print("MOSAIC Curriculum — Document Sync & Ingestion")
    print("=" * 60)

    # ── Step 1: Pull from HuggingFace ─────────────────────────────
    download_from_huggingface()

    # ── Step 2: Scan docs/ ────────────────────────────────────────
    if not DOCS_DIR.exists():
        print("docs/ folder not found — creating it")
        DOCS_DIR.mkdir(exist_ok=True)
        print("Add files to docs/ or upload to HuggingFace repo")
        return

    all_files = [
        f for f in DOCS_DIR.rglob("*")
        if f.suffix.lower() in SUPPORTED_FORMATS
        and f.is_file()
    ]

    if not all_files:
        print("No documents found in docs/ folder")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        return

    print(f"\nFound {len(all_files)} total file(s) in docs/")

    # ── Step 3: Initialise components ─────────────────────────────
    print("Initialising embedder and Pinecone...")
    embedder  = BGEEmbedder()
    retriever = RAGRetriever(embedder)
    ingester  = DocumentIngester(retriever, embedder)

    # ── Step 4: Skip already-ingested files ───────────────────────
    already_ingested = retriever.get_ingested_sources("knowledge_base")
    new_files        = [f for f in all_files if f.name not in already_ingested]

    print(f"Already in Pinecone : {len(already_ingested)} file(s)")
    print(f"New to ingest       : {len(new_files)} file(s)")

    if not new_files:
        print("Nothing new to ingest — Pinecone is up to date")
        print("=" * 60)
        return

    # ── Step 5: Ingest ────────────────────────────────────────────
    print("-" * 60)
    success = 0
    failed  = 0

    for filepath in new_files:
        topic_area = get_topic_area(filepath.name)
        print(f"\n📄 {filepath.name} → topic: {topic_area}")
        try:
            if filepath.suffix.lower() == ".ipynb":
                text = extract_text_from_ipynb(str(filepath))
                ingester.ingest_text(
                    text=text,
                    topic_area=topic_area,
                    source=filepath.name,
                    namespace="knowledge_base"
                )
            else:
                ingester.ingest_file(
                    filepath=str(filepath),
                    topic_area=topic_area,
                    source=filepath.name,
                    namespace="knowledge_base"
                )
            success += 1
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Ingestion complete:")
    print(f"  Successfully ingested : {success} file(s)")
    print(f"  Failed                : {failed} file(s)")
    print(f"  Already in Pinecone   : {len(already_ingested)} file(s)")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
