# rag/fetch_docs.py
# Incremental ingestion — scans docs/ folder and ingests any new files
# No hardcoding — just drop files into docs/ and run

import json
from pathlib import Path
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever
from rag.ingest import DocumentIngester

DOCS_DIR = Path("docs")

# Topic area mapping based on filename keywords
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

    # Python for Data Science (catch-all for general python)
    "python":      "Python for Data Science",
    "pandas":      "Python for Data Science",
    "numpy":       "Python for Data Science",
    "stats":       "Python for Data Science",
    "statistic":   "Python for Data Science",
}

SUPPORTED_FORMATS = [".pdf", ".html", ".htm", ".txt", ".md", ".ipynb"]


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


def get_topic_area(filename: str) -> str:
    """
    Infer topic area from filename keywords.
    Returns exact curriculum Topic name for precise Pinecone filtering.
    Falls back to 'Python for Data Science' if no keyword matches.
    """
    name_lower = filename.lower()
    for keyword, topic in TOPIC_MAPPING.items():
        if keyword in name_lower:
            return topic
    return "Python for Data Science"


def run_ingestion():
    """
    Scan docs/ folder and incrementally ingest any new files into Pinecone.
    Skips files already ingested. Safe to run on every app startup.
    """
    print("=" * 60)
    print("MOSAIC Curriculum — Incremental Document Ingestion")
    print("=" * 60)

    # Check docs/ folder exists
    if not DOCS_DIR.exists():
        print(f"docs/ folder not found — creating it")
        DOCS_DIR.mkdir(exist_ok=True)
        print("Add your PDF/HTML/TXT/MD files to docs/ and redeploy")
        return

    # Find all supported files
    all_files = [
        f for f in DOCS_DIR.rglob("*")
        if f.suffix.lower() in SUPPORTED_FORMATS
        and f.is_file()
    ]

    if not all_files:
        print("No documents found in docs/ folder")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        return

    print(f"Found {len(all_files)} files in docs/")

    # Initialise components
    print("Initialising embedder and Pinecone...")
    embedder  = BGEEmbedder()
    retriever = RAGRetriever(embedder)
    ingester  = DocumentIngester(retriever, embedder)

    # Check what's already in Pinecone
    already_ingested = retriever.get_ingested_sources("knowledge_base")
    new_files        = [f for f in all_files if f.name not in already_ingested]

    print(f"Already in Pinecone : {len(already_ingested)} files")
    print(f"New files to ingest : {len(new_files)} files")

    if not new_files:
        print("Nothing new to ingest — Pinecone is up to date")
        return

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
            print(f"Failed to ingest {filepath.name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Ingestion complete:")
    print(f"  Successfully ingested : {success} files")
    print(f"  Failed                : {failed} files")
    print(f"  Already in Pinecone   : {len(already_ingested)} files")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
