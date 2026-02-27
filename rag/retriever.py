# rag/retriever.py
# Pinecone vector store retrieval — used by all teaching agents
# Includes query expansion to resolve abbreviations before embedding

import os
import re
from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    RAG_TOP_K
)
from rag.embedder import BGEEmbedder

# ── Abbreviation expansion map ─────────────────────────────────────────────
ABBREVIATION_MAP = {
    # Models
    "tft":       "Temporal Fusion Transformer time series forecasting",
    "lstm":      "Long Short-Term Memory recurrent neural network",
    "cnn":       "Convolutional Neural Network",
    "rnn":       "Recurrent Neural Network",
    "gru":       "Gated Recurrent Unit",
    "bert":      "Bidirectional Encoder Representations from Transformers",
    "gpt":       "Generative Pre-trained Transformer",
    "vae":       "Variational Autoencoder",
    "gan":       "Generative Adversarial Network",
    "xgb":       "XGBoost gradient boosting",
    "lgbm":      "LightGBM gradient boosting",
    "rf":        "Random Forest ensemble",
    "svm":       "Support Vector Machine",
    "knn":       "K-Nearest Neighbors",
    # Techniques
    "pca":       "Principal Component Analysis dimensionality reduction",
    "tsne":      "t-SNE t-distributed Stochastic Neighbor Embedding",
    "t-sne":     "t-SNE t-distributed Stochastic Neighbor Embedding",
    "umap":      "Uniform Manifold Approximation and Projection",
    "smote":     "Synthetic Minority Oversampling Technique imbalanced data",
    "adasyn":    "Adaptive Synthetic Sampling imbalanced data",
    "ros":       "Random Oversampling imbalanced data",
    "rus":       "Random Undersampling imbalanced data",
    "mice":      "Multiple Imputation by Chained Equations missing values",
    # Libraries
    "tf":        "TensorFlow machine learning framework",
    "sklearn":   "scikit-learn machine learning library",
    "pd":        "pandas DataFrame data manipulation",
    "np":        "numpy numerical computing array",
    # Metrics
    "mse":       "Mean Squared Error regression metric",
    "mae":       "Mean Absolute Error regression metric",
    "rmse":      "Root Mean Squared Error regression metric",
    "mape":      "Mean Absolute Percentage Error forecasting metric",
    "auc":       "Area Under the ROC Curve classification metric",
    "roc":       "Receiver Operating Characteristic curve",
    "f1":        "F1 score precision recall harmonic mean",
    # Data concepts
    "eda":       "Exploratory Data Analysis",
    "etl":       "Extract Transform Load data pipeline",
    "nlp":       "Natural Language Processing text",
    "cv":        "Cross Validation model evaluation",
}


# Common English words to never fuzzy-correct
_COMMON_WORDS = {
    "the","and","for","are","but","not","you","all","can","has","her","was","one",
    "our","out","day","get","him","his","how","its","may","new","now","old","see",
    "two","who","did","yes","use","way","she","had","let","put","say","too","vs",
    "or","to","in","is","it","if","of","on","at","as","an","a","test","from","good",
    "data","with","this","that","have","then","when","what","will","each","they",
    "been","than","more","some","time","very","also","both","such","like","just",
    "does","show","give","make","tell","need","want","help","code","work","used",
    "using","which","where","after","before","between","difference","better",
}


def _levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]


def _looks_like_abbreviation(word: str) -> bool:
    if word in _COMMON_WORDS:
        return False
    vowels = sum(1 for c in word if c in "aeiou")
    if len(word) <= 6 and vowels <= 1:
        return True
    if len(word) <= 7 and vowels <= 2:
        return True
    return False


def correct_typos(query: str) -> str:
    known_abbrevs = list(ABBREVIATION_MAP.keys())
    words         = query.split()
    corrected     = []
    for word in words:
        clean = word.lower().strip(".,?!")
        if clean in _COMMON_WORDS:
            corrected.append(word)
            continue
        max_dist   = 2 if _looks_like_abbreviation(clean) else 1
        best_match = None
        best_dist  = max_dist + 1
        for abbrev in known_abbrevs:
            if abs(len(clean) - len(abbrev)) > max_dist:
                continue
            dist = _levenshtein(clean, abbrev)
            if dist < best_dist:
                best_dist  = dist
                best_match = abbrev
        if best_match and best_dist <= max_dist and best_match != clean:
            print(f"Typo corrected: '{clean}' -> '{best_match}'")
            corrected.append(best_match)
        else:
            corrected.append(word)
    return " ".join(corrected)


def expand_query(query: str) -> str:
    """
    1. Correct typos (e.g. "tgt" → "tft")
    2. Expand abbreviations (e.g. "tft" → "Temporal Fusion Transformer...")
    3. Append original query for hybrid matching

    e.g. "tgt vs 1D CNN-LSTM" →
         "Temporal Fusion Transformer time series forecasting vs
          1D Convolutional Neural Network Long Short-Term Memory (tgt vs 1D CNN-LSTM)"
    """
    # Step 1: fix typos first
    corrected = correct_typos(query)

    # Step 2: expand abbreviations
    expanded = corrected
    for abbrev, full in ABBREVIATION_MAP.items():
        pattern = re.compile(r'\b' + re.escape(abbrev) + r'\b', re.IGNORECASE)
        if pattern.search(expanded):
            expanded = pattern.sub(full, expanded)

    # Step 3: append original query for hybrid matching
    if expanded != query:
        expanded = f"{expanded} ({query})"

    return expanded


class RAGRetriever:
    """
    Pinecone retriever.
    All agents (Solver, Recommender, Assessment, Feedback)
    call different convenience methods based on their needs.

    Query expansion is applied in retrieve_for_solver and retrieve_for_recommender
    to resolve abbreviations before embedding.
    """

    def __init__(self, embedder: BGEEmbedder):
        self.embedder = embedder
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = [i.name for i in self.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index created: {PINECONE_INDEX_NAME}")
        else:
            print(f"Pinecone index ready: {PINECONE_INDEX_NAME}")

        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def retrieve(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        topic_filter: str = None,
        namespace: str = "knowledge_base"
    ) -> list[dict]:
        """Base retrieval — no query expansion (raw query)."""
        query_embedding = self.embedder.embed_query(query)

        filter_dict = {}
        if topic_filter:
            filter_dict["topic_area"] = {"$eq": topic_filter}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict if filter_dict else None,
            include_metadata=True
        )

        return [
            {
                "text":   match["metadata"].get("text", ""),
                "source": match["metadata"].get("source", "unknown"),
                "topic":  match["metadata"].get("topic_area", "general"),
                "score":  match["score"]
            }
            for match in results["matches"]
        ]

    def retrieve_for_solver(self, query: str, topic: str = None) -> list[dict]:
        """Solver — retrieval with query expansion."""
        expanded = expand_query(query)
        if expanded != query:
            print(f"RAG expanded: '{query[:50]}' → '{expanded[:80]}'")
        return self.retrieve(expanded, top_k=5, topic_filter=topic, namespace="knowledge_base")

    def retrieve_for_recommender(self, query: str) -> list[dict]:
        """
        Recommender — retrieval with query expansion + higher top_k.
        Needs more chunks for comparisons involving multiple methods.
        """
        expanded = expand_query(query)
        if expanded != query:
            print(f"RAG expanded: '{query[:50]}' → '{expanded[:80]}'")
        return self.retrieve(expanded, top_k=6, namespace="knowledge_base")

    def retrieve_for_assessment(self, concept: str) -> list[dict]:
        """Assessment — retrieves question context."""
        return self.retrieve(
            f"assessment questions about {concept}",
            top_k=3,
            namespace="knowledge_base"
        )

    def retrieve_for_feedback(self, misconception: str) -> list[dict]:
        """Feedback — gets explanation strategies for misconceptions."""
        return self.retrieve(
            f"how to explain {misconception} clearly",
            top_k=3,
            namespace="knowledge_base"
        )

    def get_ingested_sources(self, namespace: str = "knowledge_base") -> set[str]:
        """Returns set of already-ingested source filenames."""
        try:
            stats    = self.index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            if ns_stats.get("vector_count", 0) == 0:
                return set()

            dummy   = [0.0] * 384
            results = self.index.query(
                vector=dummy,
                top_k=10000,
                namespace=namespace,
                include_metadata=True
            )
            return {
                match["metadata"].get("source", "")
                for match in results["matches"]
                if match["metadata"].get("source")
            }
        except Exception as e:
            print(f"get_ingested_sources error: {e}")
            return set()
