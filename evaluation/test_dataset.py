# evaluation/test_dataset.py
# Builds a structured test dataset from FODS Question Bank PDF
# Used by ragas_eval.py and ares_eval.py for RAG evaluation
#
# Output: evaluation/results/test_dataset.json
#
# Format per entry:
#   question     — question from the PDF
#   ground_truth — correct answer from the PDF
#   unit         — which unit the question comes from
#   type         — mcq / short / long
#   answer       — left empty, filled by ragas_eval.py after querying Solver
#   contexts     — left empty, filled by ragas_eval.py after querying Pinecone

import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# DATASET — manually extracted from FODS Question Bank PDF
# 40 questions across 4 units covering MCQ, short, and long answer
# ─────────────────────────────────────────────────────────────

DATASET = [

    # ══════════════════════════════════════════════════════════
    # UNIT I — Data Science Fundamentals
    # ══════════════════════════════════════════════════════════

    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "What is the primary goal of data science?",
        "ground_truth": "To extract insights and knowledge from data.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "Which of the following is NOT a typical application of data science?",
        "ground_truth": "Designing user interfaces for mobile apps.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "What is a data scientist responsible for?",
        "ground_truth": "Collecting and cleaning raw data.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "Why is Python a popular choice for data analysis?",
        "ground_truth": "It has a large collection of data science libraries.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "What is the first step in the data science life cycle?",
        "ground_truth": "Data understanding and pre-processing.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "How is structured data different from unstructured data?",
        "ground_truth": "Structured data is organized in tables, while unstructured data is not.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "mcq",
        "question": "What is a synthetic dataset?",
        "ground_truth": "A dataset artificially created with specific characteristics.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "short",
        "question": "What is data science and how is it applied?",
        "ground_truth": "Data science is a field that extracts knowledge and insights from data. It is used in finance for fraud detection, retail for product recommendations, healthcare for predictive medicine, and weather forecasting.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "short",
        "question": "What is data pre-processing and why is it significant?",
        "ground_truth": "Data pre-processing is cleaning and preparing raw data for analysis. It is crucial to handle missing values, inconsistencies, and formatting issues to ensure reliable results.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit I",
        "type": "long",
        "question": "Explain the difference between structured and unstructured data and discuss the challenges of analyzing unstructured data.",
        "ground_truth": "Structured data is highly organized in a predefined format like rows and columns. Unstructured data lacks consistent format including text, images, audio, and video. Challenges include heterogeneity, scalability, and meaning extraction. Data science addresses these with NLP, computer vision, and text mining.",
        "answer": "",
        "contexts": []
    },

    # ══════════════════════════════════════════════════════════
    # UNIT II — Python and Pandas
    # ══════════════════════════════════════════════════════════

    {
        "unit": "Unit II",
        "type": "mcq",
        "question": "Which of the following is a valid variable name in Python?",
        "ground_truth": "_variable1",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "mcq",
        "question": "What is the correct syntax to print a value in Python?",
        "ground_truth": "print(\"Hello, World!\")",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "mcq",
        "question": "What is the output of the following code: print(2 + 3 * 4)?",
        "ground_truth": "14",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "mcq",
        "question": "How can you import a dataset in Python using Pandas?",
        "ground_truth": "import pandas as pd; df = pd.read_csv('file.csv')",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "mcq",
        "question": "Which function is used to display the first few rows of a DataFrame in Pandas?",
        "ground_truth": "head()",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "mcq",
        "question": "How can missing values be handled in a dataset using Pandas?",
        "ground_truth": "Both df.fillna(0) and df.dropna() can be used to handle missing values.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "short",
        "question": "What is a variable in Python?",
        "ground_truth": "A variable in Python is a name given to a memory location that stores a value. It acts as a placeholder for storing data that can be used and manipulated within a program.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "short",
        "question": "What method would you use to handle missing values by filling them with the mean in Pandas?",
        "ground_truth": "df.fillna(df.mean(), inplace=True) fills missing values with the mean of the respective columns.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "short",
        "question": "How can you get basic insights like the number of rows and columns in a Pandas DataFrame?",
        "ground_truth": "You can use the df.shape attribute or the df.info() method to get basic insights such as the number of rows and columns.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit II",
        "type": "long",
        "question": "Explain how to clean and prepare a dataset by identifying and handling missing values.",
        "ground_truth": "Missing values can be identified using df.isnull() or df.isna(). To handle them: use df.dropna() to remove rows or columns with missing values, or df.fillna(value) to fill with mean, median, or a constant. For example df['column'].fillna(df['column'].mean(), inplace=True) replaces missing values with the column mean.",
        "answer": "",
        "contexts": []
    },

    # ══════════════════════════════════════════════════════════
    # UNIT III — NumPy, Pandas, Matplotlib, Seaborn, SciPy
    # ══════════════════════════════════════════════════════════

    {
        "unit": "Unit III",
        "type": "mcq",
        "question": "Which library is primarily used for numerical computations in Python?",
        "ground_truth": "NumPy",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "mcq",
        "question": "What is the main data structure used in Pandas?",
        "ground_truth": "DataFrame",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "mcq",
        "question": "Which function in Matplotlib is used to create a basic plot?",
        "ground_truth": "plt.plot()",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "mcq",
        "question": "What is the primary purpose of the SciPy library?",
        "ground_truth": "Scientific computing",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "mcq",
        "question": "Which Matplotlib function is used to create a scatter plot?",
        "ground_truth": "plt.scatter()",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "mcq",
        "question": "What command in Matplotlib displays the plot to the screen?",
        "ground_truth": "plt.show()",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "short",
        "question": "What is a NumPy array and how is it different from a Python list?",
        "ground_truth": "A NumPy array is a powerful n-dimensional array object which is faster and more efficient for numerical computations compared to Python lists. It supports element-wise operations and broadcasting.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "short",
        "question": "What is the use of the sns.pairplot() function in Seaborn?",
        "ground_truth": "The sns.pairplot() function creates a grid of scatter plots for pairwise relationships in a dataset, allowing for quick visualization of correlations and distributions.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "short",
        "question": "What is the purpose of the plt.hist() function in Matplotlib?",
        "ground_truth": "The plt.hist() function is used to create a histogram, which is a graphical representation of the distribution of a dataset.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit III",
        "type": "long",
        "question": "Explain the basic operations that can be performed using NumPy arrays.",
        "ground_truth": "NumPy arrays support creation using np.array(), np.zeros(), np.ones(), np.arange(), np.linspace(). They support indexing and slicing for multi-dimensional arrays, element-wise operations like addition and multiplication, aggregation functions like np.sum() and np.mean(), broadcasting for arithmetic between different shapes, and reshaping using np.reshape() and np.flatten().",
        "answer": "",
        "contexts": []
    },

    # ══════════════════════════════════════════════════════════
    # UNIT IV — Machine Learning and Recommender Systems
    # ══════════════════════════════════════════════════════════

    {
        "unit": "Unit IV",
        "type": "mcq",
        "question": "What type of learning involves training a model on labeled data?",
        "ground_truth": "Supervised Learning",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "mcq",
        "question": "In unsupervised learning, which algorithm is commonly used for clustering data?",
        "ground_truth": "K-Means",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "mcq",
        "question": "What is a recommender system used for?",
        "ground_truth": "Providing personalized recommendations",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "mcq",
        "question": "Which method is commonly used to evaluate the performance of a supervised learning model?",
        "ground_truth": "Cross-validation",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "mcq",
        "question": "In decision-making models, what is a decision tree used for?",
        "ground_truth": "Making decisions based on rules and conditions",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "mcq",
        "question": "Which Python library is commonly used for building machine learning models including supervised and unsupervised learning?",
        "ground_truth": "Scikit-learn",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "short",
        "question": "What is supervised learning?",
        "ground_truth": "Supervised learning is a type of machine learning where a model is trained on labeled data. The input data comes with associated correct outputs and the model learns to map inputs to outputs to make predictions on new unseen data.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "short",
        "question": "What is the main difference between supervised and unsupervised learning?",
        "ground_truth": "Supervised learning uses labeled data to train the model, whereas unsupervised learning uses unlabeled data and aims to find hidden patterns or intrinsic structures in the input data.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "short",
        "question": "What is collaborative filtering in the context of recommender systems?",
        "ground_truth": "Collaborative filtering makes recommendations based on the preferences and behavior of similar users. It assumes that users who agreed in the past will agree in the future.",
        "answer": "",
        "contexts": []
    },
    {
        "unit": "Unit IV",
        "type": "long",
        "question": "Explain the difference between linear regression and logistic regression and provide examples of when each would be used.",
        "ground_truth": "Linear regression is used for predicting continuous values using a linear equation. Example: predicting house prices. Logistic regression is used for binary classification problems using a logistic function. Example: predicting whether a student will pass or fail. Both are supervised learning algorithms but serve different problem types.",
        "answer": "",
        "contexts": []
    },
]


def build_dataset(output_path: str = "evaluation/results/test_dataset.json"):
    """
    Save the test dataset to a JSON file.
    The answer and contexts fields are empty — they get filled
    by ragas_eval.py when it queries the Solver Agent and Pinecone.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(DATASET, f, indent=2, ensure_ascii=False)

    # Print summary
    units  = {}
    types  = {}
    for item in DATASET:
        units[item["unit"]] = units.get(item["unit"], 0) + 1
        types[item["type"]] = types.get(item["type"], 0) + 1

    print("=" * 60)
    print("MOSAIC — Test Dataset Built")
    print("=" * 60)
    print(f"Total questions : {len(DATASET)}")
    print()
    print("By unit:")
    for unit, count in sorted(units.items()):
        print(f"  {unit}: {count} questions")
    print()
    print("By type:")
    for qtype, count in sorted(types.items()):
        print(f"  {qtype}: {count} questions")
    print()
    print(f"Saved to: {output_file}")
    print("=" * 60)
    print()
    print("Next step: run ragas_eval.py to fill in 'answer' and 'contexts'")
    print("by querying your Solver Agent and Pinecone for each question.")


if __name__ == "__main__":
    build_dataset()
