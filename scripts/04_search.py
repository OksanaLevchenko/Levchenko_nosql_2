
import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ==========================================
# Settings
# ==========================================

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

# Load model and local data
print("Loading SPECTER2 model...")
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet")
embeddings = np.load("embeddings/embeddings.npy")

print("Articles loaded:", len(df))
print("Embeddings shape:", embeddings.shape)


# ==========================================
# Helper functions
# ==========================================

def encode_query(text):
    # Convert search text into a normalized vector
    vector = model.encode(
        text,
        normalize_embeddings=True
    )
    return np.asarray(vector, dtype=np.float32)


def search_pinecone(query, filters=None):
    vector = encode_query(query)

    response = index.query(
        vector=vector.tolist(),
        top_k=TOP_K,
        include_metadata=True,
        filter=filters
    )

    return response.matches


def print_results(results):
    if not results:
        print("No matching articles found.")
        return

    for number, match in enumerate(results, 1):
        metadata = match.metadata or {}

        print(f"\n{number}. {metadata.get('title', 'Unknown')}")
        print("Category:", metadata.get("category"))
        print("Year:", metadata.get("year"))
        print("Score:", round(float(match.score), 4))
        print("Abstract:", metadata.get("abstract", "")[:200])


# ==========================================
# Task 1 — Semantic Search
# ==========================================

print("\n========== TASK 1: SEMANTIC SEARCH ==========")

query = "teaching machines to recognize objects in pictures"

results = search_pinecone(query)
print_results(results)


# ==========================================
# Task 2 — Filtered Search
# ==========================================

print("\n========== TASK 2A: RECENT REINFORCEMENT LEARNING ==========")

recent_query = "reinforcement learning algorithms"

# Last 5 calendar years, including 2026
recent_filter = {
    "year": {"$gte": 2022},
    "category": {"$eq": "cs.LG"}
}

recent_results = search_pinecone(
    recent_query,
    filters=recent_filter
)

print_results(recent_results)


print("\n========== TASK 2B: OLDER ARTICLES ==========")

old_filter = {
    "year": {"$lt": 2015}
}

old_results = search_pinecone(
    recent_query,
    filters=old_filter
)

print_results(old_results)


# ==========================================
# Task 3 — Compare Similarity Metrics
# ==========================================

print("\n========== TASK 3: SIMILARITY METRICS ==========")

query_vector = encode_query(query)

# Normalize document vectors for a fair comparison
norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

normalized_embeddings = embeddings / np.maximum(norms, 1e-12)

# Cosine similarity
cosine_scores = normalized_embeddings @ query_vector

# Dot product
dot_scores = embeddings @ query_vector

# Euclidean (L2) distance
l2_distances = np.linalg.norm(
    embeddings - query_vector,
    axis=1
)


def show_top_five(scores, metric, higher_is_better=True):
    if higher_is_better:
        top_ids = np.argsort(-scores)[:TOP_K]
    else:
        top_ids = np.argsort(scores)[:TOP_K]

    print(f"\n--- Top 5 using {metric} ---")

    for rank, i in enumerate(top_ids, 1):
        print(
            f"{rank}. {df.iloc[i]['title']} "
            f"(score: {scores[i]:.4f})"
        )

    return top_ids.tolist()


cosine_top = show_top_five(
    cosine_scores,
    "Cosine Similarity"
)

dot_top = show_top_five(
    dot_scores,
    "Dot Product"
)

l2_top = show_top_five(
    l2_distances,
    "L2 Distance",
    higher_is_better=False
)

print("\n========== COMPARISON ==========")

print("Cosine vs Dot same ranking:", cosine_top == dot_top)
print("Cosine vs L2 same ranking:", cosine_top == l2_top)

print("\nSearch completed!")
