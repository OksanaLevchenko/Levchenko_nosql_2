
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

# Settings
INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"

# Smaller batches are safer for Mac memory
BATCH_SIZE = 16

# Create output folder
os.makedirs("embeddings", exist_ok=True)

# Step 1: Load articles
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df)} articles")

# Step 2: Combine title and abstract
texts = (
    df["title"] + " [SEP] " + df["abstract"]
).tolist()

# Step 3: Load embedding model
print("Loading SPECTER2 model...")
model = SentenceTransformer(MODEL_NAME)

# Step 4: Generate normalized embeddings
print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

# Step 5: Check results
print("\nTotal embeddings:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])

first_norm = np.linalg.norm(embeddings[0])
print("First embedding norm:", round(float(first_norm), 4))

assert len(embeddings) == len(df)
assert embeddings.shape[1] == 768
assert np.isclose(first_norm, 1.0, atol=0.001)

# Step 6: Save embeddings
np.save(OUTPUT_FILE, embeddings)

print("\nEmbeddings saved to:", OUTPUT_FILE)
print("Embedding generation completed!")
