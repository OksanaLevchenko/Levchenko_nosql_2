
import os
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Load API key
load_dotenv()

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if it does not exist
if INDEX_NAME not in pc.list_indexes().names():
    print("Creating Pinecone index...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print("Index already exists.")

# Wait until the index is ready
while not pc.describe_index(INDEX_NAME).status["ready"]:
    print("Waiting for index...")
    time.sleep(5)

index = pc.Index(INDEX_NAME)

# Read articles and embeddings
df = pd.read_parquet("data/arxiv_subset.parquet")
embeddings = np.load("embeddings/embeddings.npy")

print("Articles:", len(df))
print("Embeddings shape:", embeddings.shape)

# Verify data
assert embeddings.shape == (len(df), VECTOR_DIM)

# Upload in batches
for start in tqdm(
    range(0, len(df), BATCH_SIZE),
    desc="Uploading to Pinecone"
):
    batch = []

    for i in range(start, min(start + BATCH_SIZE, len(df))):
        paper = df.iloc[i]

        batch.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": str(paper["id"]),
                "title": str(paper["title"]),
                "abstract": str(paper["abstract"])[:500],
                "authors": str(paper["authors"])[:200],
                "year": int(paper["year"]),
                "category": str(paper["category"])
            }
        })

    index.upsert(vectors=batch)

# Check index
print("\nUpload completed!")
print("Index statistics:")
print(index.describe_index_stats())
