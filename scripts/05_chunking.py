
import os
import re
import time

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

FIXED_INDEX = "arxiv-chunks-fixed"
SEMANTIC_INDEX = "arxiv-chunks-semantic"

CHUNK_SIZE = 120
OVERLAP = 25
BATCH_SIZE = 50

# Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Load model and dataset
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet")

# Select 30 papers with the longest abstracts
df["abstract_length"] = df["abstract"].str.len()
long_papers = df.nlargest(30, "abstract_length")

print("Selected papers:", len(long_papers))


# ==========================================
# Fixed-size chunking
# ==========================================

def fixed_chunking(text):
    words = text.split()
    chunks = []

    step = CHUNK_SIZE - OVERLAP

    for start in range(0, len(words), step):
        chunk = words[start:start + CHUNK_SIZE]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


# ==========================================
# Sentence-boundary chunking
# ==========================================

def semantic_chunking(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks = []
    current = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()

        # Handle sentences longer than the chunk size
        if len(words) > CHUNK_SIZE:
            if current:
                chunks.append(" ".join(current))
                current = []
                word_count = 0

            for start in range(0, len(words), CHUNK_SIZE):
                chunks.append(
                    " ".join(words[start:start + CHUNK_SIZE])
                )
            continue

        if word_count + len(words) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            current = []
            word_count = 0

        current.append(sentence)
        word_count += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ==========================================
# Create chunk records
# ==========================================

def prepare_chunks(strategy):
    records = []

    for _, paper in long_papers.iterrows():
        if strategy == "fixed":
            chunks = fixed_chunking(paper["abstract"])
        else:
            chunks = semantic_chunking(paper["abstract"])

        for number, chunk in enumerate(chunks):
            records.append({
                "id": f"{strategy}_{paper['id']}_{number}",
                "title": paper["title"],
                "arxiv_id": paper["id"],
                "chunk": chunk,
                "chunk_number": number,
                "year": int(paper["year"]),
                "category": paper["category"]
            })

    return records


# ==========================================
# Create Pinecone index
# ==========================================

def get_index(name):
    if name not in pc.list_indexes().names():
        print("Creating index:", name)

        pc.create_index(
            name=name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    while not pc.describe_index(name).status["ready"]:
        time.sleep(5)

    return pc.Index(name)


# ==========================================
# Generate and upload embeddings
# ==========================================

def upload_chunks(name, records):
    index = get_index(name)

    texts = [
        record["title"] + " [SEP] " + record["chunk"]
        for record in records
    ]

    print("Generating embeddings for:", name)

    vectors = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for start in tqdm(
        range(0, len(records), BATCH_SIZE),
        desc=f"Uploading {name}"
    ):
        batch = []

        for i in range(start, min(start + BATCH_SIZE, len(records))):
            record = records[i]

            batch.append({
                "id": record["id"],
                "values": vectors[i].tolist(),
                "metadata": {
                    "arxiv_id": str(record["arxiv_id"]),
                    "title": str(record["title"])[:300],
                    "chunk": str(record["chunk"])[:3000],
                    "chunk_number": int(record["chunk_number"]),
                    "year": int(record["year"]),
                    "category": str(record["category"])
                }
            })

        index.upsert(vectors=batch)

    print("Uploaded chunks:", len(records))
    print("Index statistics:", index.describe_index_stats())


# ==========================================
# Search chunks
# ==========================================

def search_chunks(index_name, query):
    index = pc.Index(index_name)

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    results = index.query(
        vector=query_vector.tolist(),
        top_k=5,
        include_metadata=True
    )

    print("\nIndex:", index_name)
    print("Query:", query)

    for rank, match in enumerate(results.matches, 1):
        metadata = match.metadata or {}

        print(f"\n{rank}. {metadata.get('title')}")
        print("Score:", round(float(match.score), 4))
        print("Chunk:", metadata.get("chunk", "")[:200])


# ==========================================
# Run both strategies
# ==========================================

fixed_records = prepare_chunks("fixed")
semantic_records = prepare_chunks("semantic")

print("\nFixed-size chunks:", len(fixed_records))
print("Semantic chunks:", len(semantic_records))

upload_chunks(FIXED_INDEX, fixed_records)
upload_chunks(SEMANTIC_INDEX, semantic_records)

# Compare search results
test_queries = [
    "machine learning and neural networks",
    "quantum physics and particles"
]

for query in test_queries:
    search_chunks(FIXED_INDEX, query)
    search_chunks(SEMANTIC_INDEX, query)

print("\nChunking completed!")
