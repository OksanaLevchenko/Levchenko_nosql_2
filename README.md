# NoSQL Homework 2 — Scientific Paper Search with Pinecone

## Project overview

In this project I used the Cornell arXiv metadata dataset to build a small scientific-paper search system. I prepared 10,000 paper records, generated text embeddings, uploaded them to Pinecone, compared different similarity metrics, and tested two ways of splitting long abstracts into chunks.

The project follows the required parts of the homework. The hybrid BM25 + vector search part was optional and was not implemented.

## Tools

- Python 3.12, pandas, NumPy, PyArrow, tqdm
- Pinecone (managed vector database)
- `sentence-transformers` with `allenai/specter2_base`
- Kaggle arXiv metadata dataset

## Project structure

```text
Levchenko_nosql_2/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   └── 05_chunking.py
├── requirements.txt
├── .gitignore
└── README.md
```

The downloaded `arxiv.zip`, local `data/` and `embeddings/` folders, `.venv/`, and `.env` are not committed to Git. They must be created locally to reproduce the project.

## How to run the project

1. Create a **Pinecone Starter** account and get an API key. Create a `.env` file in the project root:

   ```dotenv
   PINECONE_API_KEY=YOUR_API_KEY_HERE
   ```

2. Set up Python and install the packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Download the **Cornell University arXiv** dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). Place the downloaded archive in the project root as `arxiv.zip`. To save disk space, `01_prepare_data.py` reads the `.json` file directly from the ZIP without extracting it.

4. Run the scripts **from the project root and in this order**:

   ```bash
   python scripts/01_prepare_data.py
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python scripts/02_embed.py
   python scripts/03_load_to_pinecone.py
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python scripts/04_search.py
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python scripts/05_chunking.py
   ```

   I used `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` because my machine had an expired Hugging Face token. It allows fetching this public model without sending the old token.

5. Expected Pinecone indexes: `arxiv-papers`, `arxiv-chunks-fixed`, and `arxiv-chunks-semantic`. They use **768 dimensions** and the **cosine** metric. The index-creation scripts create indexes if necessary.

**Security:** Never commit `.env` or actual API keys. The dataset and embeddings are excluded because they are downloaded or generated locally.

---

## Part 1 — Data preparation, tools, and embeddings

### 1.1 Dataset preparation

I read the arXiv metadata JSONL file and selected the first **10,000 valid papers** with both a title and an abstract. For each paper, I saved the arXiv ID, title, abstract, authors, first-version publication year, and primary category.

The year comes from the first version's creation date, not the later update date. I saved the prepared data to `data/arxiv_subset.parquet`.

Example from the output:

```text
id: 0704.0001
title: Calculation of prompt diphoton production cross sections at Tevatron and LHC energies
authors: Balázs C., Berger E. L., Nadolsky P. M., Yuan C. -P.
year: 2007
category: hep-ph
Saved to: data/arxiv_subset.parquet
```

This subset contains early arXiv records from **2007**, which later affected the date-filtered searches. It is not a representative sample of recent research.

### 1.2 Why Pinecone instead of Qdrant or Chroma?

**Pinecone** is a managed vector database. I chose it because I wanted to focus on vectors and search without operating my own database server. Its hosted service is convenient, although it is proprietary and has usage and plan limits. It is a good fit when I want a managed service with minimal infrastructure work.

**Qdrant** has an Apache 2.0 open-source engine and can be self-hosted or used as a managed cloud service. It is a good option when I need more deployment control, metadata filtering, and a production vector search service. Self-hosting also means being responsible for monitoring, scaling, and maintenance.

**Chroma** is also open source under Apache 2.0 and supports local, persistent, server, and cloud modes. I would choose it for a small prototype or experiments because the Python integration is simple. For a large production workload, I would benchmark its server/cloud deployment and operational needs rather than assuming it will be faster or slower than the others.

Performance depends on data size, indexing configuration, filters, hardware, and deployment. I did not run a performance benchmark between these three databases, so I cannot claim one is always fastest.

Sources: [Pinecone pricing](https://www.pinecone.io/pricing/), [Qdrant repository](https://github.com/qdrant/qdrant), [Chroma repository](https://github.com/chroma-core/chroma).

### Why SPECTER2 instead of all-MiniLM-L6-v2?

I chose `allenai/specter2_base` because it was developed specifically for **scientific documents**. According to the [Hugging Face model card](https://huggingface.co/allenai/specter2_base), its base model is trained using citation relationships between research papers. SPECTER2 also has task-specific adapters for classification, regression, retrieval/proximity, and ad-hoc search. This makes it a more domain-focused starting point than the general-purpose `all-MiniLM-L6-v2` model.

In my code, `SentenceTransformer("allenai/specter2_base")` displayed a message saying it created a **mean-pooling fallback**. I therefore used the base model with mean pooling, **not** a task-specific SPECTER2 retrieval adapter. This is an important limitation when judging search quality. The model card recommends using an appropriate adapter for best task-specific performance.

### Model-card similarity metric and chosen index metric

The base-model card explains the training and adapter setup, but **does not explicitly prescribe a single Pinecone similarity metric**. I chose **cosine** because I normalized the vectors to unit length, and cosine measures their directional similarity. This choice must agree with the way vectors are generated and compared; different metrics can change rankings for unnormalized vectors.

### 1.3 Embedding generation

For each paper I combined the fields in the required format:

```text
title + " [SEP] " + abstract
```

I encoded the texts in batches of 16, showed a progress bar, and set `normalize_embeddings=True`.

Verified terminal results:

```text
Loaded 10000 articles
Generating embeddings...
Batches: 100% | 625/625
Total embeddings: 10000
Embedding dimensions: 768
First embedding norm: 1.0
Embeddings saved to: embeddings/embeddings.npy
Embedding generation completed!
```

**Why does cosine similarity equal dot product for normalized vectors?**

Cosine similarity is `a·b / (||a|| ||b||)`. If both vectors have length one, the denominator is one, so cosine similarity is exactly `a·b`. Normalization makes comparison more consistent by removing vector-length differences.

---

## Part 2 — Uploading vectors to Pinecone

I created a serverless index called `arxiv-papers` using **768 dimensions** and **cosine similarity**. I uploaded the embeddings in batches of 200. Each vector has an ID such as `paper_0` and metadata: `arxiv_id`, `title`, `abstract` (up to 500 characters), `authors` (up to 200 characters), `year`, and `category`.

The full abstract remains in the local Parquet file. Shortening the abstract in metadata reduces upload/storage size and helps stay within Pinecone metadata limits. It also means full abstracts must be retrieved from the local dataset if needed.

Verified index statistics:

```text
Index: arxiv-papers
Dimension: 768
Namespace: default
Total vectors: 10000
```

---

## Part 3 — Search queries and similarity metrics

### Task 1: Semantic search

I searched for:

```text
teaching machines to recognize objects in pictures
```

One top result in the local comparison was **“Capturing knots in polymers”** (cosine score about **0.8294**). Other results included **“Symbolic sensors: one solution to the numerical-symbolic interface”** and **“The Mathematics.”** Some of these results are not closely related to image recognition, even though they receive high similarity scores. My base-model fallback may not be ideal for short search queries; I would test a task-specific ad-hoc search adapter in a future improvement.

### Task 2: Search with filters

I searched for reinforcement learning with a `cs.LG` category filter and a recent publication-year filter (2022 onward). The result was:

```text
TASK 2A: RECENT REINFORCEMENT LEARNING
No matching articles found.
```

This makes sense because I took the **first 10,000 records from 2007**, not a sample of newer research. It does not prove Pinecone filtering is broken.

For older papers (before 2015), the query returned results including:

| Rank | Paper | Category | Year | Score |
|---|---|---|---:|---:|
| 1 | Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa | cs.MA | 2007 | 0.8682 |
| 2 | Introduction to Phase Transitions in Random Optimization Problems | cond-mat.stat-mech | 2007 | 0.8359 |
| 3 | Architecture for Pseudo Acausal Evolvable Embedded Systems | cs.NE | 2007 | 0.8237 |
| 4 | A Study in a Hybrid Centralised-Swarm Agent Community | cs.NE | 2007 | 0.8200 |
| 5 | Why only few are so successful? | physics.pop-ph | 2007 | 0.8169 |

The filtered results depend on both the query text and metadata conditions. A recent-paper comparison would require a dataset sampled across multiple years.

### Task 3: Local metric comparison

For the image-recognition query, the top five papers were **in the same order** for cosine, dot product, and L2 distance:

| Rank | Paper | Cosine | Dot product | L2 distance |
|---|---|---:|---:|---:|
| 1 | Capturing knots in polymers | 0.8294 | 0.8294 | 0.5842 |
| 2 | Symbolic sensors: one solution to the numerical-symbolic interface | 0.8260 | 0.8260 | 0.5899 |
| 3 | The Mathematics | 0.8254 | 0.8254 | 0.5910 |
| 4 | Modeling the field of laser welding melt pool by RBFNN | 0.8181 | 0.8181 | 0.6032 |
| 5 | Python for Education: Computational Methods for Nonlinear Systems | 0.8142 | 0.8142 | 0.6095 |

```text
Cosine vs Dot same ranking: True
Cosine vs L2 same ranking: True
```

**Why did cosine and dot agree?** All vectors were normalized, so cosine is the dot product.

**Why did L2 also agree?** For unit vectors, `||a-b||² = 2 - 2(a·b)`. Smaller L2 distances therefore correspond to larger cosine similarities. These are different score scales but the same ordering (except possible numerical ties).

**What if vectors were not normalized?** The dot product would depend on both direction and magnitude. L2 would also include length differences, so the three rankings could differ. Normalizing both document and query vectors avoids this issue.

---

## Part 4 — Fixed-size and sentence-grouped chunking

I selected the **30 longest abstracts** and tested two strategies:

- **Fixed-size:** 120 words per chunk with a 25-word overlap.
- **Sentence-grouped (semantic):** build chunks from complete sentences, keeping chunks near a 120-word maximum. Very long single sentences may still be split. This is a simple sentence-boundary method, not an embedding-based semantic-boundary algorithm.

I embedded each chunk with the same model and uploaded the results to separate Pinecone indexes.

Verified vector counts:

| Index | Vectors |
|---|---:|
| `arxiv-chunks-fixed` | **94** |
| `arxiv-chunks-semantic` | **92** |

I tested the query **“quantum physics and particles.”** Both strategies ranked *Spin Effects in Quantum Chromodynamics and Recurrence Lattices with Multi-Site Exchanges* first. The top scores were **0.8400** for fixed-size and **0.8398** for sentence-grouped chunks. Both also returned *Conjectures on exact solution of three-dimensional simple orthorhombic Ising lattices*.

**Which chunks are more meaningful?** Sentence-grouped chunks usually read better because they try to preserve complete sentences. The fixed-size method is simpler and gives more predictable chunk lengths. In my small test, neither method clearly performed better in relevance: the best matches and scores were very close.

**Are sentences cut?** Fixed-size chunks can start or end in the middle of a sentence. That may remove context and reduce the usefulness of an embedding. The sentence-grouped approach reduces this problem, although exceptionally long sentences still need to be split.

**How does overlap affect the result?** Overlap repeats words between neighboring fixed chunks so important context around a boundary is less likely to be lost. More overlap gives more chunks, greater storage and embedding cost, and more duplicate results. Too little overlap risks missing context.

---

## Part 5 — Optional hybrid search

Not implemented. This part was marked optional and ungraded in the final grading criteria provided for this assignment.

---

## Part 6 — Analysis and conclusions

### 1. Semantic search versus keyword search

Semantic search is useful when the user describes an idea in different words from the paper's title or abstract. It can connect related concepts without exact keyword matches. Keyword search such as BM25 can be better for exact author names, rare technical terms, formulas, identifiers, and acronyms. In my results, some semantically similar articles were not truly relevant to image recognition. My rule would be: start with semantic search for meaning, use keywords when exact words matter, and consider hybrid search if both types of relevance are important. I did not implement BM25 in this homework, so this is a reasoned comparison rather than a measured BM25 benchmark.

### 2. Impact of chunk size

A very short chunk of 10–15 words may lack the context needed to understand the scientific idea. It can match individual phrases but lose the methods or conclusions around them. A very large chunk of more than 500 words may cover several topics and dilute the relevance of any one section; it may also exceed the model's token limit and be truncated. There is no single best word count for every dataset. I used 120 words as a practical starting point. A better choice would be tested against real search queries, relevant-answer judgments, and the model's token limit.

### 3. What if Pinecone used Euclidean distance for normalized vectors?

For vectors `a` and `b`, squared Euclidean distance is:

```text
||a-b||² = ||a||² + ||b||² - 2(a·b)
```

When both are unit vectors, `||a|| = ||b|| = 1`, so:

```text
||a-b||² = 2 - 2(a·b) = 2 - 2 cosine(a,b)
```

Therefore minimizing Euclidean distance is equivalent to maximizing cosine similarity for these normalized vectors. The **rankings would be the same in exact arithmetic**, although Pinecone scores would have different meanings and approximate search or floating-point effects could produce small differences. Euclidean distance is not automatically a bad choice here, but using cosine keeps the scores easier to interpret as cosine similarities.

### 4. Pinecone Starter limits and a larger dataset

I successfully created **three** indexes and stored **10,186** vectors in total (10,000 papers + 94 fixed chunks + 92 sentence-grouped chunks). I did not encounter an index-count limit during this run. Pinecone's current published Starter plan includes limits on index count, storage, reads, writes, and other usage; these can change, so I would check the dashboard and pricing documentation before scaling.

For **10 million papers**, I would not assume the free tier is enough. I would estimate the raw vector storage first: `10,000,000 × 768 × 4 bytes ≈ 30.7 GB` for float32 values alone, before metadata, indexing overhead, or extra chunk vectors. I would also consider embedding computation cost, batch ingestion, monitoring, retrieval quality, and a paid/scalable deployment. I could reduce metadata stored with each vector and keep complete abstracts in separate object storage, looking them up by arXiv ID.

---

## Terminal output evidence

The homework requests terminal output or screenshots for each required script. The verified excerpts currently recorded are below. **Before submission, add terminal screenshots or the remaining actual output for scripts 01, 03, 04, and 05 where useful; do not invent missing lines.**

### Script 01 — Prepare data

```text
Example article: {'id': '0704.0001', ..., 'year': 2007, 'category': 'hep-ph'}
Saved to: data/arxiv_subset.parquet
```

### Script 02 — Embeddings

```text
Loaded 10000 articles
Batches: 100% | 625/625
Total embeddings: 10000
Embedding dimensions: 768
First embedding norm: 1.0
Embeddings saved to: embeddings/embeddings.npy
Embedding generation completed!
```

### Script 03 — Pinecone upload

```text
Index: arxiv-papers
Total vectors: 10000
Dimension: 768
```

### Script 04 — Semantic search

```text
TASK 2A: RECENT REINFORCEMENT LEARNING
No matching articles found.
Cosine vs Dot same ranking: True
Cosine vs L2 same ranking: True
```

### Script 05 — Chunking

```text
arxiv-papers : 10000 vectors
arxiv-chunks-fixed : 94 vectors
arxiv-chunks-semantic : 92 vectors
Chunking completed!
```

## References

- [Cornell University arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [SPECTER2 base model card](https://huggingface.co/allenai/specter2_base)
- [SPECTER2 repository and adapter guidance](https://github.com/allenai/SPECTER2)
- [Pinecone pricing and Starter plan](https://www.pinecone.io/pricing/)
- [Qdrant open-source repository](https://github.com/qdrant/qdrant)
- [Chroma open-source repository](https://github.com/chroma-core/chroma)

## Final reflection

This homework helped me understand how to turn scientific texts into vectors, store them in Pinecone, and retrieve articles by meaning rather than exact words. I also learned that the dataset sample and model setup matter as much as the search code. The first 10,000 papers were all from 2007, so filtering for recent work produced no results. With normalized vectors, cosine, dot product, and L2 returned the same ranking. Comparing the two chunking approaches showed similar results in a small test, so I would need more queries and relevance checks before claiming either approach is better.
