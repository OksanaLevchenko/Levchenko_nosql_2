# NoSQL Homework 2 — Scientific Paper Search with Pinecone

## What I did

For this homework I made a small search project using scientific articles from arXiv. I worked with 10,000 papers, made embeddings for their titles and abstracts, uploaded them to Pinecone, and tried different ways to search. I also split long abstracts into smaller chunks and compared the results.

I completed Parts 1–4 and 6. I skipped Part 5 (hybrid search) because it was marked as optional in the assignment.

I used Python, pandas, NumPy, PyArrow, SentenceTransformers, the `allenai/specter2_base` model, and Pinecone Starter.

## Files and how to run

```text
Levchenko_nosql_2/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── test_pinecone.py
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset, `.env`, `data/`, `embeddings/`, and `.venv/` are local files and are ignored by Git. I did not upload my Pinecone API key or the large dataset to GitHub.

To run the project from the beginning:

1. Create a free [Pinecone account](https://www.pinecone.io/) and get an API key. Make a `.env` file in the project folder:

   ```dotenv
   PINECONE_API_KEY=YOUR_API_KEY_HERE
   ```

2. Create the Python environment and install the packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Download the [Cornell University arXiv metadata dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv). Put `arxiv.zip` in the project folder. My first script reads the JSON file **inside the ZIP**, so you don't need to unzip it. I did this because my Mac was almost out of storage.

4. Run the scripts in this order, from the project folder:

   ```bash
   python scripts/01_prepare_data.py
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python scripts/02_embed.py
   python scripts/03_load_to_pinecone.py
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python scripts/04_search.py
   HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python scripts/05_chunking.py
   ```

I added `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` because I got an error from an expired Hugging Face token on my Mac. It allowed me to load the public model. The Pinecone scripts create the indexes if needed. They use 768 dimensions and the `cosine` metric.

## Part 1 — Data and embeddings

### 1.1 Preparing the papers

I read the first 10,000 valid records from arXiv. I skipped articles without a title or abstract. For each paper I kept its ID, title, abstract, authors, publication year, and main category. I used the date of the first arXiv version for the year, not the last update date.

The file created by the script is `data/arxiv_subset.parquet`.

One example from the output was:

```text
id: 0704.0001
title: Calculation of prompt diphoton production cross sections at Tevatron and LHC energies
authors: Balázs C., Berger E. L., Nadolsky P. M., Yuan C. -P.
year: 2007
category: hep-ph
Saved to: data/arxiv_subset.parquet
```

Later I found out that the first 10,000 papers are from 2007. That affected one of my search filters in Part 3.

### 1.2 Why I used Pinecone instead of Qdrant or Chroma

**Pinecone:** It runs as a managed cloud service. I didn't need to install or manage a vector database server, so it was the easiest choice for this homework. It is proprietary, and I have to work within the limits of the plan. I would use it when I want a hosted service and don't want to manage the servers myself.

**Qdrant:** Its database engine is open source (Apache 2.0). I could run it on my own machine or server, or use its hosted cloud service. I would choose it when I need more control over how the database is deployed. But if I host it myself, I also need to maintain it.

**Chroma:** This is also open source (Apache 2.0). It is easy to start with locally in Python, and there are server/cloud options too. I would use it for a small prototype or for quickly trying an embedding idea.

I didn't run performance benchmarks between these three databases, so I can't say one is always faster. Speed depends on things like vector count, hardware, filtering, and index setup.

Sources: [Pinecone](https://www.pinecone.io/pricing/), [Qdrant](https://github.com/qdrant/qdrant), [Chroma](https://github.com/chroma-core/chroma).

### Why SPECTER2 and not all-MiniLM-L6-v2?

I used `allenai/specter2_base` because this model was made for scientific papers. Its [Hugging Face model card](https://huggingface.co/allenai/specter2_base) describes training using citation links between scientific papers. The SPECTER2 project also provides adapters for different tasks, including classification and retrieval. `all-MiniLM-L6-v2` is a general-purpose embedding model. For this homework, I wanted a model connected to scientific text.

When I loaded it with SentenceTransformers, I saw:

```text
No sentence-transformers model found with name allenai/specter2_base.
Creating a new one with mean pooling.
```

So I used the base model with mean pooling, **not** a task-specific retrieval adapter. I think that may be one reason some search results later looked strange.

### What about the similarity metric?

I used `cosine` for the Pinecone indexes. The base model card does not give me one single required index metric for every adapter/task. In my code I normalized embeddings, so cosine compares vector direction rather than length. The metric matters because Pinecone uses it to decide which papers should appear first.

### 1.3 Creating the embeddings

I combined each title and abstract like this:

```text
title + " [SEP] " + abstract
```

I used batches of 16 on my Mac and set `normalize_embeddings=True`. I saved the vectors to `embeddings/embeddings.npy`.

My output:

```text
Loaded 10000 articles
Batches: 100% | 625/625
Total embeddings: 10000
Embedding dimensions: 768
First embedding norm: 1.0
Embeddings saved to: embeddings/embeddings.npy
Embedding generation completed!
```

**Why are cosine similarity and dot product equal here?** Cosine similarity divides the dot product by the lengths of the two vectors. My vectors are normalized, so each length is 1. Dividing by `1 × 1` doesn't change the value. That is why cosine and dot product should rank these vectors in the same order.

## Part 2 — Uploading to Pinecone

I created the index `arxiv-papers` with **768 dimensions** and **cosine** similarity. I uploaded the embeddings in batches of 200.

Each vector had an ID like `paper_0` and metadata: arXiv ID, title, first 500 characters of the abstract, first 200 characters of the authors, year, and category. I kept the metadata short to avoid storing too much text in Pinecone. The full abstracts are in the Parquet file.

I checked the index after uploading:

```text
Index: arxiv-papers
Dimension: 768
Total vectors: 10000
```

The count was correct: **10,000 vectors**.

## Part 3 — Searching for papers

### Semantic search

I searched for:

```text
teaching machines to recognize objects in pictures
```

Some of the results were *Capturing knots in polymers*, *Symbolic sensors: one solution to the numerical-symbolic interface*, and *The Mathematics*. For the first paper, the local cosine score was about **0.8294**.

I expected more papers about image recognition. The script worked, but I would not call all these results good matches. My model setup used mean pooling without the retrieval adapter, and the dataset was also a limited older sample.

### Search with filters

I tried a search for reinforcement learning in category `cs.LG` from **2022 onward**. It returned:

```text
TASK 2A: RECENT REINFORCEMENT LEARNING
No matching articles found.
```

This makes sense because the first 10,000 papers I selected were from 2007. The filter cannot return articles that are not in the index.

I also searched for papers before 2015. That search did return results:

| # | Paper | Year | Score |
|---|---|---:|---:|
| 1 | Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa | 2007 | 0.8682 |
| 2 | Introduction to Phase Transitions in Random Optimization Problems | 2007 | 0.8359 |
| 3 | Architecture for Pseudo Acausal Evolvable Embedded Systems | 2007 | 0.8237 |
| 4 | A Study in a Hybrid Centralised-Swarm Agent Community | 2007 | 0.8200 |
| 5 | Why only few are so successful? | 2007 | 0.8169 |

The old-paper filter works, but I can't really compare old and recent papers using this subset.

### Cosine, dot product, and L2 distance

I compared the three metrics locally using the same query and saved embeddings:

| # | Paper | Cosine | Dot product | L2 distance |
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

**Did cosine and dot give the same top 5?** Yes. All the vectors were normalized, so cosine and dot product measure the same thing here.

**Was L2 different?** The numbers were different, but the ranking was the same. For unit-length vectors, a higher cosine similarity means a shorter L2 distance. For L2, the smallest number is the best result.

**What if I did not normalize?** The vector lengths could affect dot product and L2, and the ranking could change. Cosine still compares vector direction for nonzero vectors.

## Part 4 — Chunking

I took the 30 papers with the longest abstracts and tried two ways of splitting them:

- **Fixed-size:** 120 words per chunk, with 25 words repeated from the previous chunk.
- **Sentence-grouped:** keep whole sentences together until the chunk is about 120 words. Very long sentences sometimes still need to be split.

My second method keeps sentence boundaries, but it is a simple sentence-grouping approach, not a special embedding-based semantic chunker.

I uploaded the chunks to two Pinecone indexes. These were the counts I checked:

| Index | Vectors |
|---|---:|
| `arxiv-chunks-fixed` | 94 |
| `arxiv-chunks-semantic` | 92 |

For **“quantum physics and particles”**, both methods returned *Spin Effects in Quantum Chromodynamics and Recurrence Lattices with Multi-Site Exchanges* first. The top scores were **0.8400** for fixed-size and **0.8398** for sentence-grouped.

**Which chunks make more sense?** I prefer sentence-grouped chunks for reading because they usually keep the full sentence. But in this small search test, it did not clearly improve the top score or result.

**Can a sentence get cut?** Yes. Fixed-size chunks can cut a sentence in the middle. Then part of the meaning may be in the next chunk. Sentence grouping avoids most of that, except when one sentence is already too long.

**What does overlap change?** Overlap helps keep information around the border of two chunks. More overlap can help with context, but it also creates more chunks and uses more storage. Similar chunks may also show up repeatedly in search results.

## Part 5 — Optional hybrid search

I didn't implement BM25 + vector hybrid search because this part was optional in the final grading table.

## Part 6 — My answers and conclusions

### 1. Semantic search vs keyword search

Semantic search is useful when I know what I want to find but don't know the exact words in a paper. It tries to match meaning, not just the same keywords. In my test, it also returned some papers that didn't look relevant, so it is not perfect. For exact author names, IDs, or specific technical terms, I would choose keyword search like BM25. For general topics, I would start with semantic search. I didn't run BM25 in this homework, so this is a reasoned comparison, not a benchmark.

### 2. Does chunk size matter?

I think it does. A chunk of only **10–15 words** might be missing the context needed to understand an idea. A chunk with **500+ words** may mix different topics, and some of it could be cut off by the model's token limit. I used 120 words as a practical starting point, but the best size depends on the texts and the questions. I would test more than one size in a bigger project.

### 3. What if I used Euclidean distance instead of cosine?

The formula for squared L2 distance is:

```text
||a - b||² = ||a||² + ||b||² - 2(a · b)
```

If both vectors have length 1, then `||a||² = ||b||² = 1`, and cosine similarity is equal to the dot product. So:

```text
||a - b||² = 2 - 2 cosine(a, b)
```

When cosine similarity gets larger, L2 distance gets smaller. This means **normalized vectors have the same ranking with cosine and L2 in exact math**, even though the score values are different. Pinecone could still show small differences because of approximate search or rounding. Euclidean would not automatically be wrong for my vectors, but I used cosine consistently in this homework.

### 4. Pinecone Starter limits and 10 million papers

I didn't hit an index limit: I created all three indexes and checked the vector counts.

```text
arxiv-papers: 10000
arxiv-chunks-fixed: 94
arxiv-chunks-semantic: 92
Total: 10186 vectors
```

For a real project, I would check the current Starter limits and costs before uploading more data. With **10 million papers**, I would need more storage and processing resources. The raw 768-dimensional float32 embeddings alone would take about **30.7 GB** (`10,000,000 × 768 × 4 bytes`), before metadata or database overhead. I would use batch uploads, store long abstracts separately, and monitor costs. I would also need a suitable paid plan or another way to deploy a vector database.

## Results saved from my terminal

These are the results I recorded while running the scripts. The homework also asks for terminal evidence for each step. I should include screenshots of the full outputs if the mentor needs more detail.

**01 — Data preparation**

```text
Example article: {'id': '0704.0001', ..., 'year': 2007, 'category': 'hep-ph'}
Saved to: data/arxiv_subset.parquet
```

**02 — Embeddings**

```text
Loaded 10000 articles
Batches: 100% | 625/625
Total embeddings: 10000
Embedding dimensions: 768
First embedding norm: 1.0
Embeddings saved to: embeddings/embeddings.npy
Embedding generation completed!
```

**03 — Pinecone index check**

```text
Index: arxiv-papers
Dimension: 768
Total vectors: 10000
```

**04 — Search**

```text
TASK 2A: RECENT REINFORCEMENT LEARNING
No matching articles found.
Cosine vs Dot same ranking: True
Cosine vs L2 same ranking: True
```

**05 — Chunking and Pinecone index check**

```text
arxiv-papers : 10000 vectors
arxiv-chunks-fixed : 94 vectors
arxiv-chunks-semantic : 92 vectors
Chunking completed!
```

## References

- [Cornell arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [SPECTER2 base model card](https://huggingface.co/allenai/specter2_base)
- [SPECTER2 documentation](https://github.com/allenai/SPECTER2)
- [Pinecone plans](https://www.pinecone.io/pricing/)
- [Qdrant](https://github.com/qdrant/qdrant)
- [Chroma](https://github.com/chroma-core/chroma)
