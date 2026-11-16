
import json
import os
import zipfile

import pandas as pd
from tqdm import tqdm

# Files and settings
INPUT_ZIP = "arxiv.zip"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10000

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)


def extract_year(paper):
    # Get publication year from the first version
    try:
        versions = paper.get("versions", [])
        if versions:
            return int(versions[0]["created"].split()[3])
    except (IndexError, KeyError, ValueError, TypeError):
        pass

    # Use update date if version is missing
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper):
    # Format author names
    parsed = paper.get("authors_parsed", [])

    if parsed:
        names = []

        for author in parsed[:10]:
            last = author[0].strip() if len(author) > 0 else ""
            first = author[1].strip() if len(author) > 1 else ""

            if last:
                names.append(f"{last} {first}".strip())

        return ", ".join(names)

    return paper.get("authors", "").replace("\n", " ")


records = []

# Read JSONL directly from ZIP without extracting it
with zipfile.ZipFile(INPUT_ZIP, "r") as archive:

    json_file = next(
        name for name in archive.namelist()
        if name.endswith(".json")
    )

    with archive.open(json_file) as file:

        for line in tqdm(file, desc="Reading arXiv dataset"):

            if len(records) >= MAX_RECORDS:
                break

            paper = json.loads(line)

            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()

            # Skip records without title or abstract
            if not title or not abstract:
                continue

            categories = paper.get("categories", "unknown")
            category = categories.split()[0] if categories.split() else "unknown"

            records.append({
                "id": paper["id"],
                "title": title.replace("\n", " "),
                "abstract": abstract.replace("\n", " "),
                "authors": format_authors(paper),
                "year": extract_year(paper),
                "category": category
            })


# Convert to DataFrame
df = pd.DataFrame(records)

print("\nTotal articles:", len(df))

print("\nTop 10 categories:")
print(df["category"].value_counts().head(10))

print("\nArticles by year:")
print(df["year"].value_counts().sort_index().tail(10))

print("\nExample article:")
print(df.iloc[0].to_dict())

# Save as Parquet
df.to_parquet(OUTPUT_FILE, index=False)

print("\nSaved to:", OUTPUT_FILE)
