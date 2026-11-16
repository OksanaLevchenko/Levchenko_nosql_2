
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Read API key from .env
load_dotenv()

api_key = os.environ["PINECONE_API_KEY"]

# Connect to Pinecone
pc = Pinecone(api_key=api_key)

# Check connection
indexes = pc.list_indexes()

print("Connected to Pinecone successfully!")
print("Available indexes:")
print(indexes)
