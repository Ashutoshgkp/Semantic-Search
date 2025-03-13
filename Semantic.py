import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv("Full-data.csv") 

# Initialize Chroma DB client
chroma_client = chromadb.Client()

# Create or load a collection
collection = chroma_client.create_collection(name="semantic_search")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to embed text
def embed_text(text):
    return embedding_model.encode(text).tolist()

# Add data to Chroma DB
for idx, row in df.iterrows():
    content_embedding = embed_text(row["Content"])  # Embed the content
    collection.add(
        ids=[str(row["UUID"])],  # Use UUID as the unique ID
        embeddings=[content_embedding],  # Embedding vector
        metadatas=[{"title": row["Title"], "tags": row["Tags"]}]  
    )

# Perform semantic search
def semantic_search(query, top_k=5):
    # Embed the query
    query_embedding = embed_text(query)
    
    # Search the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Display results (UUID, Title, and Tags)
    for i, (uuid, metadata) in enumerate(zip(results["ids"][0], results["metadatas"][0])):
        print(f"Result {i+1}:")
        print(f"UUID: {uuid}")
        print(f"Title: {metadata['title']}")
        #print(f"Tags: {metadata['tags']}")
        print("-" * 50)


user_query = input("Enter your search query: ")

semantic_search(user_query)