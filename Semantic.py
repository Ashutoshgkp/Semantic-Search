import time
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(filename='semantic_search.log', level=logging.INFO, format='%(message)s')

def load_data(file_path):
    """Load CSV data into a Pandas DataFrame."""
    return pd.read_csv(file_path)

def initialize_chroma():
    """Initialize ChromaDB client and create/load a collection."""
    client = chromadb.Client()
    collection = client.create_collection(name="semantic_search")
    return collection

def initialize_embedding_model():
    """Load the SentenceTransformer model for text embedding."""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #sentence-transformers/all-mpnet-base-v2 ////// all-MiniLM-L6-v2

def embed_text(model, text):
    """Generate embeddings for the given text."""
    return model.encode(text).tolist()

def populate_collection(collection, df, embedding_model):
    """Add data to ChromaDB collection."""
    for idx, row in df.iterrows():
        content_embedding = embed_text(embedding_model, row["Content"])
        collection.add(
            ids=[str(idx + 10000)],  # Generate a 5-digit sequential number starting from 10000
            embeddings=[content_embedding],
            metadatas=[{"title": row["Title"], "tags": row["Tags"]}]
        )

def semantic_search(collection, model, query, top_k=5, threshold=0.39):
    """Perform semantic search and measure execution time."""
    start_time = time.time()
    query_embedding = embed_text(model, query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]  # Explicitly include metadatas and distances
    )
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    log_message = f"\nQuery: {query}\nExecution Time: {execution_time:.4f} seconds\n" + "=" * 50
    print(log_message)
    logging.info(log_message)
    
    # Check if metadatas and distances are present
    if results["metadatas"] is None or results["distances"] is None:
        error_message = "Error: Metadata or distances not found in results."
        print(error_message)
        logging.error(error_message)
        return
    
    for i, (uuid, metadata, distance) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["distances"][0])):
        similarity_score = 1/(1 + distance)  # Convert distance to cosine similarity
        if similarity_score >= threshold:
            result_message = (f"Result {i+1}:\n"
                              f"UUID: {uuid}\n"
                              f"Title: {metadata['title']}\n"
                              f"Similarity Score: {similarity_score:.4f}\n"
                              + "-" * 50)
            print(result_message)
            logging.info(result_message)

def main():
    """Main function to execute the script."""
    file_path = "Full-data.csv"
    df = load_data(file_path)
    collection = initialize_chroma()
    embedding_model = initialize_embedding_model()
    populate_collection(collection, df, embedding_model)
    
    while True:
        user_query = input("Enter your search query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        semantic_search(collection, embedding_model, user_query)

if __name__ == "__main__":
    main()
