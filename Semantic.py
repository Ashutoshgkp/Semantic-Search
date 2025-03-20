import time
import pandas as pd
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(filename='semantic_search.log', level=logging.INFO, format='%(message)s')

def load_data_from_sqlite(db_path, table_name):
    """Load data from an SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def initialize_chroma():
    client = chromadb.PersistentClient(path="./chromadb_data")  # Use PersistentClient for storage
    collection = client.get_or_create_collection(
        name="semantic_search",
        metadata={"hnsw:space": "cosine"}  # Explicitly set cosine similarity for ANN search
        #metadata={"hnsw:space": "cosine", "hnsw:ef_construction": 100, "hnsw:ef": 50}  # Set additional HNSW parameters
        #    "hnsw:ef_construction" = Higher values improve recall but slow indexing. "hnsw:ef" = Higher values improve accuracy but slow search.
        #     If you need high accuracy, increase "ef".
    )
    return collection

def initialize_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # Change model if needed

def embed_text(model, text):
    return model.encode(text).tolist()

def populate_collection(collection, df, embedding_model):
    """Add data to ChromaDB collection using batch processing."""
    content_embeddings = embed_text(embedding_model, df["text"].tolist())
    metadatas = [{"gender": row["gender"], "age": row["age"], "topic": row["topic"]} for _, row in df.iterrows()]
    ids = [str(idx + 10000) for idx in range(len(df))]
    
    collection.add(
        ids=ids,
        embeddings=content_embeddings,
        metadatas=metadatas
    )
def semantic_search(collection, model, query, top_k=5, threshold=0.5520):
    """Perform semantic search and measure execution time."""
    start_time = time.time()
    query_embedding = embed_text(model, query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]  
    )
    end_time = time.time()
    
    execution_time = end_time - start_time
    log_message = f"\nQuery: {query}\nExecution Time: {execution_time:.4f} seconds\n" + "=" * 50
    print(log_message)
    logging.info(log_message)
    
    if results["metadatas"] is None or results["distances"] is None:
        error_message = "Error: Metadata or distances not found in results."
        print(error_message)
        logging.error(error_message)
        return
    
    for i, (uuid, metadata, distance) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["distances"][0])):
        similarity_score = 1 / (1 + distance)  
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
    db_path = "Blog-data-large.db"
    table_name = "blog"
    
    df = load_data_from_sqlite(db_path, table_name)
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
