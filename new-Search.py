import time
import chromadb
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(filename='semantic_search.log', level=logging.INFO, format='%(message)s')

def initialize_chroma():
    """Initialize ChromaDB from persistent storage."""
    client = chromadb.PersistentClient(path="./chromadb_data")  # Load existing ChromaDB
    collection = client.get_or_create_collection(
        name="semantic_search",
        metadata={"hnsw:space": "cosine"}  # Ensure cosine similarity is used
    )
    return collection

def initialize_embedding_model():
    """Load the sentence transformer model."""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def embed_text(model, text):
    """Generate embedding for the given text."""
    return model.encode(text).tolist()

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
    
    if not results["metadatas"] or not results["distances"]:
        error_message = "Error: Metadata or distances not found in results."
        print(error_message)
        logging.error(error_message)
        return
    
    for i, (uuid, metadata, distance) in enumerate(zip(results["ids"][0], results["metadatas"][0], results["distances"][0])):
        similarity_score = 1 / (1 + distance)  
        if similarity_score >= threshold:
            result_message = (f"Result {i+1}:\n"
                              f"UUID: {uuid}\n"
                              f"Topic: {metadata.get('topic', 'N/A')}\n"
                              f"Similarity Score: {similarity_score:.4f}\n"
                              + "-" * 50)
            print(result_message)
            logging.info(result_message)

def main():
    """Main function to execute semantic search using stored ChromaDB."""
    collection = initialize_chroma()
    embedding_model = initialize_embedding_model()
    
    while True:
        user_query = input("Enter your search query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        semantic_search(collection, embedding_model, user_query)

if __name__ == "__main__":
    main()
