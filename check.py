import chromadb

def initialize_chroma():
    """Initialize ChromaDB with persistent storage."""
    client = chromadb.PersistentClient(path="./chromadb_data")
    collection = client.get_or_create_collection(name="semantic_search", metadata={"hnsw:space": "cosine"})
    return collection

def check_entries_in_chroma():
    """Check the number of entries in the ChromaDB collection."""
    collection = initialize_chroma()
    entry_count = collection.count()
    print(f"Number of entries in ChromaDB collection: {entry_count}")
    return entry_count

if __name__ == "__main__":
    check_entries_in_chroma()