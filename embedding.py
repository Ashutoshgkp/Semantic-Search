import sqlite3
import chromadb
import logging
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DB_PATH = "Blog-data-large.db"
TABLE_NAME = "blog"
BATCH_SIZE = 200  
CHROMA_DB_PATH = "./chromadb_data"
MAX_WORKERS = 6  # Half of your logical processors

def initialize_chroma():
    """Initialize ChromaDB with persistent storage."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name="semantic_search", metadata={"hnsw:space": "cosine"})
    return collection


def initialize_embedding_model():
    """Load the sentence transformer model."""
    logging.info("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logging.info("Model loaded successfully.")
    return model


def get_existing_ids(collection):
    """Retrieve already indexed IDs from ChromaDB."""
    logging.info("Fetching existing indexed IDs from ChromaDB...")
    results = collection.get(include=["metadatas"])
    existing_ids = set(results["ids"]) if results and "ids" in results else set()
    logging.info(f"Found {len(existing_ids)} existing indexed entries.")
    return existing_ids


def fetch_new_data(existing_ids, offset, batch_size):
    """Fetch a chunk of new (not yet indexed) data from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, text, gender, age, topic FROM {TABLE_NAME} LIMIT {batch_size} OFFSET {offset}")
    rows = [row for row in cursor.fetchall() if str(row[0]) not in existing_ids]
    conn.close()
    logging.info(f"Fetched {len(rows)} new records for offset {offset}.")
    return rows


def embed_and_store(data_chunk, model):
    if not data_chunk:
        logging.info("No new data to process in this batch.")
        return

    collection = initialize_chroma()
    ids, texts, metadatas = [], [], []

    for row in data_chunk:
        doc_id, text, gender, age, topic = row
        ids.append(str(doc_id))
        texts.append(text)
        metadatas.append({"gender": gender, "age": age, "topic": topic})

    logging.info(f"Embedding {len(texts)} documents...")
    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    logging.info(f"Successfully stored {len(ids)} embeddings into ChromaDB.")


def main():
    """Main function to process only new data efficiently."""
    collection = initialize_chroma()
    existing_ids = get_existing_ids(collection)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total_rows = cursor.fetchone()[0]
    conn.close()

    logging.info(f"Total records in database: {total_rows}")

    model = initialize_embedding_model()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        offsets = range(0, total_rows, BATCH_SIZE)
        futures = [executor.submit(embed_and_store, fetch_new_data(existing_ids, offset, BATCH_SIZE), model) for offset in offsets]

        for future in futures:
            future.result()  # Ensure each task completes

    logging.info("Embedding process completed.")


if __name__ == "__main__":
    main()
