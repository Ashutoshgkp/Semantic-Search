Semantic Search Project
=======================

Overview
--------
This project implements a semantic search system using ChromaDB and SentenceTransformers. The system allows users to perform semantic searches on a dataset and retrieve the most relevant results based on the similarity of the text content.

Files
-----
- `Embedding.py`: Contains the code to embed SQLite data to vector data in ChromaDB.
- `new_search.py`: Uses the saved ChromaDB vectors to perform searches.
- `csv-sqlite.py`: Converts the CSV file to SQLite format.
- `tag-generating.py`: Uses spaCy and KeyBERT to generate tags.
- `Semantic.py`: An earlier implementation of the semantic search system.

Dependencies
------------
- `pandas`
- `chromadb`
- `sentence-transformers`
- `logging`
- `sqlite3`
- `spacy` (Optional - for tag generation)
- `keybert` (Optional - for tag generation)

Installation
------------
1. Clone the repository.
2. Install and launch a virtual environment.
3. Install the required dependencies using pip:
   ```
   pip install pandas chromadb sentence-transformers
   ```

Usage
-----
1. Prepare your dataset in a CSV file containing 680,000+ records with the following columns:
   - `age`: Age of the individual.
   - `topic`: Topic of the content.
   - `text`: The text content to be indexed and searched.
   - The UUID is generated in the code.

2. Convert the CSV file to SQLite format using `csv-sqlite.py`:
   ```
   python csv-sqlite.py
   ```

3. Embed the SQLite data to vector data in ChromaDB using `Embedding.py`:
   ```
   python Embedding.py
   ```

4. Perform searches using `new_search.py`:
   ```
   python new_search.py
   ```

Logging
-------
The scripts log the search queries and results to a file named `semantic_search.log`.

Functions
---------
Embedding.py
~~~~~~~~~~~~
- `load_data_from_sqlite(db_path, table_name)`: Loads data from an SQLite database.
- `initialize_chroma()`: Initializes the ChromaDB client and creates/loads a collection.
- `initialize_embedding_model()`: Loads the SentenceTransformer model for text embedding.
- `embed_text(model, text)`: Generates embeddings for the given text.
- `populate_collection(collection, df, embedding_model)`: Adds data to the ChromaDB collection using batch processing.

new_search.py
~~~~~~~~~~~~~
- `semantic_search(collection, model, query, top_k=5, threshold=0.5520)`: Performs semantic search and measures execution time.

Customization
-------------
- You can change the embedding model by modifying the `initialize_embedding_model` function in `Embedding.py`.
- Adjust the similarity score threshold in the `semantic_search` function in `new_search.py` to filter results based on your requirements.

