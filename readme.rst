Semantic Search Project
=======================

Overview
--------
This project implements a semantic search system using ChromaDB and SentenceTransformers. The system allows users to perform semantic searches on a dataset and retrieve the most relevant results based on the similarity of the text content.

Files
-----
- `Semantic.py`: The main script that contains the implementation of the semantic search system.

Dependencies
------------
- `pandas`
- `chromadb`
- `sentence-transformers`
- `logging`

Installation
------------
1. Clone the repository.
2. Install the required dependencies using pip:
   ```
   pip install pandas chromadb sentence-transformers
   ```

Usage
-----
1. Prepare your dataset in a CSV file named `Full-data.csv` with the following columns:
   - `UUID`: Unique identifier for each entry.
   - `Title`: Title of the content.
   - `Content`: The text content to be indexed and searched.
   - `Tags`: Tags associated with the content.

2. Run the `Semantic.py` script:
   ```
   python Semantic.py
   ```

3. Enter your search query when prompted. Type `exit` to quit the program.

Logging
-------
The script logs the search queries and results to a file named `semantic_search.log`.

Functions
---------
- `load_data(file_path)`: Loads CSV data into a Pandas DataFrame.
- `initialize_chroma()`: Initializes the ChromaDB client and creates/loads a collection.
- `initialize_embedding_model()`: Loads the SentenceTransformer model for text embedding.
- `embed_text(model, text)`: Generates embeddings for the given text.
- `populate_collection(collection, df, embedding_model)`: Adds data to the ChromaDB collection.
- `semantic_search(collection, model, query, top_k=5, threshold=0.39)`: Performs semantic search and measures execution time.

Customization
-------------
- You can change the embedding model by modifying the `initialize_embedding_model` function.
- Adjust the similarity score threshold in the `semantic_search` function to filter results based on your requirements.

