import sqlite3
import csv
import sys

# Increase CSV field size limit safely
csv.field_size_limit(2**30)  # Use 1GB instead of sys.maxsize

def import_csv_to_sqlite(csv_file, db_file, table_name, batch_size=10000):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    with open(csv_file, mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get header row

        # Create table dynamically
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(headers)})")

        # Prepare for batch insert
        placeholders = ', '.join(['?'] * len(headers))
        rows = []

        cursor.execute("BEGIN TRANSACTION")  # Start transaction

        for row in csv_reader:
            rows.append(row)
            if len(rows) >= batch_size:  # Insert in batches
                cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)
                rows = []  # Clear batch

        if rows:  # Insert remaining rows
            cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)

        cursor.execute("COMMIT")  # Commit transaction

    conn.commit()
    conn.close()

# Usage example:
csv_file = 'blogtext.csv'  # Path to your CSV file
db_file = 'Blog-data-large.db'  # SQLite database
table_name = 'blog'  # Table name

# Import CSV to SQLite
import_csv_to_sqlite(csv_file, db_file, table_name)

print("CSV data successfully imported into SQLite database!")
