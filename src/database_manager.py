import sqlite3
import json

DB_PATH = "database/people.db"


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        embedding TEXT
    )
    """)

    conn.commit()
    conn.close()


def add_person(name, age):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO persons(name, age) VALUES (?, ?)",
        (name, age)
    )

    person_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return person_id


def save_embedding(person_id, embedding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO embeddings(person_id, embedding) VALUES (?, ?)",
        (person_id, json.dumps(embedding.tolist()))
    )

    conn.commit()
    conn.close()


def get_all_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT persons.id,
           persons.name,
           persons.age,
           embeddings.embedding
    FROM persons
    JOIN embeddings
    ON persons.id = embeddings.person_id
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows