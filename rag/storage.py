import os
import sqlite3
import pickle
import hashlib
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

UPLOAD_DIR = os.path.join(
    DATA_DIR,
    "uploads"
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

DATABASE_PATH = os.path.join(
    DATA_DIR,
    "database.db"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT NOT NULL,

            filename TEXT NOT NULL,

            file_hash TEXT NOT NULL,

            file_path TEXT NOT NULL,

            processed_path TEXT NOT NULL,

            chunks_count INTEGER DEFAULT 0,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(username, file_hash)
        )
        """
    )


    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT NOT NULL,

            question TEXT NOT NULL,

            answer TEXT,

            created_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            questions_asked INTEGER DEFAULT 0,

            correct_answers INTEGER DEFAULT 0,

            weak_topic TEXT,

            updated_at TIMESTAMP
            DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    connection.commit()

    connection.close()


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# CHECK DOCUMENT
# ============================================================

def document_exists(
    username,
    file_hash
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM documents
        WHERE username = ?
        AND file_hash = ?
        """,
        (
            username,
            file_hash
        )
    )


    result = cursor.fetchone()

    connection.close()

    return result


# ============================================================
# SAVE PDF
# ============================================================

def save_pdf(
    username,
    filename,
    file_bytes
):

    file_hash = calculate_file_hash(
        file_bytes
    )


    existing = document_exists(
        username,
        file_hash
    )


    if existing:

        return existing


    safe_name = (
        file_hash[:16]
        + "_"
        + filename
    )


    file_path = os.path.join(
        UPLOAD_DIR,
        safe_name
    )


    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            file_bytes
        )


    processed_path = os.path.join(
        PROCESSED_DIR,
        file_hash + ".pkl"
    )


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO documents
        (
            username,
            filename,
            file_hash,
            file_path,
            processed_path
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            username,
            filename,
            file_hash,
            file_path,
            processed_path
        )
    )


    connection.commit()

    connection.close()


    return {
        "filename": filename,
        "file_hash": file_hash,
        "file_path": file_path,
        "processed_path": processed_path
    }


# ============================================================
# SAVE PROCESSED DOCUMENT
# ============================================================

def save_processed_document(
    file_hash,
    chunks,
    embeddings
):

    processed_path = os.path.join(
        PROCESSED_DIR,
        file_hash + ".pkl"
    )


    data = {

        "chunks": chunks,

        "embeddings": embeddings
    }


    with open(
        processed_path,
        "wb"
    ) as file:

        pickle.dump(
            data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )


    return processed_path


# ============================================================
# LOAD PROCESSED DOCUMENT
# ============================================================

def load_processed_document(
    file_hash
):

    processed_path = os.path.join(
        PROCESSED_DIR,
        file_hash + ".pkl"
    )


    if not os.path.exists(
        processed_path
    ):

        return None


    try:

        with open(
            processed_path,
            "rb"
        ) as file:

            data = pickle.load(
                file
            )


        return data


    except Exception as error:

        print(
            "LOAD ERROR:",
            error
        )

        return None


# ============================================================
# UPDATE CHUNK COUNT
# ============================================================

def update_chunk_count(
    username,
    file_hash,
    count
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        UPDATE documents

        SET chunks_count = ?

        WHERE username = ?

        AND file_hash = ?
        """,
        (
            count,
            username,
            file_hash
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# GET USER DOCUMENTS
# ============================================================

def get_user_documents(
    username
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM documents
        WHERE username = ?
        ORDER BY created_at DESC
        """,
        (
            username,
        )
    )


    documents = cursor.fetchall()

    connection.close()

    return documents


# ============================================================
# SAVE QUESTION
# ============================================================

def save_question(
    username,
    question,
    answer
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO questions
        (
            username,
            question,
            answer
        )
        VALUES (?, ?, ?)
        """,
        (
            username,
            question,
            answer
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# GET QUESTIONS
# ============================================================

def get_user_questions(
    username
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM questions

        WHERE username = ?

        ORDER BY created_at DESC
        """,
        (
            username,
        )
    )


    questions = cursor.fetchall()

    connection.close()

    return questions


# ============================================================
# UPDATE PROGRESS
# ============================================================

def update_progress(
    username,
    questions_asked
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO progress
        (
            username,
            questions_asked
        )

        VALUES (?, ?)

        ON CONFLICT(username)

        DO UPDATE SET

            questions_asked = excluded.questions_asked,

            updated_at =
            CURRENT_TIMESTAMP
        """,
        (
            username,
            questions_asked
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# GET PROGRESS
# ============================================================

def get_progress(
    username
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT *
        FROM progress

        WHERE username = ?
        """,
        (
            username,
        )
    )


    progress = cursor.fetchone()

    connection.close()

    return progress


# ============================================================
# DELETE DOCUMENT
# ============================================================

def delete_document(
    username,
    file_hash
):

    document = document_exists(
        username,
        file_hash
    )


    if not document:

        return False


    # Delete original PDF

    if os.path.exists(
        document["file_path"]
    ):

        os.remove(
            document["file_path"]
        )


    # Delete processed data

    if os.path.exists(
        document["processed_path"]
    ):

        os.remove(
            document["processed_path"]
        )


    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM documents

        WHERE username = ?

        AND file_hash = ?
        """,
        (
            username,
            file_hash
        )
    )


    connection.commit()

    connection.close()

    return True


# ============================================================
# INITIALIZE
# ============================================================

initialize_database()