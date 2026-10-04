# ============================================================
# FAST RAG RETRIEVER
# ============================================================

import time
import threading

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_embedding_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    """
    Load the Sentence Transformer model only once.
    """

    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    with _model_lock:

        if _embedding_model is not None:
            return _embedding_model

        print("=" * 60)
        print("LOADING EMBEDDING MODEL")
        print("=" * 60)

        start_time = time.time()

        _embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        elapsed = time.time() - start_time

        print(
            f"EMBEDDING MODEL LOADED: "
            f"{elapsed:.2f} seconds"
        )

        print("=" * 60)

        return _embedding_model


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:

    def __init__(self):

        print("=" * 60)
        print("CREATING RETRIEVER")
        print("=" * 60)

        start_time = time.time()

        self.model = get_embedding_model()

        # ----------------------------------------------------
        # DOCUMENT DATA
        # ----------------------------------------------------

        self.chunks = []

        self.embeddings = None

        # ----------------------------------------------------
        # NORMALIZED EMBEDDINGS
        # ----------------------------------------------------
        #
        # We keep normalized vectors separately.
        #
        # This means search can use a simple dot product
        # instead of normalizing thousands of vectors again.
        # ----------------------------------------------------

        self.normalized_embeddings = None

        elapsed = time.time() - start_time

        print(
            f"RETRIEVER READY: "
            f"{elapsed:.2f} seconds"
        )

        print("=" * 60)


    # ========================================================
    # NORMALIZE EMBEDDINGS
    # ========================================================

    @staticmethod
    def normalize_embeddings(embeddings):
        """
        Normalize vectors once.
        """

        if embeddings is None:
            return None

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.size == 0:
            return embeddings

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0

        return (
            embeddings / norms
        )


    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        chunks,
        batch_size=32
    ):
        """
        Create normalized embeddings for document chunks.
        """

        self.chunks = list(
            chunks
        )

        if not self.chunks:

            print(
                "No chunks available."
            )

            self.embeddings = None

            self.normalized_embeddings = None

            return


        print("=" * 60)

        print(
            f"CREATING EMBEDDINGS "
            f"FOR {len(self.chunks)} CHUNKS"
        )

        print("=" * 60)

        start_time = time.time()


        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        self.embeddings = self.model.encode(

            self.chunks,

            convert_to_numpy=True,

            batch_size=batch_size,

            show_progress_bar=True,

            normalize_embeddings=False
        )


        # ----------------------------------------------------
        # NORMALIZE ONCE
        # ----------------------------------------------------

        self.normalized_embeddings = (
            self.normalize_embeddings(
                self.embeddings
            )
        )


        elapsed = time.time() - start_time


        print(
            f"EMBEDDINGS CREATED: "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"EMBEDDING SHAPE: "
            f"{self.embeddings.shape}"
        )

        print("=" * 60)


    # ========================================================
    # LOAD PRE-COMPUTED EMBEDDINGS
    # ========================================================

    def load_embeddings(
        self,
        chunks,
        embeddings
    ):
        """
        Load previously saved embeddings.

        This is important for PDFs that have already been
        processed and saved in storage.
        """

        self.chunks = list(
            chunks
        )

        if embeddings is None:

            self.embeddings = None

            self.normalized_embeddings = None

            return


        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )


        if len(self.chunks) != len(
            self.embeddings
        ):

            raise ValueError(
                "Number of chunks does not "
                "match number of embeddings."
            )


        # ----------------------------------------------------
        # NORMALIZE ONCE
        # ----------------------------------------------------

        self.normalized_embeddings = (
            self.normalize_embeddings(
                self.embeddings
            )
        )


        print(
            f"Loaded {len(self.chunks)} "
            f"chunks from saved storage."
        )


    # ========================================================
    # ADD PRE-COMPUTED DOCUMENT
    # ========================================================

    def add_precomputed(
        self,
        chunks,
        embeddings
    ):
        """
        Add already-embedded chunks to the current
        retriever.
        """

        if not chunks:
            return


        new_chunks = list(
            chunks
        )


        new_embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )


        if len(new_chunks) != len(
            new_embeddings
        ):

            raise ValueError(
                "Chunks and embeddings "
                "count do not match."
            )


        # ----------------------------------------------------
        # FIRST DOCUMENT
        # ----------------------------------------------------

        if self.embeddings is None:

            self.chunks = (
                new_chunks
            )

            self.embeddings = (
                new_embeddings
            )

        else:

            self.chunks.extend(
                new_chunks
            )

            self.embeddings = (
                np.vstack(
                    [
                        self.embeddings,
                        new_embeddings
                    ]
                )
            )


        # ----------------------------------------------------
        # NORMALIZE ONCE
        # ----------------------------------------------------

        self.normalized_embeddings = (
            self.normalize_embeddings(
                self.embeddings
            )
        )


        print(
            f"Added {len(new_chunks)} "
            f"precomputed chunks."
        )

        print(
            f"Total chunks: "
            f"{len(self.chunks)}"
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question,
        top_k=3,
        min_score=None
    ):
        """
        Find the most relevant document chunks.

        Uses cosine similarity through normalized
        vector dot product.
        """

        search_start = time.time()


        # ----------------------------------------------------
        # CHECK QUESTION
        # ----------------------------------------------------

        if not question or not question.strip():

            return []


        # ----------------------------------------------------
        # CHECK DOCUMENTS
        # ----------------------------------------------------

        if (
            self.normalized_embeddings is None
            or not self.chunks
        ):

            print(
                "No document embeddings available."
            )

            return []


        # ----------------------------------------------------
        # LIMIT TOP K
        # ----------------------------------------------------

        top_k = max(
            1,
            min(
                int(top_k),
                len(self.chunks)
            )
        )


        # ----------------------------------------------------
        # QUESTION EMBEDDING
        # ----------------------------------------------------

        question_embedding = (
            self.model.encode(

                question,

                convert_to_numpy=True,

                normalize_embeddings=True
            )
        )


        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # COSINE SIMILARITY
        # ----------------------------------------------------
        #
        # Both document and question vectors are normalized.
        #
        # Therefore:
        #
        # cosine similarity = dot product
        # ----------------------------------------------------

        similarities = np.dot(

            self.normalized_embeddings,

            question_embedding
        )


        # ----------------------------------------------------
        # GET TOP RESULTS
        # ----------------------------------------------------

        top_indices = np.argpartition(
            similarities,
            -top_k
        )[-top_k:]


        # Sort the selected results
        # from highest score to lowest.

        top_indices = top_indices[
            np.argsort(
                similarities[top_indices]
            )[::-1]
        ]


        # ----------------------------------------------------
        # BUILD RESULTS
        # ----------------------------------------------------

        results = []


        for index in top_indices:

            score = float(
                similarities[index]
            )


            # ------------------------------------------------
            # OPTIONAL SCORE FILTER
            # ------------------------------------------------

            if (
                min_score is not None
                and score < min_score
            ):

                continue


            results.append(

                {
                    "text":
                        self.chunks[index],

                    "score":
                        score
                }

            )


        elapsed = (
            time.time()
            - search_start
        )


        print(
            f"Retrieved "
            f"{len(results)} results "
            f"in {elapsed:.4f} seconds."
        )


        if results:

            print(
                "Scores:",
                [
                    round(
                        r["score"],
                        4
                    )
                    for r in results
                ]
            )


        return results