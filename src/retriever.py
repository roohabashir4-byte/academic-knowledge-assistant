from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "rag_database"

INDEX_PATH = DATABASE_DIR / "faiss.index"
METADATA_PATH = DATABASE_DIR / "metadata.json"
CONFIG_PATH = DATABASE_DIR / "config.json"


class RAGRetriever:
    def __init__(self):
        # Load configuration
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Load FAISS index
        self.index = faiss.read_index(str(INDEX_PATH))

        # Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Get embedding model name
        model_name = self.config["embedding_model"]

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Validate database
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata have different numbers of records."
            )

    def search(self, query, top_k=5):
        """Find the most relevant document chunks for a question."""

        if not query or not query.strip():
            return []

        # Convert question into an embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Search FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index_id in zip(scores[0], indices[0]):

            # FAISS uses -1 when no result exists
            if index_id == -1:
                continue

            record = self.metadata[index_id].copy()

            record["score"] = float(score)

            results.append(record)

        return results
