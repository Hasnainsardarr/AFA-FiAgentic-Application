"""
Vector Memory for AFA using ChromaDB and OpenAI embeddings.

This module provides persistent storage and retrieval of analysis reports
using ChromaDB as the vector store and OpenAI's text-embedding-3-small
model for embeddings (via OpenRouter if configured).

Usage:
    from backend.memory import VectorMemory

    # Initialize (creates/loads collection)
    memory = VectorMemory()

    # Store a report
    memory.store_report_embedding(
        "AAPL showed strong performance...",
        metadata={
            "ticker": "AAPL",
            "date": "2024-03-01",
            "tags": ["earnings", "analysis"],
        }
    )

    # Query similar reports
    similar = memory.query_similar_reports(
        "What were AAPL's recent earnings?",
        n=3
    )
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import EmbeddingFunction


# Load environment variables
load_dotenv()


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """OpenAI embedding function for ChromaDB that supports OpenRouter."""

    def __init__(self) -> None:
        """Initialize with API key and optional OpenRouter base URL."""
        # Try OpenRouter first, fall back to OpenAI
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not self.api_key:
                raise ValueError(
                    "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set in environment"
                )
            self.client = OpenAI(api_key=self.api_key)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # OpenAI's text-embedding-3-small via direct API or OpenRouter
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            encoding_format="float",
        )
        return [e.embedding for e in response.data]


class VectorMemory:
    """Vector store for AFA analysis reports using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "afa-memory",
        persist_dir: str = "data/chromadb",
    ) -> None:
        """Initialize ChromaDB with persistent storage.

        Args:
            collection_name: Name of the ChromaDB collection to use
            persist_dir: Directory to store the database files
        """
        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.db = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False,
            )
        )

        # Get or create collection with OpenAI embeddings
        self.collection = self.db.get_or_create_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(),
            metadata={"description": "AFA analysis reports and metadata"},
        )

    def store_report_embedding(
        self,
        report_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a report with its metadata in the vector store.

        Args:
            report_text: The analysis report text to embed
            metadata: Optional dict with fields like ticker, date, tags

        Returns:
            id: The unique ID of the stored document

        Example metadata:
            {
                "ticker": "AAPL",
                "date": "2024-03-01",
                "tags": ["earnings", "technical", "sentiment"],
                "author": "AFA Agent",
                "type": "analysis"
            }
        """
        # Generate a unique ID based on timestamp
        doc_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Normalize metadata
        if metadata is None:
            metadata = {}
        
        # Ensure date is present
        if "date" not in metadata:
            metadata["date"] = datetime.utcnow().strftime("%Y-%m-%d")

        # Convert any non-string metadata to strings for ChromaDB
        for k, v in metadata.items():
            if isinstance(v, (list, tuple)):
                metadata[k] = ",".join(str(x) for x in v)
            elif not isinstance(v, str):
                metadata[k] = str(v)

        # Store in ChromaDB
        self.collection.add(
            documents=[report_text],
            metadatas=[metadata],
            ids=[doc_id],
        )

        return doc_id

    def query_similar_reports(
        self,
        query_text: str,
        n: int = 3,
        filter_metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar reports using semantic search.

        Args:
            query_text: The text to find similar reports for
            n: Maximum number of results to return
            filter_metadata: Optional metadata filters (e.g., {"ticker": "AAPL"})

        Returns:
            List of dicts with fields:
                - text: The report text
                - metadata: The stored metadata
                - distance: Cosine distance (lower is more similar)
                - id: The document ID
        """
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n,
            where=filter_metadata,  # ChromaDB will handle the filtering
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        similar_reports = []
        for i in range(len(results["ids"][0])):  # ChromaDB returns nested lists
            similar_reports.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": float(results["distances"][0][i]),
                "id": results["ids"][0][i],
            })

        return similar_reports

    def get_all_reports(
        self,
        filter_metadata: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all stored reports, optionally filtered by metadata.

        Args:
            filter_metadata: Optional metadata filters (e.g., {"ticker": "AAPL"})

        Returns:
            List of dicts with fields:
                - text: The report text
                - metadata: The stored metadata
                - id: The document ID
        """
        # Get all documents (ChromaDB handles the filtering)
        results = self.collection.get(
            where=filter_metadata,
            include=["documents", "metadatas"]
        )

        # Format results
        reports = []
        for i in range(len(results["ids"])):
            reports.append({
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
                "id": results["ids"][i],
            })

        return reports
