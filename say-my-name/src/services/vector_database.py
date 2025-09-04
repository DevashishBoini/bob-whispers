"""
AI Assistant Vector Database Service

This module provides ChromaDB integration for semantic memory storage and retrieval.
ChromaDB stores conversation messages as high-dimensional vectors (embeddings) that enable
semantic similarity search across conversation history.

Key Concepts:
- Embeddings: Text converted to numerical vectors that capture semantic meaning
- Collections: Isolated storage containers (one per conversation in our case)
- Similarity Search: Finding semantically related content using vector mathematics
- Metadata: Additional information stored with each embedding (timestamps, message IDs, etc.)

Usage Example:
    from services.vector_database import VectorDatabase, init_vector_database
    
    # Initialize vector database
    vector_db = init_vector_database()
    
    # Create conversation collection
    vector_db.create_conversation_collection("conv_abc123")
    
    # Store message embedding
    vector_db.store_message_embedding(
        conversation_id="conv_abc123",
        message_text="How do I use FastAPI?",
        message_id=42,
        embedding_vector=[0.1, 0.2, 0.3, ...]  # 768-dimensional vector
    )

Theory Background:
ChromaDB is a vector database designed for AI applications. It stores text as mathematical
vectors (embeddings) where semantically similar text has similar vector representations.
This enables "semantic search" - finding relevant content based on meaning rather than
exact keyword matches.

For example:
- "debugging Python code" and "fixing Python errors" would have similar embeddings
- "FastAPI endpoints" and "API route creation" would be considered related
- This works even when no words match exactly between queries and stored content

Architecture:
VectorDatabaseConfig - Configuration settings for ChromaDB
VectorDatabase - Main interface for vector operations  
ConversationCollection - Manages embeddings for a single conversation
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings
from chromadb import Collection, PersistentClient
from chromadb.utils import embedding_functions
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

# Handle imports for different execution contexts
try:
    from ..config import get_config
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from config import get_config
    except ImportError:
        from src.config import get_config


class VectorDatabaseConfig(BaseModel):
    """
    Configuration settings for ChromaDB vector database.
    
    Defines database paths, collection settings, and operational parameters
    for the vector storage system.
    
    Attributes:
        persist_directory: Path where ChromaDB stores its data files
        collection_prefix: Prefix for conversation collection names
        embedding_dimension: Expected dimension of embedding vectors
        max_results: Default maximum number of search results to return
        similarity_threshold: Minimum similarity score for relevant results
        enable_anonymized_telemetry: Whether to send usage data to ChromaDB team
    """
    
    model_config = ConfigDict(
        frozen=True,  # Immutable configuration
        validate_assignment=True
    )
    
    persist_directory: Path = Field(
        description="Directory path where ChromaDB persists its data",
        examples=[Path("data/chromadb")]
    )
    
    collection_prefix: str = Field(
        default="conversation_",
        description="Prefix added to all conversation collection names",
        min_length=1,
        max_length=50
    )
    
    embedding_dimension: int = Field(
        default=768,
        description="Expected dimension of embedding vectors (Google embeddings: 768)",
        gt=0,
        le=2048
    )
    
    max_results: int = Field(
        default=10,
        description="Default maximum number of search results to return",
        gt=0,
        le=100
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum cosine similarity score for considering results relevant",
        ge=0.0,
        le=1.0
    )
    
    enable_anonymized_telemetry: bool = Field(
        default=False,
        description="Whether ChromaDB can send anonymized usage statistics"
    )


@dataclass
class ConversationEmbedding:
    """
    Represents a stored conversation message embedding with metadata.
    
    This data structure contains both the semantic vector representation
    and contextual information about the original message.
    
    Attributes:
        id: Unique identifier for this embedding record
        conversation_id: ID of the parent conversation
        message_id: Database ID of the original message
        message_text: Original text content that was embedded
        embedding_vector: High-dimensional vector representation
        timestamp: When this embedding was created
        metadata: Additional key-value pairs for filtering and context
    
    Example:
        embedding = ConversationEmbedding(
            id="emb_xyz789",
            conversation_id="conv_abc123", 
            message_id=42,
            message_text="How do I create FastAPI endpoints?",
            embedding_vector=[0.1, 0.2, 0.3, ...],  # 768 dimensions
            timestamp=datetime.now(),
            metadata={"input_type": "text", "user_message": True}
        )
    """
    
    id: str
    conversation_id: str
    message_id: int
    message_text: str
    embedding_vector: List[float]
    timestamp: datetime
    metadata: Dict[str, Any]


class ConversationCollection:
    """
    Manages embeddings for a single conversation using ChromaDB.
    
    Each conversation gets its own ChromaDB collection to enable conversation-scoped
    semantic search. This prevents irrelevant results from other conversations
    while maintaining rich context within each conversation thread.
    
    This class provides a clean interface for common operations on a conversation's
    embedding collection without exposing ChromaDB implementation details.
    """
    
    def __init__(self, collection: Collection, conversation_id: str):
        """
        Initialize collection manager for a specific conversation.
        
        Args:
            collection: ChromaDB collection object
            conversation_id: Unique identifier for the conversation
            
        Example:
            collection_obj = chroma_client.get_collection("conversation_conv_abc123")
            conv_collection = ConversationCollection(collection_obj, "conv_abc123")
        """
        self.collection = collection
        self.conversation_id = conversation_id
        self._validate_collection()
    
    def _validate_collection(self) -> None:
        """
        Validate that the collection is properly configured.
        
        Ensures the ChromaDB collection exists and has expected properties.
        
        Raises:
            ValueError: If collection is invalid or misconfigured
        """
        if not self.collection:
            raise ValueError(f"Collection for conversation {self.conversation_id} is None")
        
        # Verify collection name matches conversation ID
        expected_name = f"conversation_{self.conversation_id}"
        if self.collection.name != expected_name:
            raise ValueError(
                f"Collection name mismatch: expected '{expected_name}', "
                f"got '{self.collection.name}'"
            )
    
    def add_message_embedding(self, embedding: ConversationEmbedding) -> None:
        """
        Add a message embedding to this conversation's collection.
        
        Stores the embedding vector along with metadata for later retrieval.
        Each embedding represents one message in the conversation.
        
        Args:
            embedding: ConversationEmbedding object with vector and metadata
            
        Raises:
            ValueError: If embedding conversation_id doesn't match this collection
            Exception: If ChromaDB storage operation fails
            
        Example:
            embedding = ConversationEmbedding(
                id="emb_123",
                conversation_id="conv_abc123",
                message_id=42,
                message_text="How do I debug FastAPI?",
                embedding_vector=[0.1, 0.2, ...],
                timestamp=datetime.now(),
                metadata={"type": "user_message"}
            )
            conv_collection.add_message_embedding(embedding)
        """
        # Validate embedding belongs to this conversation
        if embedding.conversation_id != self.conversation_id:
            raise ValueError(
                f"Embedding conversation_id '{embedding.conversation_id}' "
                f"doesn't match collection conversation_id '{self.conversation_id}'"
            )
        
        try:
            # Prepare metadata for ChromaDB storage
            chroma_metadata = {
                "conversation_id": embedding.conversation_id,
                "message_id": str(embedding.message_id),
                "timestamp": embedding.timestamp.isoformat(),
                "message_length": len(embedding.message_text),
                **embedding.metadata  # Include any additional metadata
            }
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=[embedding.id],
                embeddings=[embedding.embedding_vector],
                documents=[embedding.message_text],
                metadatas=[chroma_metadata]
            )
            
            print(f"Added embedding {embedding.id} to conversation {self.conversation_id}")
            
        except Exception as e:
            print(f"Failed to add embedding to ChromaDB: {e}")
            raise
    
    def search_similar_messages(
        self, 
        query_embedding: List[float], 
        max_results: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[ConversationEmbedding]:
        """
        Find messages similar to the query embedding within this conversation.
        
        Performs semantic similarity search using cosine similarity between
        the query vector and stored message embeddings.
        
        Args:
            query_embedding: Vector representation of the search query
            max_results: Maximum number of similar messages to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List[ConversationEmbedding]: Similar messages ordered by relevance
            
        Example:
            # Search for messages similar to "debugging FastAPI errors"
            query_vector = [0.2, 0.3, 0.1, ...]  # From embedding service
            similar_messages = conv_collection.search_similar_messages(
                query_embedding=query_vector,
                max_results=3,
                similarity_threshold=0.75
            )
            
            for msg in similar_messages:
                print(f"Similar: {msg.message_text[:50]}...")
        """
        try:
            # Query ChromaDB for similar vectors
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results into ConversationEmbedding objects
            similar_embeddings = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Convert ChromaDB distance to similarity score
                    distance = results['distances'][0][i]
                    similarity_score = 1.0 - distance  # ChromaDB returns L2 distance
                    
                    # Filter by similarity threshold
                    if similarity_score >= similarity_threshold:
                        metadata = results['metadatas'][0][i]
                        
                        # Reconstruct ConversationEmbedding
                        embedding = ConversationEmbedding(
                            id=results['ids'][0][i],
                            conversation_id=metadata['conversation_id'],
                            message_id=int(metadata['message_id']),
                            message_text=results['documents'][0][i],
                            embedding_vector=query_embedding,  # Placeholder - we don't store original vectors
                            timestamp=datetime.fromisoformat(metadata['timestamp']),
                            metadata={
                                'similarity_score': similarity_score,
                                **{k: v for k, v in metadata.items() 
                                   if k not in ['conversation_id', 'message_id', 'timestamp']}
                            }
                        )
                        
                        similar_embeddings.append(embedding)
            
            print(f"Found {len(similar_embeddings)} similar messages in conversation {self.conversation_id}")
            return similar_embeddings
            
        except Exception as e:
            print(f"Error searching similar messages: {e}")
            return []
    
    def get_total_embeddings(self) -> int:
        """
        Get the total number of embeddings stored in this conversation.
        
        Returns:
            int: Number of message embeddings in this conversation
        """
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            print(f"Error getting embedding count: {e}")
            return 0


class VectorDatabase:
    """
    Main interface for vector database operations using ChromaDB.
    
    Provides high-level methods for managing conversation embeddings while
    abstracting away ChromaDB implementation details. This class handles
    database initialization, collection management, and cross-conversation operations.
    
    The vector database enables semantic search across conversation history
    by storing message embeddings and providing similarity search capabilities.
    """
    
    def __init__(self, config: VectorDatabaseConfig):
        """
        Initialize vector database with configuration.
        
        Sets up ChromaDB client with persistent storage and prepares
        the database for conversation embedding operations.
        
        Args:
            config: VectorDatabaseConfig with database settings
            
        Raises:
            Exception: If ChromaDB initialization fails
        """
        self.config = config
        self.client: Optional[PersistentClient] = None
        self._initialize_database()
        
        print(f"Vector database initialized at: {config.persist_directory}")
    
    def _initialize_database(self) -> None:
        """
        Initialize ChromaDB client with proper configuration.
        
        Creates the persistent database client and ensures the storage
        directory exists with appropriate permissions.
        
        Raises:
            Exception: If database initialization fails
        """
        try:
            # Ensure persist directory exists
            self.config.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with settings
            self.client = chromadb.PersistentClient(
                path=str(self.config.persist_directory),
                settings=Settings(
                    anonymized_telemetry=self.config.enable_anonymized_telemetry,
                    allow_reset=False  # Prevent accidental data loss
                )
            )
            
            print(f"ChromaDB client initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def create_conversation_collection(self, conversation_id: str) -> ConversationCollection:
        """
        Create or get a collection for a specific conversation.
        
        Each conversation gets its own ChromaDB collection to enable
        conversation-scoped semantic search and prevent cross-conversation pollution.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            ConversationCollection: Manager for the conversation's embeddings
            
        Raises:
            Exception: If collection creation fails
            
        Example:
            conv_collection = vector_db.create_conversation_collection("conv_abc123")
            # Now you can add embeddings and search within this conversation
        """
        if not self.client:
            raise RuntimeError("Vector database not initialized")
        
        collection_name = f"{self.config.collection_prefix}{conversation_id}"
        
        try:
            # Get or create collection (ChromaDB handles both cases)
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"conversation_id": conversation_id}
            )
            
            conv_collection = ConversationCollection(collection, conversation_id)
            print(f"Created/retrieved collection: {collection_name}")
            
            return conv_collection
            
        except Exception as e:
            print(f"Failed to create collection for conversation {conversation_id}: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Provides insights into database usage, collection counts, and
        storage information for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Database statistics including:
                - total_collections: Number of conversation collections
                - total_embeddings: Total number of stored embeddings
                - database_size_mb: Approximate database size
                - collections_info: Details about each collection
        """
        if not self.client:
            return {"error": "Database not initialized"}
        
        try:
            # Get all collections
            collections = self.client.list_collections()
            
            total_collections = len(collections)
            total_embeddings = 0
            collections_info = []
            
            # Gather stats from each collection
            for collection in collections:
                count = collection.count()
                total_embeddings += count
                
                collections_info.append({
                    "name": collection.name,
                    "embedding_count": count,
                    "conversation_id": collection.name.replace(self.config.collection_prefix, "")
                })
            
            # Estimate database size (rough approximation)
            db_size_mb = 0
            if self.config.persist_directory.exists():
                for file_path in self.config.persist_directory.rglob('*'):
                    if file_path.is_file():
                        db_size_mb += file_path.stat().st_size
                db_size_mb = round(db_size_mb / (1024 * 1024), 2)  # Convert to MB
            
            stats = {
                "total_collections": total_collections,
                "total_embeddings": total_embeddings,
                "database_size_mb": db_size_mb,
                "persist_directory": str(self.config.persist_directory),
                "collections_info": collections_info
            }
            
            print(f"Database stats: {total_collections} collections, {total_embeddings} embeddings")
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"error": str(e)}


def init_vector_database() -> VectorDatabase:
    """
    Initialize vector database with application configuration.
    
    Creates a VectorDatabase instance using settings from the main
    application configuration. This is the primary way other parts
    of the application should get a configured vector database.
    
    Returns:
        VectorDatabase: Fully configured vector database instance
        
    Raises:
        Exception: If initialization fails
        
    Example:
        vector_db = init_vector_database()
        
        # Create collection for a conversation
        conv_collection = vector_db.create_conversation_collection("conv_abc123")
        
        # Store message embedding
        embedding = ConversationEmbedding(...)
        conv_collection.add_message_embedding(embedding)
    """
    try:
        # Get application configuration
        app_config = get_config()
        
        # Create vector database configuration
        vector_config = VectorDatabaseConfig(
            persist_directory=app_config.database.chromadb_path,
            collection_prefix="conversation_",
            embedding_dimension=768,  # Google Gemini embeddings dimension
            max_results=10,
            similarity_threshold=0.7,
            enable_anonymized_telemetry=False
        )
        
        # Initialize vector database
        vector_db = VectorDatabase(vector_config)
        
        print("Vector database initialization completed successfully")
        return vector_db
        
    except Exception as e:
        print(f"Failed to initialize vector database: {e}")
        raise


def test_vector_database() -> None:
    """
    Test vector database functionality with sample data.
    
    This function validates that ChromaDB is working correctly by:
    1. Initializing the database
    2. Creating a test conversation collection
    3. Adding sample embeddings
    4. Performing similarity searches
    5. Checking database statistics
    """
    print("\n🧪 Testing vector database functionality...")
    
    try:
        # Test 1: Initialize database
        print("\n1. Initializing vector database...")
        vector_db = init_vector_database()
        
        # Test 2: Create conversation collection
        print("\n2. Creating test conversation collection...")
        test_conv_id = "test_conv_123"
        conv_collection = vector_db.create_conversation_collection(test_conv_id)
        
        # Test 3: Create sample embeddings
        print("\n3. Adding sample embeddings...")
        sample_embeddings = [
            ConversationEmbedding(
                id=f"test_emb_{i}",
                conversation_id=test_conv_id,
                message_id=i,
                message_text=f"Sample message {i} about AI and programming",
                embedding_vector=[0.1 * i, 0.2 * i, 0.3 * i] + [0.0] * 765,  # 768-dim vector
                timestamp=datetime.now(),
                metadata={"test": True, "message_type": "sample"}
            )
            for i in range(1, 4)
        ]
        
        for embedding in sample_embeddings:
            conv_collection.add_message_embedding(embedding)
        
        # Test 4: Check collection stats
        print("\n4. Checking collection statistics...")
        embedding_count = conv_collection.get_total_embeddings()
        print(f"Total embeddings in test collection: {embedding_count}")
        
        # Test 5: Test similarity search
        print("\n5. Testing similarity search...")
        query_vector = [0.15, 0.25, 0.35] + [0.0] * 765  # Similar to embedding 1
        similar_messages = conv_collection.search_similar_messages(
            query_embedding=query_vector,
            max_results=2,
            similarity_threshold=0.5
        )
        
        print(f"Found {len(similar_messages)} similar messages")
        for msg in similar_messages:
            print(f"  - {msg.message_text} (similarity: {msg.metadata.get('similarity_score', 'N/A')})")
        
        # Test 6: Database statistics
        print("\n6. Getting database statistics...")
        stats = vector_db.get_database_stats()
        print(f"Database contains {stats.get('total_collections', 0)} collections")
        print(f"Database size: {stats.get('database_size_mb', 0)} MB")
        
        print("\n✅ All vector database tests passed!")
        
    except Exception as e:
        print(f"\n❌ Vector database test failed: {e}")
        raise


# Development utility - run this file directly to test vector database
if __name__ == "__main__":
    """
    Test vector database functionality when run directly.
    
    Usage:
        python src/services/vector_database.py
        
    This will test ChromaDB initialization, collection creation,
    embedding storage, and similarity search operations.
    """
    test_vector_database()