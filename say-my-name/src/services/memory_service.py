# """
# AI Assistant Smart Memory Service - Conversation-Scoped

# This module provides semantic memory capabilities using ChromaDB for vector storage
# and Google embeddings for semantic search. Memory is scoped to individual conversations
# to maintain privacy boundaries and provide predictable, Claude-like behavior.

# Key Concepts Explained:
# 1. Vector Embeddings: Convert text to numerical representations that capture meaning
# 2. Semantic Search: Find similar content based on meaning, not just word matches
# 3. ChromaDB: Database that stores and searches vectors efficiently
# 4. Conversation Scoping: Memory searches limited to current conversation only
# 5. Hybrid Memory: Combines recent conversation history with relevant past context

# Architecture Benefits:
# - Clean conversation boundaries (each thread is independent)
# - Predictable AI behavior (only references current conversation)
# - Privacy-friendly (no cross-contamination between conversations)
# - Claude-like user experience (familiar conversation model)

# Usage Example:
#     memory = SmartMemoryService()
    
#     # Store a conversation exchange in a specific conversation
#     memory.store_conversation_memory(
#         conv_id="conv_123",
#         user_msg="How do I build REST APIs?",
#         ai_response="You can use FastAPI to build REST APIs..."
#     )
    
#     # Later, search within THAT conversation only
#     relevant_context = memory.search_relevant_memories(
#         "API development tips",
#         conversation_id="conv_123"  # Scoped to this conversation
#     )
#     # Returns: Previous API discussions from conv_123 ONLY


# Class Structure :

# SmartMemoryService (Main Orchestrator)
# ├── EmbeddingManager (Text → Vector Conversion)
# │   ├── create_embedding() - Single text to 768D vector
# │   ├── create_batch_embeddings() - Multiple texts efficiently  
# │   └── Google API integration with error handling
# │
# ├── ChromaMemoryStore (Vector Storage & Retrieval)
# │   ├── store_memory() - Save vectors with conversation metadata
# │   ├── search_similar_memories() - Find similar vectors with conversation filtering
# │   ├── get_collection_stats() - Collection analytics  
# │   └── ChromaDB persistent storage management
# │
# └── MemoryResult (Structured Search Results)
#     ├── content - The relevant conversation text
#     ├── conversation_id - Which conversation it came from  
#     ├── similarity_score - How relevant (0.0 to 1.0)
#     ├── timestamp - When this memory was created
#     └── metadata - Additional context information
# """

# import json
# import uuid
# from datetime import datetime
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass

# import chromadb
# from chromadb.config import Settings

# # Use the correct LangChain imports for installed versions
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document

# # from config import get_config
# # Handle imports for different execution contexts
# try:
#     # When running from services/ directory, use relative import
#     from ..config import get_config
# except ImportError:
#     try:
#         # When running from ai-assistant/ directory
#         import sys
#         import os
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#         from config import get_config
#     except ImportError:
#         # When running as module from other contexts
#         from src.config import get_config

# @dataclass
# class MemoryResult:
#     """
#     Represents a relevant memory found through semantic search.
    
#     This class structures the results of memory searches, providing
#     both the content and metadata about why this memory was relevant.
#     All results are scoped to the conversation being searched.
    
#     Attributes:
#         content: The actual conversation content or summary
#         conversation_id: Which conversation this memory came from
#         timestamp: When this conversation happened
#         similarity_score: How relevant this memory is (0.0 to 1.0)
#         metadata: Additional information about the conversation
#         memory_type: Type of memory ('conversation', 'summary', 'fact')
#     """
    
#     content: str
#     conversation_id: str
#     timestamp: datetime
#     similarity_score: float
#     metadata: Dict[str, Any]
#     memory_type: str = 'conversation'
    
#     def __str__(self) -> str:
#         """String representation showing key info."""
#         score_percent = int(self.similarity_score * 100)
#         conv_short = self.conversation_id[:8] + "..."
#         return f"Memory({score_percent}% relevant, {conv_short}): {self.content[:50]}..."


# class EmbeddingManager:
#     """
#     Manages text-to-vector conversions using Google's embedding model.
    
#     This class handles the complex process of converting human-readable text
#     into numerical vectors that capture semantic meaning. These vectors
#     are what enable semantic search capabilities within conversations.
    
#     Concept Explanation:
#     - Text like "FastAPI development" becomes a vector like [0.2, -0.1, 0.8, ...]
#     - Similar texts get similar vectors
#     - We can measure vector similarity to find related content
#     - All within the same conversation thread for privacy
    
#     Responsibilities:
#     - Convert single text strings to embedding vectors
#     - Handle batch conversions for efficiency
#     - Manage Google API integration and error handling
#     - Provide consistent vector representations
#     """
    
#     def __init__(self, model_name: str = "models/embedding-001"):
#         """
#         Initialize embedding manager with Google's model.
        
#         Args:
#             model_name: Google embedding model to use
#         """
#         self.model_name = model_name
        
#         try:
#             config = get_config()
#             self.embeddings = GoogleGenerativeAIEmbeddings(
#                 model=model_name,
#                 google_api_key=config.gemini.api_key
#             )
#             print(f"✅ Embedding manager initialized with {model_name}")
            
#         except Exception as e:
#             print(f"❌ Failed to initialize embedding manager: {e}")
#             raise
    
#     def create_embedding(self, text: str) -> List[float]:
#         """
#         Convert text to embedding vector.
        
#         This is where the magic happens! Text gets converted into a list
#         of numbers (usually 768 of them) that represent the meaning.
        
#         Args:
#             text: Text to convert to embedding
            
#         Returns:
#             List[float]: Vector representation of the text (768 dimensions)
            
#         Example:
#             vector = create_embedding("I love Python programming")
#             # Returns: [0.234, -0.567, 0.123, ..., 0.890]  (768 numbers)
            
#             # Similar text gets similar vector:
#             vector2 = create_embedding("Python is great for coding")
#             # Returns: [0.198, -0.543, 0.156, ..., 0.823]  (similar pattern!)
#         """
#         try:
#             # Clean and prepare text
#             cleaned_text = text.strip()
#             if not cleaned_text:
#                 raise ValueError("Cannot create embedding for empty text")
            
#             # Convert to embedding using Google's model
#             embedding = self.embeddings.embed_query(cleaned_text)
            
#             print(f"🔢 Created embedding for: '{text[:50]}...' → {len(embedding)} dimensions")
#             return embedding
            
#         except Exception as e:
#             print(f"❌ Error creating embedding: {e}")
#             # Return zero vector as fallback
#             return [0.0] * 768  # Standard embedding dimension
    
#     def create_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """
#         Create embeddings for multiple texts efficiently.
        
#         Batch processing is faster than creating embeddings one by one.
#         Useful for processing multiple conversation memories at once.
        
#         Args:
#             texts: List of texts to convert
            
#         Returns:
#             List[List[float]]: List of embedding vectors
#         """
#         try:
#             cleaned_texts = [text.strip() for text in texts if text.strip()]
            
#             if not cleaned_texts:
#                 return []
            
#             # Batch embedding creation
#             embeddings = self.embeddings.embed_documents(cleaned_texts)
            
#             print(f"🔢 Created {len(embeddings)} embeddings in batch")
#             return embeddings
            
#         except Exception as e:
#             print(f"❌ Error creating batch embeddings: {e}")
#             # Return zero vectors as fallback
#             return [[0.0] * 768 for _ in texts]


# class ChromaMemoryStore:
#     """
#     Manages ChromaDB vector storage for conversation memories.
    
#     This class handles all interactions with ChromaDB, which is our
#     vector database. Think of it as a specialized database that stores
#     and searches by meaning instead of exact text matches.
    
#     ChromaDB Concepts:
#     - Collection: Like a table, stores related vectors
#     - Document: The original text content
#     - Embedding: The vector representation of the document
#     - Metadata: Additional information about the document (conversation_id, etc.)
#     - Query with Filtering: Search for similar vectors within specific conversations
    
#     Conversation Scoping:
#     - All memories include conversation_id in metadata
#     - Searches can be filtered by conversation_id
#     - Ensures privacy boundaries between conversation threads
#     - Maintains predictable AI behavior within conversations
    
#     Responsibilities:
#     - Store conversation memories with metadata
#     - Search memories with conversation filtering
#     - Manage ChromaDB collections and persistence
#     - Provide conversation isolation and privacy
#     """
    
#     def __init__(self, collection_name: str = "conversation_memories"):
#         """
#         Initialize ChromaDB connection and collection.
        
#         Args:
#             collection_name: Name of the ChromaDB collection to use
#         """
#         self.collection_name = collection_name
        
#         try:
#             config = get_config()
            
#             # Create ChromaDB client with persistent storage
#             # This means our vectors survive application restarts!
#             self.client = chromadb.PersistentClient(
#                 path=str(config.database.chromadb_path),
#                 settings=Settings(
#                     anonymized_telemetry=False,  # Disable analytics
#                     is_persistent=True           # Save to disk
#                 )
#             )
            
#             # Get or create collection for our conversation memories
#             # A collection is like a table in traditional databases
#             self.collection = self.client.get_or_create_collection(
#                 name=collection_name,
#                 metadata={"description": "AI Assistant conversation memories for semantic search"}
#             )
            
#             print(f"✅ ChromaDB initialized: '{collection_name}' collection ready")
#             print(f"📊 Current collection size: {self.collection.count()} memories")
            
#         except Exception as e:
#             print(f"❌ Failed to initialize ChromaDB: {e}")
#             raise
    
#     def store_memory(self, memory_id: str, content: str, embedding: List[float], 
#                     metadata: Dict[str, Any]) -> None:
#         """
#         Store a memory in ChromaDB with conversation metadata.
        
#         This saves both the original text content and its vector representation,
#         plus metadata about when and where this memory came from, including
#         the conversation_id for scoping searches.
        
#         Args:
#             memory_id: Unique identifier for this memory
#             content: The text content to store
#             embedding: Vector representation of the content
#             metadata: Additional information (conversation_id, timestamp, etc.)
#         """
#         try:
#             # Store in ChromaDB
#             # ChromaDB needs 4 things: documents, embeddings, metadata, and IDs
#             self.collection.add(
#                 documents=[content],        # The original text
#                 embeddings=[embedding],     # The vector representation
#                 metadatas=[metadata],       # Extra info including conversation_id
#                 ids=[memory_id]            # Unique identifier
#             )
            
#             conv_id = metadata.get('conversation_id', 'unknown')[:8]
#             print(f"💾 Stored memory in {conv_id}...: {memory_id} → '{content[:50]}...'")
            
#         except Exception as e:
#             print(f"❌ Error storing memory in ChromaDB: {e}")
#             raise
    
#     def search_similar_memories(self, query_embedding: List[float], 
#                                conversation_id: Optional[str] = None,
#                                limit: int = 5, 
#                                min_similarity: float = 0.7) -> List[Dict[str, Any]]:
#         """
#         Search for memories similar to the query embedding within a conversation.
        
#         This searches for semantically similar content ONLY within the specified
#         conversation thread, maintaining conversation boundaries and privacy.
        
#         Args:
#             query_embedding: Vector representation of what we're looking for
#             conversation_id: Limit search to this conversation (None = search all)
#             limit: Maximum number of results to return
#             min_similarity: Minimum similarity score to include (0.0 to 1.0)
            
#         Returns:
#             List[Dict]: Similar memories with content and metadata
            
#         Conversation-Scoped Search Benefits:
#         - Maintains conversation privacy and boundaries
#         - Provides predictable, relevant context
#         - User controls what context AI can reference
#         - Prevents unexpected cross-conversation references
        
#         How Similarity Works:
#         - 1.0 = Identical meaning (very rare)
#         - 0.8-0.9 = Very similar (same topic, related concepts)
#         - 0.7-0.8 = Somewhat similar (related but different aspects)
#         - 0.5-0.7 = Loosely related (might be relevant)
#         - Below 0.5 = Probably not relevant
#         """
#         try:
#             # Build query parameters
#             query_params = {
#                 'query_embeddings': [query_embedding],
#                 'n_results': limit,
#                 'include': ['documents', 'metadatas', 'distances']
#             }
            
#             # Add conversation filter if specified
#             if conversation_id:
#                 query_params['where'] = {'conversation_id': conversation_id}
#                 print(f"🔍 Searching memories within conversation: {conversation_id[:8]}...")
#             else:
#                 print("🔍 Searching all memories (no conversation filter)...")
            
#             # Query ChromaDB with optional conversation filter
#             results = self.collection.query(**query_params)
            
#             # Process results and calculate similarity scores
#             memories = []
            
#             if results['ids'] and results['ids'][0]:  # Check if we got results
#                 for i in range(len(results['ids'][0])):
#                     # ChromaDB returns "distance" (lower = more similar)
#                     # We convert to "similarity" (higher = more similar)
#                     distance = results['distances'][0][i]
#                     similarity = max(0.0, 1.0 - distance)  # Convert distance to similarity
                    
#                     # Only include memories above similarity threshold
#                     if similarity >= min_similarity:
#                         memory = {
#                             'id': results['ids'][0][i],
#                             'content': results['documents'][0][i],
#                             'metadata': results['metadatas'][0][i],
#                             'similarity': similarity
#                         }
#                         memories.append(memory)
                        
#                         mem_conv = memory['metadata'].get('conversation_id', 'unknown')[:8]
#                         print(f"🔍 Found similar memory: {similarity:.3f} similarity in {mem_conv}... → '{memory['content'][:50]}...'")
            
#             scope_desc = f"within conversation {conversation_id[:8]}..." if conversation_id else "across all conversations"
#             print(f"🎯 Search complete: {len(memories)} relevant memories found {scope_desc}")
#             return memories
            
#         except Exception as e:
#             print(f"❌ Error searching ChromaDB: {e}")
#             return []
    
#     def get_collection_stats(self) -> Dict[str, Any]:
#         """
#         Get statistics about the memory collection.
        
#         Returns:
#             Dict: Collection statistics including conversation breakdown
#         """
#         try:
#             count = self.collection.count()
            
#             # Get sample of recent memories
#             sample_results = self.collection.get(limit=5)
#             sample_memories = []
#             conversation_ids = set()
            
#             if sample_results['documents']:
#                 for i, doc in enumerate(sample_results['documents']):
#                     metadata = sample_results['metadatas'][i] if sample_results['metadatas'] else {}
#                     conv_id = metadata.get('conversation_id', 'unknown')
#                     conversation_ids.add(conv_id)
                    
#                     sample_memories.append({
#                         'content': doc[:100] + '...' if len(doc) > 100 else doc,
#                         'metadata': metadata,
#                         'conversation_id': conv_id
#                     })
            
#             return {
#                 'total_memories': count,
#                 'collection_name': self.collection_name,
#                 'unique_conversations': len(conversation_ids),
#                 'sample_memories': sample_memories
#             }
            
#         except Exception as e:
#             print(f"❌ Error getting collection stats: {e}")
#             return {'total_memories': 0, 'error': str(e)}


# class SmartMemoryService:
#     """
#     Main service combining embedding generation and ChromaDB storage.
    
#     This is the high-level interface that combines all the memory components.
#     It handles the complete workflow: text → embeddings → storage → search,
#     with conversation scoping for privacy and predictable behavior.
    
#     Memory Workflow:
#     1. Store Phase: Text → Embedding → ChromaDB storage (with conversation_id)
#     2. Search Phase: Query → Embedding → ChromaDB search → Relevant memories (filtered by conversation_id)
    
#     Conversation Scoping Benefits:
#     - Each conversation thread maintains separate memory context
#     - No unexpected references to unrelated conversations  
#     - Predictable AI behavior similar to Claude/ChatGPT
#     - Privacy boundaries between different conversation topics
#     - User controls what context AI can access
    
#     Responsibilities:
#     - Coordinate embedding generation and vector storage
#     - Provide high-level memory operations (store, search)
#     - Maintain conversation boundaries and privacy
#     - Handle error cases and provide fallback behavior
#     - Generate meaningful conversation summaries for storage
#     """
    
#     def __init__(self):
#         """Initialize smart memory service with all components."""
#         try:
#             self.embedding_manager = EmbeddingManager()
#             self.memory_store = ChromaMemoryStore()
            
#             print("✅ Smart Memory Service initialized successfully")
            
#         except Exception as e:
#             print(f"❌ Failed to initialize Smart Memory Service: {e}")
#             raise
    
#     def store_conversation_memory(self, conversation_id: str, user_message: str, 
#                                 ai_response: str, input_type: str = "text") -> str:
#         """
#         Store a conversation exchange in semantic memory.
        
#         This method takes a user-AI conversation exchange and stores it in a way
#         that enables future semantic search within the same conversation thread.
#         It creates a meaningful summary, converts it to vectors, and stores it
#         in ChromaDB with conversation metadata.
        
#         Args:
#             conversation_id: ID of the conversation this exchange belongs to
#             user_message: What the user said
#             ai_response: How the AI responded
#             input_type: Type of user input ('text' or 'voice')
            
#         Returns:
#             str: Unique memory ID for this stored exchange
#         """
#         try:
#             # Create a meaningful summary of the conversation exchange
#             # This summary captures both what the user wanted and what was discussed
#             conversation_summary = self._create_conversation_summary(
#                 user_message, ai_response
#             )
            
#             # Generate unique ID for this memory
#             memory_id = f"memory_{conversation_id}_{int(datetime.now().timestamp())}"
            
#             # Convert summary to embedding vector
#             embedding = self.embedding_manager.create_embedding(conversation_summary)
            
#             # Create metadata to store alongside the content
#             # conversation_id is crucial for scoping searches
#             metadata = {
#                 'conversation_id': conversation_id,
#                 'timestamp': datetime.now().isoformat(),
#                 'user_message': user_message[:200],  # Truncated for metadata
#                 'input_type': input_type,
#                 'topics': self._extract_topics(user_message, ai_response)
#             }
            
#             # Store in ChromaDB
#             self.memory_store.store_memory(
#                 memory_id=memory_id,
#                 content=conversation_summary,
#                 embedding=embedding,
#                 metadata=metadata
#             )
            
#             conv_short = conversation_id[:8] + "..."
#             print(f"🧠 Stored conversation memory in {conv_short}: '{conversation_summary[:50]}...'")
#             return memory_id
            
#         except Exception as e:
#             print(f"❌ Error storing conversation memory: {e}")
#             return ""
    
#     def search_relevant_memories(self, query: str, conversation_id: Optional[str] = None,
#                                 limit: int = 5, min_similarity: float = 0.7) -> List[MemoryResult]:
#         """
#         Search for relevant memories based on a query within a conversation.
        
#         This searches for semantically similar content within the specified
#         conversation thread, maintaining conversation boundaries and context.
        
#         Args:
#             query: What to search for (user's current message or topic)
#             conversation_id: Limit search to this conversation thread
#             limit: Maximum number of memories to return
#             min_similarity: Minimum relevance score (0.7 = 70% similar)
            
#         Returns:
#             List[MemoryResult]: Relevant memories sorted by relevance
            
#         Example Usage:
#             # Search within current conversation only
#             memories = search_relevant_memories(
#                 "database optimization", 
#                 conversation_id="conv_123"
#             )
#             # Returns: Previous database discussions from conv_123 ONLY
#         """
#         try:
#             # Convert search query to embedding vector
#             query_embedding = self.embedding_manager.create_embedding(query)
            
#             # Search ChromaDB for similar vectors within conversation
#             raw_memories = self.memory_store.search_similar_memories(
#                 query_embedding=query_embedding,
#                 conversation_id=conversation_id,  # Scope to conversation
#                 limit=limit,
#                 min_similarity=min_similarity
#             )
            
#             # Convert raw results to structured MemoryResult objects
#             memory_results = []
#             for memory in raw_memories:
#                 result = MemoryResult(
#                     content=memory['content'],
#                     conversation_id=memory['metadata'].get('conversation_id', ''),
#                     timestamp=datetime.fromisoformat(memory['metadata'].get('timestamp', datetime.now().isoformat())),
#                     similarity_score=memory['similarity'],
#                     metadata=memory['metadata'],
#                     memory_type='conversation'
#                 )
#                 memory_results.append(result)
            
#             # Sort by relevance (highest similarity first)
#             memory_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
#             conv_desc = f"within conversation {conversation_id[:8]}..." if conversation_id else "across all conversations"
#             print(f"🎯 Found {len(memory_results)} relevant memories {conv_desc} for: '{query}'")
            
#             return memory_results
            
#         except Exception as e:
#             print(f"❌ Error searching relevant memories: {e}")
#             return []
    
#     def _create_conversation_summary(self, user_message: str, ai_response: str) -> str:
#         """
#         Create a searchable summary of a conversation exchange.
        
#         This method distills a user-AI exchange into a summary that captures
#         the key topics and concepts discussed. This summary is what gets
#         converted to embeddings and stored for semantic search.
        
#         Args:
#             user_message: User's input
#             ai_response: AI's response
            
#         Returns:
#             str: Meaningful summary for semantic search
#         """
#         # Extract key topics and concepts
#         topics = self._extract_topics(user_message, ai_response)
        
#         # Create a summary that captures the essence of the exchange
#         summary_parts = []
        
#         # Add user's intent/question
#         if len(user_message) <= 100:
#             summary_parts.append(f"User asked: {user_message}")
#         else:
#             summary_parts.append(f"User asked about: {user_message[:100]}...")
        
#         # Add key topics discussed
#         if topics:
#             summary_parts.append(f"Topics covered: {', '.join(topics)}")
        
#         # Add brief response summary
#         ai_summary = ai_response[:150] + "..." if len(ai_response) > 150 else ai_response
#         summary_parts.append(f"AI discussed: {ai_summary}")
        
#         return " | ".join(summary_parts)
    
#     def _extract_topics(self, user_message: str, ai_response: str) -> List[str]:
#         """
#         Extract key topics from conversation content.
        
#         This simple method identifies important keywords and concepts
#         from the conversation. In a more advanced version, we could use
#         NLP techniques or even AI to extract topics more intelligently.
        
#         Args:
#             user_message: User's message
#             ai_response: AI's response
            
#         Returns:
#             List[str]: List of identified topics
#         """
#         # Combine both messages for topic extraction
#         combined_text = f"{user_message} {ai_response}".lower()
        
#         # Common topic keywords (this could be expanded significantly)
#         topic_keywords = {
#             'programming': ['python', 'javascript', 'code', 'programming', 'development'],
#             'web_development': ['fastapi', 'django', 'flask', 'api', 'web', 'http', 'rest'],
#             'database': ['database', 'sql', 'sqlite', 'postgresql', 'query', 'table'],
#             'ai_ml': ['ai', 'machine learning', 'ml', 'model', 'neural network'],
#             'deployment': ['deploy', 'deployment', 'server', 'hosting', 'cloud'],
#             'testing': ['test', 'testing', 'pytest', 'unit test', 'debug']
#         }
        
#         # Find matching topics
#         found_topics = []
#         for topic, keywords in topic_keywords.items():
#             if any(keyword in combined_text for keyword in keywords):
#                 found_topics.append(topic)
        
#         return found_topics[:3]  # Limit to 3 most relevant topics
    
#     def get_memory_stats(self) -> Dict[str, Any]:
#         """
#         Get statistics about the memory system.
        
#         Returns:
#             Dict[str, Any]: Memory system statistics including conversation breakdown
#         """
#         try:
#             chroma_stats = self.memory_store.get_collection_stats()
            
#             return {
#                 'embedding_model': self.embedding_manager.model_name,
#                 'total_stored_memories': chroma_stats.get('total_memories', 0),
#                 'unique_conversations': chroma_stats.get('unique_conversations', 0),
#                 'collection_name': chroma_stats.get('collection_name', ''),
#                 'sample_memories': chroma_stats.get('sample_memories', []),
#                 'status': 'operational'
#             }
            
#         except Exception as e:
#             return {
#                 'status': 'error',
#                 'error': str(e),
#                 'total_stored_memories': 0,
#                 'unique_conversations': 0
#             }


# def init_smart_memory_service() -> SmartMemoryService:
#     """
#     Initialize smart memory service with all dependencies.
    
#     Returns:
#         SmartMemoryService: Configured memory service
        
#     Raises:
#         Exception: If initialization fails
#     """
#     try:
#         print("🚀 Initializing Smart Memory Service...")
        
#         memory_service = SmartMemoryService()
        
#         # Test the system to ensure it's working
#         stats = memory_service.get_memory_stats()
#         print(f"📊 Memory system ready: {stats['total_stored_memories']} memories across {stats.get('unique_conversations', 0)} conversations")
        
#         return memory_service
        
#     except Exception as e:
#         print(f"❌ Failed to initialize Smart Memory Service: {e}")
#         raise


# def test_smart_memory_system():
#     """
#     Test the complete smart memory system with conversation scoping.
    
#     This function demonstrates and tests all aspects of the conversation-scoped
#     semantic memory:
#     - Storing conversation memories with conversation IDs
#     - Converting text to embeddings
#     - Searching by semantic similarity within conversations
#     - Testing conversation isolation and privacy boundaries
#     - Retrieving relevant context from specific conversations
#     """
#     print("\n🧪 Testing Smart Memory System (Conversation-Scoped)...")
    
#     try:
#         # Test 1: Initialize memory service
#         print("\n1. Initializing memory service...")
#         memory_service = init_smart_memory_service()
        
#         # Test 2: Store some test conversations in different threads
#         print("\n2. Storing test conversation memories across different conversations...")
        
#         # Store FastAPI conversation in conv_1
#         memory_id_1 = memory_service.store_conversation_memory(
#             conversation_id="test_conv_1",
#             user_message="How do I create REST APIs with FastAPI?",
#             ai_response="FastAPI is excellent for creating REST APIs. You can define endpoints using decorators like @app.get('/items'). It provides automatic validation, serialization, and API documentation."
#         )
        
#         # Store Python learning conversation in conv_2
#         memory_id_2 = memory_service.store_conversation_memory(
#             conversation_id="test_conv_2",
#             user_message="What are Python decorators and how do they work?",
#             ai_response="Python decorators are functions that modify other functions. They use the @ syntax and are commonly used for logging, authentication, and caching. For example, @property turns a method into an attribute."
#         )
        
#         # Store database conversation in conv_3
#         memory_id_3 = memory_service.store_conversation_memory(
#             conversation_id="test_conv_3",
#             user_message="How do I optimize database queries in SQLite?",
#             ai_response="To optimize SQLite queries, use indexes on frequently queried columns, avoid SELECT *, use LIMIT for large results, and consider using EXPLAIN QUERY PLAN to analyze performance."
#         )
        
#         # Store another FastAPI conversation in conv_1 (same conversation as first)
#         memory_id_4 = memory_service.store_conversation_memory(
#             conversation_id="test_conv_1",  # Same conversation as memory_1
#             user_message="How do I handle authentication in FastAPI?",
#             ai_response="FastAPI supports various authentication methods including JWT tokens, OAuth2, and API keys. You can use dependencies to protect routes and validate tokens."
#         )
        
#         print(f"✅ Stored 4 test memories across 3 conversations:")
#         print(f"   {memory_id_1} (test_conv_1)")
#         print(f"   {memory_id_2} (test_conv_2)") 
#         print(f"   {memory_id_3} (test_conv_3)")
#         print(f"   {memory_id_4} (test_conv_1)")
        
#         # Test 3: Search for relevant memories (conversation-scoped)
#         print("\n3. Testing conversation-scoped semantic search...")
        
#         # Search within test_conv_1 only (should find FastAPI content)
#         print("\n  → Searching for 'web development' within test_conv_1:")
#         conv1_memories = memory_service.search_relevant_memories(
#             query="web development",
#             conversation_id="test_conv_1",  # Scope to specific conversation
#             limit=3,
#             min_similarity=0.5
#         )
        
#         for memory in conv1_memories:
#             print(f"    📝 {memory.similarity_score:.2f} similarity: {memory.content[:80]}...")
#             print(f"       From: {memory.conversation_id}")
        
#         # Search within test_conv_2 only (should find Python content)
#         print("\n  → Searching for 'Python concepts' within test_conv_2:")
#         conv2_memories = memory_service.search_relevant_memories(
#             query="Python concepts",
#             conversation_id="test_conv_2",  # Different conversation scope
#             limit=3,
#             min_similarity=0.5
#         )
        
#         for memory in conv2_memories:
#             print(f"    📝 {memory.similarity_score:.2f} similarity: {memory.content[:80]}...")
#             print(f"       From: {memory.conversation_id}")
        
#         # Test cross-conversation isolation
#         print("\n  → Testing conversation isolation:")
#         print("    Searching for 'FastAPI' within test_conv_2 (should find nothing):")
#         isolated_search = memory_service.search_relevant_memories(
#             query="FastAPI endpoints",
#             conversation_id="test_conv_2",  # Looking for FastAPI in Python conversation
#             limit=3,
#             min_similarity=0.5
#         )
        
#         if isolated_search:
#             print("    ❌ Found unexpected results - conversation isolation not working")
#             for memory in isolated_search:
#                 print(f"      📝 {memory.similarity_score:.2f}: {memory.content[:50]}...")
#         else:
#             print("    ✅ No results found - conversation isolation working correctly")
        
#         # Search within test_conv_3 only (database content)
#         print("\n  → Searching for 'optimization' within test_conv_3:")
#         conv3_memories = memory_service.search_relevant_memories(
#             query="optimization",
#             conversation_id="test_conv_3",
#             limit=3,
#             min_similarity=0.5
#         )
        
#         for memory in conv3_memories:
#             print(f"    📝 {memory.similarity_score:.2f} similarity: {memory.content[:80]}...")
#             print(f"       From: {memory.conversation_id}")
        
#         # Test multiple memories within same conversation
#         print("\n  → Testing multiple memories in same conversation (test_conv_1):")
#         fastapi_memories = memory_service.search_relevant_memories(
#             query="FastAPI development",
#             conversation_id="test_conv_1",
#             limit=5,
#             min_similarity=0.5
#         )
        
#         print(f"    Found {len(fastapi_memories)} FastAPI-related memories in test_conv_1:")
#         for memory in fastapi_memories:
#             print(f"    📝 {memory.similarity_score:.2f} similarity: {memory.content[:80]}...")
        
#         # Test 4: Get memory system statistics
#         print("\n4. Getting memory system statistics...")
#         stats = memory_service.get_memory_stats()
#         print(f"📊 Total memories stored: {stats['total_stored_memories']}")
#         print(f"🗂️ Unique conversations: {stats['unique_conversations']}")
#         print(f"🔤 Embedding model: {stats['embedding_model']}")
#         print(f"🗃️ Collection: {stats['collection_name']}")
        
#         print("\n✅ Conversation-scoped semantic search tests completed!")
#         print("🔒 Each conversation's memories are properly isolated")
#         print("🎯 Semantic search works within conversation boundaries")
#         print("🧠 The AI can now find relevant past context within each conversation!")
        
#     except Exception as e:
#         print(f"\n❌ Smart Memory System test failed: {e}")
#         raise


# if __name__ == "__main__":
#     """
#     Test smart memory system when run directly.
    
#     Usage:
#         pip install chromadb langchain langchain-google-genai
#         python src/services/memory_service.py
#     """
#     print("🧠 AI Assistant - Smart Memory System (Conversation-Scoped)")
#     print("=" * 60)
#     print("Testing semantic memory capabilities with conversation boundaries...")
#     print("=" * 60)
    
#     test_smart_memory_system()


"""
AI Assistant Memory Service

This module implements semantic memory storage and retrieval for conversations.
It combines embedding generation with vector database storage to enable
intelligent context retrieval based on semantic similarity.

Key Concepts:
- Semantic Memory: Long-term storage of conversation content as searchable embeddings
- Conversation-Scoped Memory: Each conversation maintains its own memory collection
- Hybrid Context: Combination of recent messages (short-term) and relevant past content (long-term)
- Memory Indexing: Automatic background processing of new messages into searchable vectors

Theory Background:
Traditional chatbots only remember the last few messages. Semantic memory enables:
- Finding relevant past discussions even from weeks ago
- Understanding context across conversation breaks
- Connecting related topics discussed at different times
- Providing richer, more informed responses

For example, if you discussed "FastAPI debugging" last month and now ask about 
"API error handling", the system can retrieve and use that previous context.

Usage Example:
    from services.memory_service import MemoryService, init_memory_service
    
    # Initialize memory service
    memory_service = init_memory_service()
    
    # Store message in semantic memory
    memory_service.store_message_memory(
        conversation_id="conv_123",
        message_id=42,
        user_message="How do I debug FastAPI?",
        ai_response="FastAPI debugging can be done using..."
    )
    
    # Retrieve relevant memories
    relevant_context = memory_service.get_relevant_memories(
        conversation_id="conv_123",
        query="API error handling"
    )

Architecture:
MemoryConfig - Configuration for memory storage and retrieval
MessageMemory - Data structure representing stored conversation memories
MemoryIndexer - Handles background indexing of new messages
SemanticMemoryStorage - Core storage operations with vector database
MemoryService - High-level interface for memory operations
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from pydantic import BaseModel, Field, ConfigDict

# Import our custom services
try:
    from .vector_database import (
        VectorDatabase, 
        ConversationCollection, 
        ConversationEmbedding,
        init_vector_database
    )
    from .embedding_service import (
        EmbeddingService, 
        EmbeddingResult, 
        init_embedding_service
    )
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from services.vector_database import (
            VectorDatabase, 
            ConversationCollection, 
            ConversationEmbedding,
            init_vector_database
        )
        from services.embedding_service import (
            EmbeddingService, 
            EmbeddingResult, 
            init_embedding_service
        )
    except ImportError:
        from src.services.vector_database import (
            VectorDatabase, 
            ConversationCollection, 
            ConversationEmbedding,
            init_vector_database
        )
        from src.services.embedding_service import (
            EmbeddingService, 
            EmbeddingResult, 
            init_embedding_service
        )

# Handle config import
try:
    from ..config import get_config
except ImportError:
    try:
        from config import get_config
    except ImportError:
        from src.config import get_config


class MemoryConfig(BaseModel):
    """
    Configuration settings for semantic memory operations.
    
    Defines parameters for memory storage, retrieval, and maintenance
    operations including similarity thresholds and indexing behavior.
    
    Attributes:
        similarity_threshold: Minimum cosine similarity for relevant memories (0.0-1.0)
        max_relevant_memories: Maximum number of memories to retrieve per query
        memory_retention_days: How long to keep memories before archival
        background_indexing: Whether to index new messages asynchronously
        batch_index_size: Number of messages to process in each indexing batch
        index_delay_seconds: Delay between background indexing operations
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score for considering memories relevant",
        ge=0.0,
        le=1.0
    )
    
    max_relevant_memories: int = Field(
        default=5,
        description="Maximum number of relevant memories to retrieve",
        gt=0,
        le=20
    )
    
    memory_retention_days: int = Field(
        default=90,
        description="Days to retain memories before archival consideration",
        gt=0,
        le=365
    )
    
    background_indexing: bool = Field(
        default=True,
        description="Enable background indexing of new messages"
    )
    
    batch_index_size: int = Field(
        default=10,
        description="Number of messages to process per indexing batch",
        gt=0,
        le=100
    )
    
    index_delay_seconds: float = Field(
        default=1.0,
        description="Delay between background indexing operations",
        ge=0.1,
        le=10.0
    )


@dataclass
class MessageMemory:
    """
    Represents a stored conversation memory with semantic context.
    
    Contains the original message content, its vector representation,
    and metadata for filtering and relevance scoring.
    
    Attributes:
        conversation_id: ID of the parent conversation
        message_id: Database ID of the original message
        user_message: Original user input text
        ai_response: Original AI response text
        combined_text: Concatenated text used for embedding generation
        embedding_id: Unique identifier for the stored embedding
        similarity_score: Relevance score when retrieved (0.0-1.0)
        timestamp: When this memory was created
        metadata: Additional context and filtering information
    
    Example:
        memory = MessageMemory(
            conversation_id="conv_123",
            message_id=42,
            user_message="How do I debug FastAPI?",
            ai_response="FastAPI debugging involves...",
            combined_text="User: How do I debug FastAPI? AI: FastAPI debugging involves...",
            embedding_id="emb_xyz789",
            similarity_score=0.85,
            timestamp=datetime.now(),
            metadata={"topic": "debugging", "framework": "fastapi"}
        )
    """
    
    conversation_id: str
    message_id: int
    user_message: str
    ai_response: str
    combined_text: str
    embedding_id: str
    similarity_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def get_context_summary(self) -> str:
        """
        Generate a brief summary of this memory for context inclusion.
        
        Returns:
            str: Formatted summary suitable for AI context
        """
        return f"Previous discussion: {self.user_message} | Response: {self.ai_response[:100]}..."
    
    def is_recent(self, hours: int = 24) -> bool:
        """
        Check if this memory is from recent conversation activity.
        
        Args:
            hours: Number of hours to consider "recent"
            
        Returns:
            bool: True if memory is within the specified time window
        """
        time_diff = datetime.now() - self.timestamp
        return time_diff.total_seconds() < (hours * 3600)


class MemoryIndexer:
    """
    Handles background indexing of conversation messages into semantic memory.
    
    Processes new messages asynchronously to avoid blocking conversation flow
    while maintaining up-to-date semantic search capabilities.
    """
    
    def __init__(self, embedding_service: EmbeddingService, 
                 vector_db: VectorDatabase, config: MemoryConfig):
        """
        Initialize memory indexer with required services.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_db: Vector database for storage
            config: Memory configuration settings
        """
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.config = config
        self.indexing_queue: List[Dict[str, Any]] = []
        self.indexing_lock = threading.Lock()
        self.indexing_thread: Optional[threading.Thread] = None
        self.stop_indexing = threading.Event()
        
        if config.background_indexing:
            self._start_background_indexing()
    
    def _start_background_indexing(self) -> None:
        """Start background thread for processing indexing queue."""
        self.indexing_thread = threading.Thread(
            target=self._background_indexing_loop,
            daemon=True
        )
        self.indexing_thread.start()
        print("Background memory indexing started")
    
    def _background_indexing_loop(self) -> None:
        """Main loop for background indexing operations."""
        while not self.stop_indexing.is_set():
            try:
                # Process pending indexing tasks
                self._process_indexing_queue()
                
                # Wait before next processing cycle
                time.sleep(self.config.index_delay_seconds)
                
            except Exception as e:
                print(f"Background indexing error: {e}")
                time.sleep(5.0)  # Longer delay on error
    
    def _process_indexing_queue(self) -> None:
        """Process pending messages in the indexing queue."""
        with self.indexing_lock:
            if not self.indexing_queue:
                return
            
            # Get batch of messages to process
            batch = self.indexing_queue[:self.config.batch_index_size]
            self.indexing_queue = self.indexing_queue[self.config.batch_index_size:]
        
        if batch:
            print(f"Processing {len(batch)} messages for memory indexing")
            
            for message_data in batch:
                try:
                    self._index_single_message(message_data)
                except Exception as e:
                    print(f"Error indexing message {message_data.get('message_id')}: {e}")
    
    def _index_single_message(self, message_data: Dict[str, Any]) -> None:
        """
        Index a single message into semantic memory storage.
        
        Args:
            message_data: Dictionary containing message information
        """
        # Create combined text for embedding
        combined_text = f"User: {message_data['user_message']} | AI: {message_data['ai_response']}"
        
        # Generate embedding
        embedding_result = self.embedding_service.create_embedding(combined_text)
        
        if not embedding_result.success:
            print(f"Failed to generate embedding for message {message_data['message_id']}: "
                  f"{embedding_result.error_message}")
            return
        
        # Store in vector database
        conversation_collection = self.vector_db.create_conversation_collection(
            message_data['conversation_id']
        )
        
        # Create embedding record
        embedding_id = f"mem_{message_data['conversation_id']}_{message_data['message_id']}"
        
        conversation_embedding = ConversationEmbedding(
            id=embedding_id,
            conversation_id=message_data['conversation_id'],
            message_id=message_data['message_id'],
            message_text=combined_text,
            embedding_vector=embedding_result.embedding,
            timestamp=datetime.now(),
            metadata={
                "user_message": message_data['user_message'],
                "ai_response": message_data['ai_response'],
                "indexed_at": datetime.now().isoformat(),
                "embedding_model": embedding_result.model_used
            }
        )
        
        conversation_collection.add_message_embedding(conversation_embedding)
        
        print(f"Indexed message {message_data['message_id']} into semantic memory")
    
    def queue_message_for_indexing(self, conversation_id: str, message_id: int,
                                  user_message: str, ai_response: str) -> None:
        """
        Add a message to the background indexing queue.
        
        Args:
            conversation_id: ID of the conversation
            message_id: Database ID of the message
            user_message: User's input text
            ai_response: AI's response text
        """
        message_data = {
            'conversation_id': conversation_id,
            'message_id': message_id,
            'user_message': user_message,
            'ai_response': ai_response,
            'queued_at': datetime.now()
        }
        
        with self.indexing_lock:
            self.indexing_queue.append(message_data)
        
        print(f"Queued message {message_id} for background indexing")
    
    def stop(self) -> None:
        """Stop background indexing and clean up resources."""
        if self.indexing_thread:
            self.stop_indexing.set()
            self.indexing_thread.join(timeout=5.0)
            print("Background memory indexing stopped")


class SemanticMemoryStorage:
    """
    Core storage operations for semantic memory using vector database.
    
    Provides low-level operations for storing and retrieving conversation
    memories with proper conversation scoping and similarity search.
    """
    
    def __init__(self, vector_db: VectorDatabase, embedding_service: EmbeddingService):
        """
        Initialize semantic memory storage with required services.
        
        Args:
            vector_db: Vector database for embeddings storage
            embedding_service: Service for generating query embeddings
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
    
    def store_message_embedding(self, conversation_id: str, message_id: int,
                               user_message: str, ai_response: str) -> bool:
        """
        Store a message exchange in semantic memory immediately.
        
        Args:
            conversation_id: ID of the conversation
            message_id: Database ID of the message
            user_message: User's input text
            ai_response: AI's response text
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            # Create combined text for embedding
            combined_text = f"User: {user_message} | AI: {ai_response}"
            
            # Generate embedding
            embedding_result = self.embedding_service.create_embedding(combined_text)
            
            if not embedding_result.success:
                print(f"Failed to generate embedding: {embedding_result.error_message}")
                return False
            
            # Get conversation collection
            conversation_collection = self.vector_db.create_conversation_collection(conversation_id)
            
            # Create embedding record
            embedding_id = f"mem_{conversation_id}_{message_id}"
            
            conversation_embedding = ConversationEmbedding(
                id=embedding_id,
                conversation_id=conversation_id,
                message_id=message_id,
                message_text=combined_text,
                embedding_vector=embedding_result.embedding,
                timestamp=datetime.now(),
                metadata={
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "stored_at": datetime.now().isoformat(),
                    "embedding_model": embedding_result.model_used
                }
            )
            
            # Store in vector database
            conversation_collection.add_message_embedding(conversation_embedding)
            
            print(f"Stored semantic memory for message {message_id}")
            return True
            
        except Exception as e:
            print(f"Error storing message embedding: {e}")
            return False
    
    def search_relevant_memories(self, conversation_id: str, query_text: str,
                                max_results: int = 5, 
                                similarity_threshold: float = 0.7) -> List[MessageMemory]:
        """
        Search for semantically relevant memories within a conversation.
        
        Args:
            conversation_id: ID of the conversation to search within
            query_text: Text to find similar memories for
            max_results: Maximum number of memories to return
            similarity_threshold: Minimum similarity score for relevance
            
        Returns:
            List[MessageMemory]: Relevant memories ordered by similarity
        """
        try:
            # Generate query embedding
            query_embedding_result = self.embedding_service.create_embedding(query_text)
            
            if not query_embedding_result.success:
                print(f"Failed to generate query embedding: {query_embedding_result.error_message}")
                return []
            
            # Get conversation collection
            conversation_collection = self.vector_db.create_conversation_collection(conversation_id)
            
            # Search for similar embeddings
            similar_embeddings = conversation_collection.search_similar_messages(
                query_embedding=query_embedding_result.embedding,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            # Convert to MessageMemory objects
            memories = []
            for embedding in similar_embeddings:
                metadata = embedding.metadata or {}
                
                memory = MessageMemory(
                    conversation_id=embedding.conversation_id,
                    message_id=embedding.message_id,
                    user_message=metadata.get('user_message', ''),
                    ai_response=metadata.get('ai_response', ''),
                    combined_text=embedding.message_text,
                    embedding_id=embedding.id,
                    similarity_score=metadata.get('similarity_score', 0.0),
                    timestamp=embedding.timestamp,
                    metadata=metadata
                )
                
                memories.append(memory)
            
            print(f"Found {len(memories)} relevant memories for query: '{query_text[:50]}...'")
            return memories
            
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []


class MemoryService:
    """
    High-level interface for semantic memory operations.
    
    Provides the main API for storing conversation messages in semantic memory
    and retrieving relevant context for enhanced AI responses. This service
    coordinates between embedding generation, vector storage, and background indexing.
    """
    
    def __init__(self, config: MemoryConfig, embedding_service: EmbeddingService,
                 vector_db: VectorDatabase):
        """
        Initialize memory service with configuration and dependencies.
        
        Args:
            config: Memory configuration settings
            embedding_service: Service for generating embeddings
            vector_db: Vector database for storage
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        
        # Initialize core components
        self.storage = SemanticMemoryStorage(vector_db, embedding_service)
        self.indexer = MemoryIndexer(embedding_service, vector_db, config)
        
        print("Memory service initialized with semantic storage and background indexing")
    
    def store_message_memory(self, conversation_id: str, message_id: int,
                            user_message: str, ai_response: str,
                            immediate: bool = False) -> bool:
        """
        Store a conversation message in semantic memory.
        
        By default, uses background indexing for performance. Can be forced
        to store immediately for critical messages or testing.
        
        Args:
            conversation_id: ID of the conversation
            message_id: Database ID of the message
            user_message: User's input text
            ai_response: AI's response text
            immediate: Whether to store immediately vs. background queue
            
        Returns:
            bool: True if storage was initiated successfully
            
        Example:
            success = memory_service.store_message_memory(
                conversation_id="conv_123",
                message_id=42,
                user_message="How do I debug FastAPI?",
                ai_response="FastAPI debugging involves checking logs...",
                immediate=False  # Use background indexing
            )
        """
        try:
            if immediate or not self.config.background_indexing:
                # Store immediately
                return self.storage.store_message_embedding(
                    conversation_id, message_id, user_message, ai_response
                )
            else:
                # Queue for background indexing
                self.indexer.queue_message_for_indexing(
                    conversation_id, message_id, user_message, ai_response
                )
                return True
                
        except Exception as e:
            print(f"Error storing message memory: {e}")
            return False
    
    def get_relevant_memories(self, conversation_id: str, query_text: str,
                             max_memories: Optional[int] = None,
                             similarity_threshold: Optional[float] = None) -> List[MessageMemory]:
        """
        Retrieve semantically relevant memories from a conversation.
        
        Searches the conversation's semantic memory for content similar
        to the query text, returning the most relevant past discussions.
        
        Args:
            conversation_id: ID of the conversation to search
            query_text: Text to find similar content for
            max_memories: Maximum memories to return (uses config default if None)
            similarity_threshold: Minimum similarity score (uses config default if None)
            
        Returns:
            List[MessageMemory]: Relevant memories ordered by similarity score
            
        Example:
            memories = memory_service.get_relevant_memories(
                conversation_id="conv_123",
                query_text="API error handling strategies"
            )
            
            for memory in memories:
                print(f"Relevant: {memory.get_context_summary()}")
                print(f"Similarity: {memory.similarity_score:.3f}")
        """
        max_results = max_memories or self.config.max_relevant_memories
        threshold = similarity_threshold or self.config.similarity_threshold
        
        return self.storage.search_relevant_memories(
            conversation_id=conversation_id,
            query_text=query_text,
            max_results=max_results,
            similarity_threshold=threshold
        )
    
    def get_conversation_memory_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics about semantic memory for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dict[str, Any]: Memory statistics including count and recent activity
        """
        try:
            conversation_collection = self.vector_db.create_conversation_collection(conversation_id)
            total_memories = conversation_collection.get_total_embeddings()
            
            return {
                "conversation_id": conversation_id,
                "total_memories": total_memories,
                "background_indexing_enabled": self.config.background_indexing,
                "similarity_threshold": self.config.similarity_threshold,
                "max_relevant_memories": self.config.max_relevant_memories
            }
            
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up resources and stop background processes."""
        if self.indexer:
            self.indexer.stop()
        print("Memory service cleanup completed")


def init_memory_service() -> MemoryService:
    """
    Initialize memory service with application configuration.
    
    Creates a fully configured MemoryService by initializing all required
    dependencies including embedding service and vector database.
    
    Returns:
        MemoryService: Configured memory service ready for use
        
    Raises:
        Exception: If service initialization fails
        
    Example:
        memory_service = init_memory_service()
        
        # Store a message in semantic memory
        memory_service.store_message_memory(
            conversation_id="conv_123",
            message_id=42,
            user_message="How do I debug FastAPI?",
            ai_response="FastAPI debugging involves..."
        )
        
        # Retrieve relevant memories
        relevant_memories = memory_service.get_relevant_memories(
            conversation_id="conv_123",
            query_text="API error handling"
        )
    """
    try:
        print("Initializing memory service with semantic capabilities...")
        
        # Get application configuration
        app_config = get_config()
        
        # Create memory configuration
        memory_config = MemoryConfig(
            similarity_threshold=0.7,  # Fairly strict similarity requirement
            max_relevant_memories=5,   # Reasonable number for context
            memory_retention_days=90,  # 3-month retention
            background_indexing=True,  # Enable async processing
            batch_index_size=10,       # Process in small batches
            index_delay_seconds=1.0    # Small delay between batches
        )
        
        # Initialize required services
        embedding_service = init_embedding_service()
        vector_db = init_vector_database()
        
        # Create memory service
        memory_service = MemoryService(
            config=memory_config,
            embedding_service=embedding_service,
            vector_db=vector_db
        )
        
        print("Memory service initialization completed successfully")
        return memory_service
        
    except Exception as e:
        print(f"Failed to initialize memory service: {e}")
        raise


def test_memory_service() -> None:
    """
    Test complete memory service functionality including storage and retrieval.
    
    Validates semantic memory operations with sample conversation data:
    - Message storage with immediate and background indexing
    - Semantic similarity search across stored memories
    - Memory statistics and metadata handling
    """
    print("\n🧪 Testing memory service functionality...")
    
    try:
        # Test 1: Initialize memory service
        print("\n1. Initializing memory service...")
        memory_service = init_memory_service()
        
        # Test conversation ID
        test_conv_id = "test_memory_conv_456"
        
        # Test 2: Store sample messages immediately
        print("\n2. Storing sample messages in semantic memory...")
        
        sample_messages = [
            {
                "message_id": 101,
                "user_message": "How do I debug FastAPI applications effectively?",
                "ai_response": "FastAPI debugging involves using logging, error handling, and testing tools. Start with proper logging configuration and use debuggers like pdb."
            },
            {
                "message_id": 102,
                "user_message": "What are the best practices for API error handling?",
                "ai_response": "API error handling should include proper HTTP status codes, structured error responses, and comprehensive logging for debugging purposes."
            },
            {
                "message_id": 103,
                "user_message": "How do I optimize database queries in Python?",
                "ai_response": "Database optimization includes using indexes, query analysis, connection pooling, and ORM best practices like eager loading."
            }
        ]
        
        for msg in sample_messages:
            success = memory_service.store_message_memory(
                conversation_id=test_conv_id,
                message_id=msg["message_id"],
                user_message=msg["user_message"],
                ai_response=msg["ai_response"],
                immediate=True  # Store immediately for testing
            )
            print(f"Stored message {msg['message_id']}: {'✅' if success else '❌'}")
        
        # Test 3: Search for relevant memories
        print("\n3. Testing semantic memory search...")
        
        search_queries = [
            "FastAPI debugging techniques",
            "handling API errors properly", 
            "database performance optimization"
        ]
        
        for query in search_queries:
            memories = memory_service.get_relevant_memories(
                conversation_id=test_conv_id,
                query_text=query
            )
            
            print(f"\nQuery: '{query}'")
            print(f"Found {len(memories)} relevant memories:")
            
            for memory in memories:
                print(f"  - Message {memory.message_id}: {memory.user_message[:50]}...")
                print(f"    Similarity: {memory.similarity_score:.3f}")
        
        # Test 4: Get memory statistics
        print("\n4. Getting memory statistics...")
        stats = memory_service.get_conversation_memory_stats(test_conv_id)
        print(f"Memory stats: {stats.get('total_memories', 0)} memories stored")
        print(f"Background indexing: {'✅ Enabled' if stats.get('background_indexing_enabled') else '❌ Disabled'}")
        
        # Test 5: Test background indexing
        print("\n5. Testing background indexing...")
        background_message = {
            "message_id": 104,
            "user_message": "How do I test FastAPI endpoints?",
            "ai_response": "FastAPI testing uses pytest and TestClient for comprehensive endpoint testing with fixtures and mocking."
        }
        
        success = memory_service.store_message_memory(
            conversation_id=test_conv_id,
            message_id=background_message["message_id"],
            user_message=background_message["user_message"],
            ai_response=background_message["ai_response"],
            immediate=False  # Use background indexing
        )
        
        print(f"Queued message for background indexing: {'✅' if success else '❌'}")
        
        # Wait a moment for background processing
        print("Waiting for background indexing...")
        time.sleep(3)
        
        # Check if message was indexed
        test_memories = memory_service.get_relevant_memories(
            conversation_id=test_conv_id,
            query_text="testing FastAPI"
        )
        
        background_found = any(m.message_id == 104 for m in test_memories)
        print(f"Background indexed message found: {'✅' if background_found else '❌'}")
        
        print("\n✅ All memory service tests passed!")
        
        # Cleanup
        memory_service.cleanup()
        
    except Exception as e:
        print(f"\n❌ Memory service test failed: {e}")
        raise


# Development utility - run this file directly to test memory service
if __name__ == "__main__":
    """
    Test memory service functionality when run directly.
    
    Usage:
        python src/services/memory_service.py
        
    This will test semantic memory storage, retrieval, background indexing,
    and similarity search capabilities with sample conversation data.
    """
    test_memory_service()