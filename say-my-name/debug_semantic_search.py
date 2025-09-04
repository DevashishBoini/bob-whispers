#!/usr/bin/env python3
"""
Debug script to diagnose semantic search issues.
"""

import sys
sys.path.append('src')

from services.memory_service import init_memory_service
from services.vector_database import init_vector_database

def main():
    print("🔍 SEMANTIC SEARCH DEBUG SCRIPT")
    print("=" * 50)
    
    # Initialize services
    print("Initializing memory service...")
    memory_service = init_memory_service()
    vector_db = init_vector_database()
    
    # Test with our conversation ID from the test
    conversation_id = "conv_98a431fe"
    
    print(f"\n📊 Checking conversation: {conversation_id}")
    
    # Get conversation collection
    try:
        conv_collection = vector_db.create_conversation_collection(conversation_id)
        total_embeddings = conv_collection.get_total_embeddings()
        print(f"✅ Total embeddings in collection: {total_embeddings}")
        
        if total_embeddings == 0:
            print("❌ No embeddings found! This explains why search returns 0 results.")
            return
            
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        return
    
    # Test embedding generation
    print(f"\n🧪 Testing embedding generation...")
    from services.embedding_service import init_embedding_service
    embedding_service = init_embedding_service()
    
    test_query = "FastAPI error handling"
    result = embedding_service.create_embedding(test_query)
    
    if not result.success:
        print(f"❌ Embedding generation failed: {result.error_message}")
        return
        
    print(f"✅ Generated embedding: {len(result.embedding)} dimensions")
    
    # Test raw ChromaDB search with very low threshold
    print(f"\n🔍 Testing raw ChromaDB search...")
    try:
        similar_embeddings = conv_collection.search_similar_messages(
            query_embedding=result.embedding,
            max_results=5,
            similarity_threshold=0.0  # Very low threshold
        )
        
        print(f"✅ Raw search found: {len(similar_embeddings)} results")
        
        for i, embedding in enumerate(similar_embeddings[:3]):
            similarity = embedding.metadata.get('similarity_score', 'Unknown')
            print(f"  {i+1}. Similarity: {similarity:.3f} - {embedding.message_text[:100]}...")
            
    except Exception as e:
        print(f"❌ Raw search failed: {e}")
        return
    
    # Test with different thresholds
    print(f"\n🎯 Testing different similarity thresholds...")
    for threshold in [0.0, 0.3, 0.5, 0.7, 0.9]:
        memories = memory_service.get_relevant_memories(
            conversation_id=conversation_id,
            query_text=test_query,
            max_memories=5,
            similarity_threshold=threshold
        )
        print(f"  Threshold {threshold:.1f}: {len(memories)} results")
    
    print(f"\n🏁 Debug complete!")

if __name__ == "__main__":
    main()
