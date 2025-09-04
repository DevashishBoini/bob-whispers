#!/usr/bin/env python3
"""
Quick test to verify semantic search is now working.
"""

import sys
sys.path.append('src')

from services.memory_service import init_memory_service

def main():
    print("🎯 SEMANTIC SEARCH VERIFICATION TEST")
    print("=" * 50)
    
    # Initialize memory service
    memory_service = init_memory_service()
    
    # Test with our conversation ID from the test
    conversation_id = "conv_98a431fe"
    
    # Test queries that should now work
    test_queries = [
        "FastAPI error handling best practices",
        "JWT authentication implementation",
        "database connection pooling",
        "Python type hints",
        "debugging techniques"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        memories = memory_service.get_relevant_memories(
            conversation_id=conversation_id,
            query_text=query,
            max_memories=3
        )
        
        print(f"  ✅ Found {len(memories)} relevant memories")
        
        for i, memory in enumerate(memories):
            similarity = memory.similarity_score
            message_preview = memory.combined_text[:80].replace('\n', ' ')
            print(f"    {i+1}. Score: {similarity:.3f} - {message_preview}...")
    
    print(f"\n🎉 Semantic search is now working!")

if __name__ == "__main__":
    main()
