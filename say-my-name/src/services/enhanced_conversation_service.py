"""
Enhanced AI Assistant Conversation Service with Semantic Memory

This module upgrades the existing conversation service with hybrid memory capabilities.
It combines short-term context (recent messages) with long-term semantic memory
to provide AI responses enriched with relevant past conversation content.

Key Concepts:
- Hybrid Memory: Combines recent messages + semantically relevant past content
- Context Enhancement: Enriches AI prompts with intelligent memory retrieval
- Smart Context Selection: Balances recent relevance with semantic similarity
- Memory-Aware Responses: AI can reference and build upon past discussions
- Seamless Integration: Maintains existing API while adding semantic capabilities

Theory Background:
Traditional conversation systems only use recent message history for context.
This hybrid approach adds "long-term memory" by:

1. Recent Context: Last 10-15 messages for immediate conversation flow
2. Semantic Context: Relevant past discussions found through similarity search
3. Intelligent Merging: Combines both types without overwhelming the AI
4. Progressive Enhancement: Gradually builds richer context over time

For example, if discussing "API testing" now, the system might recall a previous
detailed discussion about "FastAPI debugging" from weeks ago and include that
relevant context in the AI's prompt.

Usage Example:
    from services.enhanced_conversation_service import EnhancedConversationService
    
    # Initialize enhanced service
    service = init_enhanced_conversation_service()
    
    # Process message with semantic memory
    response = service.process_message_with_memory(
        conversation_id="conv_123",
        user_message="How do I handle API errors?",
        input_type="text"
    )
    
    # Response includes context from relevant past discussions
    print(response.ai_content)  # References previous API-related conversations

Architecture:
ContextConfig - Configuration for memory integration behavior
ContextBuilder - Constructs hybrid context from recent + semantic memories
EnhancedConversationService - Main service with semantic memory integration
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field, ConfigDict

# Import existing services
try:
    from .conversation_service import (
        ConversationService, 
        ConversationResponse, 
        TitleGenerator,
        init_conversation_service
    )
    from .memory_service import MemoryService, init_memory_service
    from .semantic_search import SemanticSearchEngine, SearchResult, init_search_engine
    from .gemini_client import GeminiClient, ConversationManager
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from services.conversation_service import (
            ConversationService, 
            ConversationResponse, 
            TitleGenerator,
            init_conversation_service
        )
        from services.memory_service import MemoryService, init_memory_service
        from services.semantic_search import SemanticSearchEngine, SearchResult, init_search_engine
        from services.gemini_client import GeminiClient, ConversationManager
    except ImportError:
        from src.services.conversation_service import (
            ConversationService, 
            ConversationResponse, 
            TitleGenerator,
            init_conversation_service
        )
        from src.services.memory_service import MemoryService, init_memory_service
        from src.services.semantic_search import SemanticSearchEngine, SearchResult, init_search_engine
        from src.services.gemini_client import GeminiClient, ConversationManager


class ContextConfig(BaseModel):
    """
    Configuration for hybrid context building and memory integration.
    
    Defines how recent messages and semantic memories are combined
    to create optimal context for AI responses.
    
    Attributes:
        recent_message_limit: Number of recent messages to include
        semantic_memory_limit: Number of semantic memories to retrieve
        min_semantic_similarity: Minimum similarity threshold for memories
        max_total_context_length: Maximum characters in combined context
        enable_context_summarization: Whether to summarize long contexts
        memory_weight_factor: Weight given to semantic memories vs recent messages
        include_memory_timestamps: Whether to include when memories occurred
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True
    )
    
    recent_message_limit: int = Field(
        default=10,
        description="Maximum number of recent messages to include in context",
        gt=0,
        le=50
    )
    
    semantic_memory_limit: int = Field(
        default=3,
        description="Maximum number of semantic memories to retrieve",
        gt=0,
        le=10
    )
    
    min_semantic_similarity: float = Field(
        default=0.75,
        description="Minimum similarity score for including semantic memories",
        ge=0.0,
        le=1.0
    )
    
    max_total_context_length: int = Field(
        default=4000,
        description="Maximum character length for total context",
        gt=100,
        le=10000
    )
    
    enable_context_summarization: bool = Field(
        default=True,
        description="Enable intelligent context summarization for long contexts"
    )
    
    memory_weight_factor: float = Field(
        default=0.8,
        description="Relative importance of semantic memories (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    include_memory_timestamps: bool = Field(
        default=True,
        description="Include timestamps for when memories occurred"
    )


@dataclass
class HybridContext:
    """
    Combined context from recent messages and semantic memories.
    
    Contains the merged context information used to enhance AI prompts
    along with metadata about the context composition.
    
    Attributes:
        recent_messages: List of recent message exchanges
        semantic_memories: List of relevant past memories
        formatted_context: Ready-to-use context string for AI
        total_length: Character count of formatted context
        memory_count: Number of semantic memories included
        recent_count: Number of recent messages included
        context_summary: Brief summary of context composition
    """
    
    recent_messages: List[Dict[str, Any]]
    semantic_memories: List[SearchResult]
    formatted_context: str
    total_length: int
    memory_count: int
    recent_count: int
    context_summary: str


class ContextBuilder:
    """
    Constructs hybrid context from recent messages and semantic memories.
    
    Intelligently combines immediate conversation context with relevant
    past discussions to create enriched context for AI responses.
    """
    
    def __init__(self, config: ContextConfig):
        """
        Initialize context builder with configuration.
        
        Args:
            config: ContextConfig with context building parameters
        """
        self.config = config
    
    def build_hybrid_context(self, conversation_id: str, current_message: str,
                           recent_messages: List[Dict[str, Any]],
                           search_engine: SemanticSearchEngine) -> HybridContext:
        """
        Build hybrid context combining recent messages with semantic memories.
        
        Args:
            conversation_id: ID of the current conversation
            current_message: User's current input message
            recent_messages: Recent message history from database
            search_engine: Semantic search engine for memory retrieval
            
        Returns:
            HybridContext: Combined context ready for AI prompt
            
        Example:
            context = builder.build_hybrid_context(
                conversation_id="conv_123",
                current_message="How do I handle API timeouts?",
                recent_messages=[...],
                search_engine=search_engine
            )
            
            # Context includes recent chat + relevant past API discussions
            ai_prompt = f"Context: {context.formatted_context}\n\nUser: {current_message}"
        """
        # Step 1: Get relevant semantic memories
        semantic_results = search_engine.search_memories(
            conversation_id=conversation_id,
            query=current_message,
            max_results=self.config.semantic_memory_limit,
            include_context_analysis=True
        )
        
        # Filter by similarity threshold
        relevant_memories = [
            result for result in semantic_results 
            if result.relevance_score >= self.config.min_semantic_similarity
        ]
        
        # Step 2: Prepare recent message context
        recent_context = self._format_recent_messages(
            recent_messages[:self.config.recent_message_limit]
        )
        
        # Step 3: Prepare semantic memory context
        memory_context = self._format_semantic_memories(relevant_memories)
        
        # Step 4: Combine contexts intelligently
        formatted_context = self._combine_contexts(recent_context, memory_context)
        
        # Step 5: Apply length limits and summarization if needed
        final_context = self._apply_context_limits(formatted_context)
        
        # Step 6: Generate context summary
        context_summary = self._generate_context_summary(
            len(recent_messages), len(relevant_memories)
        )
        
        return HybridContext(
            recent_messages=recent_messages[:self.config.recent_message_limit],
            semantic_memories=relevant_memories,
            formatted_context=final_context,
            total_length=len(final_context),
            memory_count=len(relevant_memories),
            recent_count=min(len(recent_messages), self.config.recent_message_limit),
            context_summary=context_summary
        )
    
    def _format_recent_messages(self, recent_messages: List[Dict[str, Any]]) -> str:
        """
        Format recent messages for context inclusion.
        
        Args:
            recent_messages: List of recent message dictionaries
            
        Returns:
            str: Formatted recent message context
        """
        if not recent_messages:
            return ""
        
        formatted_lines = []
        formatted_lines.append("=== Recent Conversation ===")
        
        for msg in recent_messages:
            user_msg = msg.get('user_message', '').strip()
            ai_msg = msg.get('ai_response', '').strip()
            
            if user_msg and ai_msg:
                formatted_lines.append(f"User: {user_msg}")
                formatted_lines.append(f"Assistant: {ai_msg}")
                formatted_lines.append("")  # Blank line for readability
        
        return "\n".join(formatted_lines)
    
    def _format_semantic_memories(self, memories: List[SearchResult]) -> str:
        """
        Format semantic memories for context inclusion.
        
        Args:
            memories: List of SearchResult objects from semantic search
            
        Returns:
            str: Formatted semantic memory context
        """
        if not memories:
            return ""
        
        formatted_lines = []
        formatted_lines.append("=== Relevant Past Discussions ===")
        
        for i, result in enumerate(memories, 1):
            memory = result.memory
            
            # Add timestamp if configured
            timestamp_info = ""
            if self.config.include_memory_timestamps:
                time_ago = self._get_relative_time(memory.timestamp)
                timestamp_info = f" ({time_ago})"
            
            formatted_lines.append(
                f"[Past Discussion {i}{timestamp_info} - Relevance: {result.relevance_score:.2f}]"
            )
            formatted_lines.append(f"User: {memory.user_message}")
            formatted_lines.append(f"Assistant: {memory.ai_response}")
            formatted_lines.append("")  # Blank line for readability
        
        return "\n".join(formatted_lines)
    
    def _combine_contexts(self, recent_context: str, memory_context: str) -> str:
        """
        Intelligently combine recent and semantic contexts.
        
        Args:
            recent_context: Formatted recent message context
            memory_context: Formatted semantic memory context
            
        Returns:
            str: Combined context string
        """
        context_parts = []
        
        # Add semantic memories first (older, background context)
        if memory_context:
            context_parts.append(memory_context)
        
        # Add recent messages (immediate context)
        if recent_context:
            context_parts.append(recent_context)
        
        # Add instructional context for AI
        if memory_context and recent_context:
            instruction = (
                "=== Instructions ===\n"
                "Use both the recent conversation and past discussions to provide "
                "a contextual, informed response. Reference relevant past information "
                "when helpful, but prioritize the current conversation flow.\n"
            )
            context_parts.insert(-1, instruction)  # Insert before recent context
        
        return "\n".join(context_parts)
    
    def _apply_context_limits(self, context: str) -> str:
        """
        Apply length limits and summarization to context.
        
        Args:
            context: Raw combined context string
            
        Returns:
            str: Context within configured length limits
        """
        if len(context) <= self.config.max_total_context_length:
            return context
        
        if self.config.enable_context_summarization:
            # Simple truncation with ellipsis (more sophisticated summarization could be added)
            max_length = self.config.max_total_context_length - 50  # Reserve space for ellipsis
            truncated = context[:max_length]
            
            # Try to truncate at sentence boundary
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.8:  # If period is reasonably close to end
                truncated = truncated[:last_period + 1]
            
            return truncated + "\n\n[... context truncated for length ...]"
        else:
            # Simple truncation
            return context[:self.config.max_total_context_length]
    
    def _get_relative_time(self, timestamp: datetime) -> str:
        """
        Get human-readable relative time string.
        
        Args:
            timestamp: Datetime to convert to relative time
            
        Returns:
            str: Human-readable time description
        """
        now = datetime.now()
        diff = now - timestamp
        
        hours = diff.total_seconds() / 3600
        
        if hours < 1:
            return "less than an hour ago"
        elif hours < 24:
            return f"{int(hours)} hour{'s' if hours > 1 else ''} ago"
        elif hours < 48:
            return "yesterday"
        elif hours < 168:  # 1 week
            days = int(hours / 24)
            return f"{days} day{'s' if days > 1 else ''} ago"
        else:
            weeks = int(hours / 168)
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    
    def _generate_context_summary(self, recent_count: int, memory_count: int) -> str:
        """
        Generate summary of context composition.
        
        Args:
            recent_count: Number of recent messages included
            memory_count: Number of semantic memories included
            
        Returns:
            str: Context composition summary
        """
        parts = []
        
        if recent_count > 0:
            parts.append(f"{recent_count} recent message{'s' if recent_count > 1 else ''}")
        
        if memory_count > 0:
            parts.append(f"{memory_count} relevant past discussion{'s' if memory_count > 1 else ''}")
        
        if not parts:
            return "No context available"
        
        return "Context includes: " + " and ".join(parts)


class EnhancedConversationService:
    """
    Enhanced conversation service with hybrid semantic memory integration.
    
    Extends the base conversation service with semantic memory capabilities,
    providing AI responses enriched with relevant past conversation content.
    """
    
    def __init__(self, base_service: ConversationService, 
                 memory_service: MemoryService,
                 search_engine: SemanticSearchEngine,
                 config: ContextConfig):
        """
        Initialize enhanced conversation service.
        
        Args:
            base_service: Base ConversationService for core functionality
            memory_service: MemoryService for semantic memory storage
            search_engine: SemanticSearchEngine for memory retrieval
            config: ContextConfig for memory integration behavior
        """
        self.base_service = base_service
        self.memory_service = memory_service
        self.search_engine = search_engine
        self.config = config
        self.context_builder = ContextBuilder(config)
        
        print("Enhanced conversation service initialized with semantic memory")
    
    def process_message(self, conversation_id: str, user_message: str,
                      input_type: str = 'text') -> ConversationResponse:
        """
        Process user message with hybrid memory context enhancement.
        
        This method maintains API compatibility with the base ConversationService
        while providing enhanced responses with semantic memory integration.
        
        Args:
            conversation_id: ID of conversation
            user_message: User's input message
            input_type: Type of input ('text' or 'voice')
            
        Returns:
            ConversationResponse: Enhanced response with memory context
        """
        return self.process_message_with_memory(conversation_id, user_message, input_type)

    def process_message_with_memory(self, conversation_id: str, user_message: str,
                                  input_type: str = 'text') -> ConversationResponse:
        """
        Process user message with hybrid memory context enhancement.
        
        This is the main method that combines recent conversation history
        with relevant semantic memories to provide contextually rich AI responses.
        
        Args:
            conversation_id: ID of conversation
            user_message: User's input message
            input_type: Type of input ('text' or 'voice')
            
        Returns:
            ConversationResponse: Enhanced response with memory context
            
        Example:
            response = service.process_message_with_memory(
                conversation_id="conv_123",
                user_message="What were those FastAPI debugging tips?",
                input_type="text"
            )
            
            # Response might reference specific debugging discussion from weeks ago
            print(response.ai_content)  # Includes context from past FastAPI conversations
        """
        try:
            # Step 1: Get recent conversation history
            recent_messages = self.base_service.db.get_conversation_history(
                conversation_id,
                limit=self.config.recent_message_limit * 2  # Get extra for filtering
            )
            
            # Step 2: Build hybrid context with semantic memories
            hybrid_context = self.context_builder.build_hybrid_context(
                conversation_id=conversation_id,
                current_message=user_message,
                recent_messages=recent_messages,
                search_engine=self.search_engine
            )
            
            print(f"Built hybrid context: {hybrid_context.context_summary}")
            print(f"Context length: {hybrid_context.total_length} characters")
            
            # Step 3: Generate AI response with enhanced context
            enhanced_response = self._generate_enhanced_response(
                user_message, hybrid_context
            )
            
            if not enhanced_response.content:
                # Fallback to base service if enhanced generation fails
                print("Enhanced response failed, falling back to base service")
                return self.base_service.process_message(
                    conversation_id, user_message, input_type
                )
            
            # Step 4: Store the exchange in database
            message_id = self.base_service.db.add_message(
                conversation_id=conversation_id,
                user_message=user_message,
                ai_response=enhanced_response.content,
                input_type=input_type
            )
            
            # Step 5: Store in semantic memory (background)
            self.memory_service.store_message_memory(
                conversation_id=conversation_id,
                message_id=message_id,
                user_message=user_message,
                ai_response=enhanced_response.content,
                immediate=False  # Use background indexing
            )
            
            # Step 6: Update conversation title if needed
            if len(recent_messages) == 0:
                self.base_service._update_conversation_title_if_needed(
                    conversation_id, user_message
                )
            
            # Step 7: Create enhanced response object
            return ConversationResponse(
                conversation_id=conversation_id,
                user_message=user_message,
                ai_content=enhanced_response.content,
                message_id=message_id,
                timestamp=enhanced_response.timestamp,
                input_type=input_type,
                success=True,
                token_usage=enhanced_response.total_tokens
            )
            
        except Exception as e:
            print(f"Error in enhanced message processing: {e}")
            # Fallback to base service
            return self.base_service.process_message(
                conversation_id, user_message, input_type
            )
    
    def _generate_enhanced_response(self, user_message: str, 
                                  context: HybridContext):
        """
        Generate AI response using hybrid context.
        
        Args:
            user_message: User's input message
            context: HybridContext with combined recent + semantic context
            
        Returns:
            GeminiResponse: AI response with enhanced context
        """
        # Build enhanced prompt with hybrid context
        enhanced_prompt_parts = []
        
        # Add context if available
        if context.formatted_context:
            enhanced_prompt_parts.append(context.formatted_context)
            enhanced_prompt_parts.append("=== Current User Message ===")
        
        enhanced_prompt_parts.append(user_message)
        
        enhanced_prompt = "\n".join(enhanced_prompt_parts)
        
        # Generate response using the base gemini manager
        return self.base_service.gemini_manager.client.generate_response(enhanced_prompt)
    
    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """
        Start new conversation (delegates to base service).
        
        Args:
            title: Optional conversation title
            
        Returns:
            str: New conversation ID
        """
        return self.base_service.start_new_conversation(title)
    
    def get_conversation_list(self) -> List[Dict[str, Any]]:
        """
        Get conversation list (delegates to base service).
        
        Returns:
            List[Dict]: Conversation list with metadata
        """
        return self.base_service.get_conversation_list()
    
    def get_conversation_history(self, conversation_id: str, 
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history (delegates to base service).
        
        Args:
            conversation_id: Conversation ID
            limit: Optional message limit
            
        Returns:
            List[Dict]: Message history
        """
        return self.base_service.get_conversation_history(conversation_id, limit)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete conversation (delegates to base service).
        
        Args:
            conversation_id: Conversation ID to delete
            
        Returns:
            bool: True if successful
        """
        return self.base_service.delete_conversation(conversation_id)
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """
        Get enhanced statistics including memory usage.
        
        Returns:
            Dict[str, Any]: Enhanced statistics
        """
        base_stats = self.base_service.get_conversation_stats()
        
        # Add memory statistics
        base_stats.update({
            "semantic_memory_enabled": True,
            "context_config": {
                "recent_message_limit": self.config.recent_message_limit,
                "semantic_memory_limit": self.config.semantic_memory_limit,
                "min_similarity": self.config.min_semantic_similarity,
                "max_context_length": self.config.max_total_context_length
            }
        })
        
        return base_stats
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.memory_service:
            self.memory_service.cleanup()
        print("Enhanced conversation service cleanup completed")


def init_enhanced_conversation_service() -> EnhancedConversationService:
    """
    Initialize enhanced conversation service with all dependencies.
    
    Creates a fully integrated service combining base conversation functionality
    with semantic memory capabilities for enriched AI responses.
    
    Returns:
        EnhancedConversationService: Complete service with memory integration
        
    Example:
        service = init_enhanced_conversation_service()
        
        # Create conversation
        conv_id = service.start_new_conversation("Technical Discussion")
        
        # Process messages with memory enhancement
        response = service.process_message_with_memory(
            conversation_id=conv_id,
            user_message="How do I optimize API performance?",
            input_type="text"
        )
    """
    try:
        print("Initializing enhanced conversation service with semantic memory...")
        
        # Initialize base dependencies
        base_service = init_conversation_service()
        memory_service = init_memory_service()
        search_engine = init_search_engine(memory_service)
        
        # Create context configuration
        context_config = ContextConfig(
            recent_message_limit=10,        # Include last 10 exchanges
            semantic_memory_limit=3,        # Up to 3 relevant memories
            min_semantic_similarity=0.75,   # High similarity threshold
            max_total_context_length=4000,  # Reasonable context size
            enable_context_summarization=True,
            memory_weight_factor=0.8,       # High importance on memories
            include_memory_timestamps=True  # Show when memories occurred
        )
        
        # Create enhanced service
        enhanced_service = EnhancedConversationService(
            base_service=base_service,
            memory_service=memory_service,
            search_engine=search_engine,
            config=context_config
        )
        
        print("Enhanced conversation service initialization completed successfully")
        return enhanced_service
        
    except Exception as e:
        print(f"Failed to initialize enhanced conversation service: {e}")
        raise


def test_enhanced_conversation_service() -> None:
    """
    Test enhanced conversation service with semantic memory integration.
    
    Validates the complete hybrid memory workflow including context building,
    memory retrieval, and enhanced AI response generation.
    """
    print("\n🧪 Testing enhanced conversation service functionality...")
    
    try:
        # Test 1: Initialize enhanced service
        print("\n1. Initializing enhanced conversation service...")
        service = init_enhanced_conversation_service()
        
        # Test 2: Create conversation
        print("\n2. Creating test conversation...")
        conv_id = service.start_new_conversation("FastAPI Development Chat")
        print(f"Created conversation: {conv_id}")
        
        # Test 3: Build conversation history
        print("\n3. Building conversation history...")
        
        test_messages = [
            "How do I create FastAPI endpoints with proper error handling?",
            "What are the best practices for FastAPI middleware?",
            "How do I optimize database queries in FastAPI applications?",
            "Can you explain FastAPI dependency injection?"
        ]
        
        responses = []
        for i, message in enumerate(test_messages, 1):
            print(f"\n   Processing message {i}: '{message[:50]}...'")
            
            response = service.process_message_with_memory(
                conversation_id=conv_id,
                user_message=message,
                input_type="text"
            )
            
            if response.success:
                responses.append(response)
                print(f"   Response length: {len(response.ai_content)} characters")
                print(f"   Tokens used: {response.token_usage}")
            else:
                print(f"   Error: {response.error_message}")
        
        # Test 4: Test semantic memory retrieval
        print("\n4. Testing semantic memory with follow-up question...")
        
        followup_message = "What were those error handling techniques you mentioned earlier?"
        followup_response = service.process_message_with_memory(
            conversation_id=conv_id,
            user_message=followup_message,
            input_type="text"
        )
        
        if followup_response.success:
            print(f"Follow-up response successfully generated")
            print(f"Response preview: {followup_response.ai_content[:200]}...")
            
            # Check if response seems to reference past content
            has_context = any(
                keyword in followup_response.ai_content.lower()
                for keyword in ['mentioned', 'discussed', 'earlier', 'previously', 'before']
            )
            print(f"Contains contextual references: {'✅' if has_context else '❌'}")
        else:
            print(f"Follow-up failed: {followup_response.error_message}")
        
        # Test 5: Get enhanced statistics
        print("\n5. Getting enhanced service statistics...")
        stats = service.get_enhanced_stats()
        print(f"Total conversations: {stats.get('total_conversations', 0)}")
        print(f"Total messages: {stats.get('total_messages', 0)}")
        print(f"Semantic memory enabled: {'✅' if stats.get('semantic_memory_enabled') else '❌'}")
        
        context_config = stats.get('context_config', {})
        print(f"Context configuration:")
        print(f"  Recent messages: {context_config.get('recent_message_limit')}")
        print(f"  Semantic memories: {context_config.get('semantic_memory_limit')}")
        print(f"  Similarity threshold: {context_config.get('min_similarity')}")
        
        print("\n✅ All enhanced conversation service tests passed!")
        
        # Cleanup
        service.cleanup()
        
    except Exception as e:
        print(f"\n❌ Enhanced conversation service test failed: {e}")
        raise


# Development utility - run this file directly to test enhanced service
if __name__ == "__main__":
    """
    Test enhanced conversation service when run directly.
    
    Usage:
        python src/services/enhanced_conversation_service.py
        
    This will test the complete hybrid memory integration including
    context building, semantic memory retrieval, and enhanced AI responses.
    """
    test_enhanced_conversation_service()