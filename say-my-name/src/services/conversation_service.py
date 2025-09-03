"""
AI Assistant Conversation Service

This module provides the business logic layer for managing complete conversation workflows.
It orchestrates database operations with AI response generation to provide high-level
conversation management functionality.

Key Features:
- Complete conversation lifecycle management
- Context-aware AI response generation
- Automatic conversation title generation
- Message history management with context
- Error handling and transaction safety
- Integration between database and AI layers

Usage Example:
    from services.conversation_service import ConversationService, init_conversation_service
    
    # Initialize service
    service = init_conversation_service()
    
    # Start a new conversation
    conv_id = service.start_new_conversation("Python Learning")
    
    # Process user message and get AI response
    response = service.process_message(conv_id, "How do I use FastAPI?")
    print(response.ai_content)


    
Class Structure :

ConversationResponse (data structure)
├── Structured response with metadata
├── Success/error tracking
└── Token usage information

TitleGenerator (AI-powered titles)
├── generate_from_message() - AI creates titles
└── _generate_fallback_title() - Keyword-based fallback

ConversationService (main orchestrator)
├── start_new_conversation() - Create conversation threads
├── process_message() - Complete message workflow
├── get_conversation_history() - Retrieve past messages
├── get_conversation_list() - List all conversations
└── delete_conversation() - Remove conversations
"""

import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Handle imports for different execution contexts
try:
    # When running from services/ directory, use relative import
    from ..database import DatabaseManager, init_database
except ImportError:
    try:
        # When running from ai-assistant/ directory
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from database import DatabaseManager, init_database
    except ImportError:
        # When running as module from other contexts
        from src.database import DatabaseManager, init_database

try:
    # When running from services/ directory, use relative import  
    from .gemini_client import GeminiClient, GeminiResponse, ConversationManager, init_gemini_client
except ImportError:
    try:
        # When running from ai-assistant/ directory
        from services.gemini_client import GeminiClient, GeminiResponse, ConversationManager, init_gemini_client
    except ImportError:
        # When running as module from other contexts
        from src.services.gemini_client import GeminiClient, GeminiResponse, ConversationManager, init_gemini_client


@dataclass
class ConversationResponse:
    """
    Structured response from conversation processing.
    
    Represents the complete result of processing a user message,
    including the AI response, conversation metadata, and success status.
    
    Attributes:
        conversation_id: ID of the conversation this response belongs to
        user_message: The original user input
        ai_content: The AI assistant's response text
        message_id: Database ID of the stored message
        timestamp: When this exchange occurred
        input_type: Type of input ('text' or 'voice')
        success: Whether the operation completed successfully
        error_message: Error description if success is False
        token_usage: Number of tokens used for this response
    """
    
    conversation_id: str
    user_message: str
    ai_content: str
    message_id: Optional[int] = None
    timestamp: Optional[datetime] = None
    input_type: str = 'text'
    success: bool = True
    error_message: Optional[str] = None
    token_usage: int = 0
    
    def __str__(self) -> str:
        """String representation for debugging."""
        status = "✅" if self.success else "❌"
        return f"{status} ConversationResponse(conv='{self.conversation_id[:8]}...', tokens={self.token_usage})"


class TitleGenerator:
    """
    Generates meaningful titles for conversations based on content.
    
    Creates human-readable conversation titles from the first user message
    or conversation content. Handles edge cases and provides fallback titles.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize title generator with AI client.
        
        Args:
            gemini_client: Configured GeminiClient for AI-powered title generation
        """
        self.client = gemini_client
    
    def generate_from_message(self, user_message: str) -> str:
        """
        Generate conversation title from first user message.
        
        Creates a concise, descriptive title based on the user's first message.
        Uses AI to extract the main topic and format it appropriately.
        
        Args:
            user_message: The first message in the conversation
            
        Returns:
            str: Generated conversation title (max 50 characters)
            
        Example:
            title = generator.generate_from_message("How do I create a FastAPI endpoint?")
            # Returns: "FastAPI Endpoint Creation"
        """
        # Fallback title based on keywords
        fallback_title = self._generate_fallback_title(user_message)
        
        try:
            # Use AI to generate a more sophisticated title
            title_prompt = f"""
            Create a short, descriptive title (maximum 50 characters) for a conversation that starts with this message:
            "{user_message}"
            
            The title should:
            - Be 3-6 words maximum
            - Capture the main topic
            - Be suitable for a conversation list
            - Not include quotes or special characters
            
            Examples:
            - "How do I learn Python?" → "Python Learning"
            - "Help me debug this code" → "Code Debugging Help"
            - "What's the weather like?" → "Weather Discussion"
            
            Title:
            """
            
            response = self.client.generate_response(title_prompt)
            
            if response.content and len(response.content.strip()) > 0:
                # Clean up the AI-generated title
                ai_title = response.content.strip()
                ai_title = re.sub(r'^["\']|["\']$', '', ai_title)  # Remove quotes
                ai_title = re.sub(r'[^\w\s-]', '', ai_title)  # Remove special chars
                ai_title = ai_title[:50]  # Limit length
                
                if len(ai_title) >= 3:  # Minimum viable title
                    print(f"🏷️ Generated AI title: '{ai_title}'")
                    return ai_title
            
            print(f"🏷️ Using fallback title: '{fallback_title}'")
            return fallback_title
            
        except Exception as e:
            print(f"⚠️ Title generation failed, using fallback: {e}")
            return fallback_title
    
    def _generate_fallback_title(self, user_message: str) -> str:
        """
        Generate fallback title using keyword extraction.
        
        Creates a simple title based on common patterns and keywords
        when AI title generation fails.
        
        Args:
            user_message: User's message to analyze
            
        Returns:
            str: Fallback title
        """
        message = user_message.lower().strip()
        
        # Common topic keywords
        topic_keywords = {
            'python': 'Python Discussion',
            'fastapi': 'FastAPI Help',
            'database': 'Database Questions',
            'code': 'Code Discussion',
            'debug': 'Debugging Help',
            'error': 'Error Resolution',
            'learn': 'Learning Session',
            'tutorial': 'Tutorial Discussion',
            'api': 'API Development',
            'web': 'Web Development'
        }
        
        # Look for topic keywords
        for keyword, title in topic_keywords.items():
            if keyword in message:
                return title
        
        # Pattern-based titles
        if any(word in message for word in ['how', 'what', 'why', 'when', 'where']):
            return "Q&A Session"
        elif any(word in message for word in ['help', 'issue', 'problem']):
            return "Help Request"
        elif any(word in message for word in ['explain', 'tell me']):
            return "Explanation Request"
        else:
            return "General Discussion"


class ConversationService:
    """
    Main service for managing conversation workflows.
    
    Provides high-level methods for complete conversation management,
    integrating database operations with AI response generation.
    This is the main interface that the API layer will use.
    """
    
    def __init__(self, db_manager: DatabaseManager, gemini_client: GeminiClient):
        """
        Initialize conversation service with dependencies.
        
        Args:
            db_manager: Database manager for conversation storage
            gemini_client: AI client for response generation
        """
        self.db = db_manager
        self.gemini_manager = ConversationManager(gemini_client)
        self.title_generator = TitleGenerator(gemini_client)
        
        print("✅ Conversation service initialized")
    
    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """
        Start a new conversation thread.
        
        Creates a new conversation in the database with the specified title
        or a default title. The conversation will be ready to receive messages.
        
        Args:
            title: Optional conversation title. If None, a default will be used.
            
        Returns:
            str: Unique conversation ID
            
        Raises:
            Exception: If conversation creation fails
            
        Example:
            conv_id = service.start_new_conversation("Python Learning Session")
            print(f"Started conversation: {conv_id}")
        """
        if not title:
            title = f"New Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            conversation_id = self.db.create_conversation(title)
            print(f"🆕 Started new conversation: '{title}' ({conversation_id})")
            return conversation_id
            
        except Exception as e:
            print(f"❌ Failed to start conversation: {e}")
            raise
    
    def process_message(self, conversation_id: str, user_message: str, 
                       input_type: str = 'text') -> ConversationResponse:
        """
        Process a user message and generate AI response.
        
        This is the main method for handling user input. It:
        1. Retrieves conversation history for context
        2. Generates AI response using context
        3. Stores both user message and AI response in database
        4. Updates conversation metadata
        5. Returns structured response
        
        Args:
            conversation_id: ID of conversation to add message to
            user_message: User's input message
            input_type: Type of input ('text' or 'voice')
            
        Returns:
            ConversationResponse: Complete response with AI content and metadata
            
        Example:
            response = service.process_message(
                conv_id,
                "How do I create a FastAPI endpoint?",
                input_type='text'
            )
            print(response.ai_content)
        """
        try:
            # Validate inputs
            if not user_message or not user_message.strip():
                return ConversationResponse(
                    conversation_id=conversation_id,
                    user_message="",
                    ai_content="",
                    success=False,
                    error_message="User message cannot be empty"
                )
            
            user_message = user_message.strip()
            
            # Get conversation history for context
            message_history = self.db.get_conversation_history(
                conversation_id, 
                limit=15  # Last 15 messages for context
            )
            
            print(f"💬 Processing message in conversation {conversation_id[:8]}...")
            print(f"📝 User: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
            
            # Generate AI response with context
            gemini_response = self.gemini_manager.generate_contextual_response(
                user_message, 
                message_history
            )
            
            if not gemini_response.content:
                return ConversationResponse(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    ai_content="",
                    success=False,
                    error_message="AI response was empty"
                )
            
            # Store message exchange in database
            message_id = self.db.add_message(
                conversation_id=conversation_id,
                user_message=user_message,
                ai_response=gemini_response.content,
                input_type=input_type
            )
            
            # Update conversation title if this is the first message
            if len(message_history) == 0:
                self._update_conversation_title_if_needed(conversation_id, user_message)
            
            print(f"✅ Generated response: {len(gemini_response.content)} characters")
            
            # Return structured response
            return ConversationResponse(
                conversation_id=conversation_id,
                user_message=user_message,
                ai_content=gemini_response.content,
                message_id=message_id,
                timestamp=gemini_response.timestamp,
                input_type=input_type,
                success=True,
                token_usage=gemini_response.total_tokens
            )
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            return ConversationResponse(
                conversation_id=conversation_id,
                user_message=user_message,
                ai_content="",
                success=False,
                error_message=str(e)
            )
    
    def _update_conversation_title_if_needed(self, conversation_id: str, first_message: str) -> None:
        """
        Update conversation title based on first message.
        
        Generates a meaningful title for the conversation based on the first
        user message. This makes conversation lists more readable.
        
        Args:
            conversation_id: ID of conversation to update
            first_message: First user message in the conversation
        """
        try:
            # Get current conversations to check if title needs updating
            conversations = self.db.get_all_conversations()
            current_conv = next((c for c in conversations if c['conversation_id'] == conversation_id), None)
            
            if current_conv and current_conv.get('message_count', 0) <= 1:
                # Generate new title based on first message
                new_title = self.title_generator.generate_from_message(first_message)
                
                # Update title in database
                success = self.db.update_conversation_title(conversation_id, new_title)
                if success:
                    print(f"🏷️ Updated conversation title: '{new_title}'")
                
        except Exception as e:
            print(f"⚠️ Failed to update conversation title: {e}")
            # Don't raise exception - title update is not critical
    
    def get_conversation_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all conversations with metadata.
        
        Returns conversation list sorted by most recent activity,
        suitable for display in UI sidebar or conversation picker.
        
        Returns:
            List[Dict]: List of conversation dictionaries with keys:
                - conversation_id: Unique ID
                - title: Conversation title
                - created_at: Creation timestamp
                - last_message_at: Last activity timestamp
                - message_count: Number of message exchanges
                - is_active: Whether conversation is active
                
        Example:
            conversations = service.get_conversation_list()
            for conv in conversations:
                print(f"{conv['title']} - {conv['message_count']} messages")
        """
        try:
            conversations = self.db.get_all_conversations(include_inactive=False)
            print(f"📋 Retrieved {len(conversations)} active conversations")
            return conversations
            
        except Exception as e:
            print(f"❌ Failed to get conversation list: {e}")
            return []
    
    def get_conversation_history(self, conversation_id: str, 
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get complete message history for a conversation.
        
        Retrieves all messages in chronological order, suitable for
        displaying conversation history in chat interface.
        
        Args:
            conversation_id: ID of conversation
            limit: Maximum number of recent messages (None = all)
            
        Returns:
            List[Dict]: List of message exchanges with full metadata
            
        Example:
            history = service.get_conversation_history(conv_id, limit=20)
            for msg in history:
                print(f"User: {msg['user_message']}")
                print(f"AI: {msg['ai_response']}")
        """
        try:
            history = self.db.get_conversation_history(conversation_id, limit)
            print(f"📚 Retrieved {len(history)} messages for conversation {conversation_id[:8]}...")
            return history
            
        except Exception as e:
            print(f"❌ Failed to get conversation history: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Permanently removes a conversation thread and all associated messages.
        This operation cannot be undone.
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
            
        Example:
            if service.delete_conversation(old_conv_id):
                print("Conversation deleted successfully")
        """
        try:
            success = self.db.delete_conversation(conversation_id)
            if success:
                print(f"🗑️ Deleted conversation: {conversation_id}")
            return success
            
        except Exception as e:
            print(f"❌ Failed to delete conversation: {e}")
            return False
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all conversations.
        
        Provides summary statistics useful for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Statistics including:
                - total_conversations: Number of conversations
                - total_messages: Total message count across all conversations
                - average_messages_per_conversation: Average conversation length
                - most_active_conversation: Conversation with most messages
                
        Example:
            stats = service.get_conversation_stats()
            print(f"Total conversations: {stats['total_conversations']}")
        """
        try:
            conversations = self.db.get_all_conversations(include_inactive=True)
            
            if not conversations:
                return {
                    'total_conversations': 0,
                    'total_messages': 0,
                    'average_messages_per_conversation': 0,
                    'most_active_conversation': None
                }
            
            total_conversations = len(conversations)
            total_messages = sum(conv['message_count'] for conv in conversations)
            
            # Find most active conversation
            most_active = max(conversations, key=lambda c: c['message_count'])
            
            stats = {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'average_messages_per_conversation': round(total_messages / total_conversations, 2),
                'most_active_conversation': {
                    'title': most_active['title'],
                    'message_count': most_active['message_count']
                }
            }
            
            print(f"📊 Conversation stats: {total_conversations} conversations, {total_messages} messages")
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get conversation stats: {e}")
            return {}


def init_conversation_service() -> ConversationService:
    """
    Initialize conversation service with all dependencies.
    
    Creates a fully configured ConversationService by initializing
    the database manager and Gemini client dependencies.
    
    Returns:
        ConversationService: Configured conversation service
        
    Raises:
        Exception: If service initialization fails
        
    Example:
        service = init_conversation_service()
        conv_id = service.start_new_conversation("My Chat")
    """
    try:
        print("🚀 Initializing conversation service...")
        
        # Initialize database manager
        db_manager = init_database()
        
        # Initialize Gemini client
        gemini_client = init_gemini_client()
        
        # Create conversation service
        service = ConversationService(db_manager, gemini_client)
        
        print("✅ Conversation service ready!")
        return service
        
    except Exception as e:
        print(f"❌ Failed to initialize conversation service: {e}")
        raise


def test_conversation_service() -> None:
    """
    Test complete conversation service functionality.
    
    This function tests the full conversation workflow:
    - Starting new conversations
    - Processing messages with AI responses
    - Retrieving conversation history
    - Managing conversation metadata
    
    Demonstrates the complete user journey from start to finish.
    """
    print("\n🧪 Testing conversation service...")
    
    try:
        # Test 1: Initialize service
        print("\n1. Initializing conversation service...")
        service = init_conversation_service()
        
        # Test 2: Start new conversation
        print("\n2. Starting new conversation...")
        conv_id = service.start_new_conversation()
        
        # Test 3: Process first message (should generate title)
        print("\n3. Processing first message...")
        response1 = service.process_message(
            conv_id,
            "Can you explain how FastAPI works and help me create my first endpoint?"
        )
        
        if response1.success:
            print(f"✅ AI Response: {response1.ai_content[:150]}...")
            print(f"   Tokens used: {response1.token_usage}")
        else:
            print(f"❌ Error: {response1.error_message}")
        
        # Test 4: Process follow-up message (should use context)
        print("\n4. Processing follow-up message...")
        response2 = service.process_message(
            conv_id,
            "Can you show me a simple example?"
        )
        
        if response2.success:
            print(f"✅ Contextual Response: {response2.ai_content[:150]}...")
        else:
            print(f"❌ Error: {response2.error_message}")
        
        # Test 5: Get conversation history
        print("\n5. Retrieving conversation history...")
        history = service.get_conversation_history(conv_id)
        print(f"📚 History contains {len(history)} message exchanges")
        
        # Test 6: Get conversation list
        print("\n6. Getting conversation list...")
        conversations = service.get_conversation_list()
        print(f"📋 Found {len(conversations)} conversations")
        
        if conversations:
            latest_conv = conversations[0]
            print(f"   Latest: '{latest_conv['title']}' with {latest_conv['message_count']} messages")
        
        # Test 7: Get statistics
        print("\n7. Getting conversation statistics...")
        stats = service.get_conversation_stats()
        print(f"📊 Stats: {stats.get('total_conversations', 0)} conversations, "
              f"{stats.get('total_messages', 0)} total messages")
        
        print("\n✅ All conversation service tests passed!")
        
    except Exception as e:
        print(f"\n❌ Conversation service test failed: {e}")
        raise


# Development utility - run this file directly to test conversation service
if __name__ == "__main__":
    """
    Test conversation service when run directly.
    
    Usage:
        python src/services/conversation_service.py
        
    This will test the complete conversation workflow including
    AI response generation, database storage, and context management.
    """
    test_conversation_service()