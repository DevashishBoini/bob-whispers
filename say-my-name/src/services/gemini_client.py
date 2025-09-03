"""
AI Assistant Gemini Client

This module provides integration with Google's Gemini API for AI conversations.
Handles API authentication, request formatting, response processing, and error handling.

Key Features:
- Type-safe Gemini API integration
- Automatic retry logic for transient failures
- Rate limiting to respect API quotas
- Context-aware conversation handling
- Comprehensive error handling and logging
- Support for different Gemini models

Usage Example:
    from services.gemini_client import GeminiClient, init_gemini_client
    
    # Initialize client
    client = init_gemini_client()
    
    # Generate response
    response = client.generate_response(
        user_message="How do I use FastAPI?",
        conversation_context=["Previous message 1", "Previous message 2"]
    )
    print(response.content)

Class Structure :

GeminiResponse (data structure)
├── content - AI response text
├── model_used - Which Gemini model
├── token_usage - Cost tracking
└── safety_ratings - Content safety

RateLimiter (API quota management)
├── can_make_request() - Check if under limits
├── wait_if_needed() - Automatic delay
└── record_request() - Track usage

GeminiClient (main API client)
├── generate_response() - Get AI response
├── test_connection() - Verify API works
└── _build_context_prompt() - Format messages

ConversationManager (context handling)
├── format_conversation_context() - Prepare history
└── generate_contextual_response() - Main conversation method
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
from functools import wraps

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerateContentResponse
from google.api_core.exceptions import (
    ResourceExhausted, 
    InvalidArgument, 
    PermissionDenied,
    ServiceUnavailable,
    DeadlineExceeded
)

# Handle imports for different execution contexts
try:
    # When running from services/ directory, use relative import
    from ..config import get_config
except ImportError:
    try:
        # When running from ai-assistant/ directory
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from config import get_config
    except ImportError:
        # When running as module from other contexts
        from src.config import get_config


@dataclass
class GeminiResponse:
    """
    Structured response from Gemini API.
    
    Provides a clean interface for AI responses with metadata about
    the generation process, safety ratings, and usage statistics.
    
    Attributes:
        content: The actual AI response text
        model_used: Which Gemini model generated the response
        timestamp: When the response was generated
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the response
        total_tokens: Total tokens used (prompt + completion)
        finish_reason: Why generation stopped ('stop', 'length', 'safety', etc.)
        safety_ratings: Content safety assessment from Gemini
    """
    
    content: str
    model_used: str
    timestamp: datetime
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None
    safety_ratings: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation showing key info."""
        return f"GeminiResponse(content='{self.content[:50]}...', tokens={self.total_tokens})"
    
    @property
    def is_safe(self) -> bool:
        """Check if response passed all safety checks."""
        if not self.safety_ratings:
            return True  # No ratings means safe
        
        # Check if any safety rating is HIGH probability
        for category, rating in self.safety_ratings.items():
            if rating.get('probability', 'NEGLIGIBLE') in ['HIGH', 'MEDIUM']:
                return False
        
        return True


class RateLimiter:
    """
    Simple rate limiter for API requests.
    
    Tracks request counts and implements delays to respect Gemini API limits.
    Prevents hitting rate limits that would cause API errors.
    
    This is a lightweight implementation focused on preventing errors
    rather than maximizing throughput.
    """
    
    def __init__(self, requests_per_minute: int = 10, requests_per_day: int = 250):
        """
        Initialize rate limiter with API limits.
        
        Args:
            requests_per_minute: Maximum requests per minute (Gemini free tier: 10)
            requests_per_day: Maximum requests per day (Gemini free tier: 250)
        """
        self.rpm_limit = requests_per_minute
        self.rpd_limit = requests_per_day
        
        # Track recent requests (timestamps)
        self.recent_requests: List[datetime] = []
        self.daily_requests: List[datetime] = []
        
        print(f"🚦 Rate limiter initialized: {requests_per_minute}/min, {requests_per_day}/day")
    
    def can_make_request(self) -> bool:
        """
        Check if we can make a request without hitting limits.
        
        Returns:
            bool: True if request can be made, False if rate limited
        """
        now = datetime.now()
        
        # Clean old requests (older than 1 minute)
        minute_ago = now - timedelta(minutes=1)
        self.recent_requests = [req for req in self.recent_requests if req > minute_ago]
        
        # Clean old daily requests (older than 24 hours)
        day_ago = now - timedelta(days=1)
        self.daily_requests = [req for req in self.daily_requests if req > day_ago]
        
        # Check limits
        rpm_ok = len(self.recent_requests) < self.rpm_limit
        rpd_ok = len(self.daily_requests) < self.rpd_limit
        
        return rpm_ok and rpd_ok
    
    def record_request(self) -> None:
        """Record that a request was made."""
        now = datetime.now()
        self.recent_requests.append(now)
        self.daily_requests.append(now)
    
    def wait_if_needed(self) -> float:
        """
        Wait if necessary to respect rate limits.
        
        Returns:
            float: Seconds waited (0 if no wait was needed)
        """
        if self.can_make_request():
            return 0.0
        
        # Calculate wait time (simple approach: wait until we're under the limit)
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Find oldest request in current window
        recent_in_window = [req for req in self.recent_requests if req > minute_ago]
        
        if recent_in_window:
            # Wait until the oldest request is older than 1 minute
            oldest_request = min(recent_in_window)
            wait_until = oldest_request + timedelta(minutes=1, seconds=1)  # Add 1 second buffer
            wait_seconds = (wait_until - now).total_seconds()
            
            if wait_seconds > 0:
                print(f"⏳ Rate limit hit, waiting {wait_seconds:.1f} seconds...")
                time.sleep(wait_seconds)
                return wait_seconds
        
        return 0.0


def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for retrying failed API calls.
    
    Implements exponential backoff for transient failures like network issues
    or temporary service unavailability. Does not retry on permanent errors
    like authentication failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (doubles each attempt)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except (ServiceUnavailable, DeadlineExceeded, ResourceExhausted) as e:
                    # Transient errors - retry with backoff
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ Transient error (attempt {attempt + 1}), retrying in {delay}s: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries exceeded for transient error: {e}")
                        raise
                        
                except (PermissionDenied, InvalidArgument) as e:
                    # Permanent errors - don't retry
                    print(f"❌ Permanent error, not retrying: {e}")
                    raise
                    
                except Exception as e:
                    # Unknown errors - don't retry to avoid infinite loops
                    print(f"❌ Unknown error, not retrying: {e}")
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Unexpected error: retry loop completed without success or exception")
        
        return wrapper
    return decorator


class GeminiClient:
    """
    Google Gemini API client for AI conversations.
    
    Provides high-level interface for generating AI responses with proper
    error handling, rate limiting, and context management. Designed to be
    the main interface for AI functionality throughout the application.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client with API credentials.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use for text generation
            
        Raises:
            ValueError: If API key is invalid
            Exception: If client initialization fails
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize the Gemini API
        try:
            genai.configure(api_key=api_key)
            
            # Create generative model with safety settings
            self.model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            # Initialize rate limiter
            config = get_config()
            self.rate_limiter = RateLimiter(
                requests_per_minute=config.gemini.rpm_limit,
                requests_per_day=config.gemini.rpd_limit
            )
            
            print(f"✅ Gemini client initialized with model: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini client: {e}")
            raise
    
    def _build_context_prompt(self, user_message: str, 
                             conversation_context: Optional[List[str]] = None) -> str:
        """
        Build a context-aware prompt for Gemini.
        
        Combines conversation history with the current user message to provide
        context for more coherent responses.
        
        Args:
            user_message: Current user input
            conversation_context: List of recent conversation messages
            
        Returns:
            str: Formatted prompt with context
        """
        prompt_parts = []
        
        # Add system message for AI behavior
        prompt_parts.append(
            "You are a helpful AI assistant. Provide clear, accurate, contextual and helpful responses. "
            "The answer you return must be concise by default[preferably 2-3 sentences], unless stated otherwise."
            "If you're unsure about something, say so rather than guessing."
        )
        
        # Add conversation context if available
        if conversation_context and len(conversation_context) > 0:
            prompt_parts.append("\nRecent conversation context:")
            for i, context_msg in enumerate(conversation_context[-10:], 1):  # Last 10 messages
                prompt_parts.append(f"{i}. {context_msg}")
        
        # Add current user message
        prompt_parts.append(f"\nCurrent user message: {user_message}")
        prompt_parts.append("\nResponse:")
        
        return "\n".join(prompt_parts)
    
    @retry_on_failure(max_retries=3, base_delay=1.0)
    def generate_response(self, user_message: str, 
                         conversation_context: Optional[List[str]] = None) -> GeminiResponse:
        """
        Generate AI response to user message.
        
        Main method for getting AI responses. Handles context building,
        rate limiting, API calls, and response processing.
        
        Args:
            user_message: The user's input message
            conversation_context: List of recent messages for context
            
        Returns:
            GeminiResponse: Structured response with content and metadata
            
        Raises:
            ValueError: If user message is empty
            ResourceExhausted: If API quota is exceeded
            Exception: For other API errors
            
        Example:
            response = client.generate_response(
                "How do I create a FastAPI endpoint?",
                conversation_context=["Previous message about web frameworks"]
            )
            print(response.content)
        """
        # Validate input
        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty")
        
        user_message = user_message.strip()
        
        # Check and wait for rate limits
        self.rate_limiter.wait_if_needed()
        
        try:
            # Build context-aware prompt
            prompt = self._build_context_prompt(user_message, conversation_context)
            
            print(f"🤖 Generating response for: '{user_message[:50]}...'")
            
            # Make API call to Gemini
            raw_response: GenerateContentResponse = self.model.generate_content(prompt)
            
            # Record the request for rate limiting
            self.rate_limiter.record_request()
            
            # Process response
            response_content = raw_response.text if raw_response.text else ""
            
            # Extract usage data if available
            usage_metadata = getattr(raw_response, 'usage_metadata', None)
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
            completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
            
            # Extract safety ratings
            safety_ratings = {}
            if hasattr(raw_response, 'candidates') and raw_response.candidates:
                candidate = raw_response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        safety_ratings[rating.category.name] = {
                            'probability': rating.probability.name,
                            'blocked': rating.blocked if hasattr(rating, 'blocked') else False
                        }
            
            # Create structured response
            gemini_response = GeminiResponse(
                content=response_content,
                model_used=self.model_name,
                timestamp=datetime.now(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason=getattr(raw_response.candidates[0], 'finish_reason', None) if raw_response.candidates else None,
                safety_ratings=safety_ratings if safety_ratings else None
            )
            
            print(f"✅ Generated response: {len(response_content)} characters, {gemini_response.total_tokens} tokens")
            
            # Check if response is safe
            if not gemini_response.is_safe:
                print("⚠️ Response flagged by safety filters")
            
            return gemini_response
            
        except ResourceExhausted as e:
            print(f"❌ API quota exceeded: {e}")
            raise
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection to Gemini API.
        
        Sends a simple test message to verify that the API key works
        and the service is accessible.
        
        Returns:
            bool: True if connection works, False otherwise
        """
        try:
            print("🔍 Testing Gemini API connection...")
            
            test_response = self.generate_response(
                "Say 'Hello, world!' to test the connection.",
                conversation_context=None
            )
            
            if test_response.content and len(test_response.content) > 0:
                print("✅ Gemini API connection successful")
                return True
            else:
                print("❌ Gemini API returned empty response")
                return False
                
        except Exception as e:
            print(f"❌ Gemini API connection failed: {e}")
            return False


class ConversationManager:
    """
    Manages conversation context for Gemini API calls.
    
    Handles context building, message formatting, and conversation state
    for more coherent multi-turn conversations. This class bridges the
    gap between raw database messages and Gemini API context.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize conversation manager with Gemini client.
        
        Args:
            gemini_client: Initialized GeminiClient instance
        """
        self.client = gemini_client
    
    def format_conversation_context(self, message_history: List[Dict[str, Any]]) -> List[str]:
        """
        Format database message history for Gemini context.
        
        Converts database message records into formatted strings suitable
        for inclusion in Gemini prompts as conversation context.
        
        Args:
            message_history: List of message dictionaries from database
            
        Returns:
            List[str]: Formatted context messages
            
        Example:
            history = [
                {'user_message': 'Hello', 'ai_response': 'Hi there!', 'timestamp': datetime.now()},
                {'user_message': 'How are you?', 'ai_response': 'I am doing well!', 'timestamp': datetime.now()}
            ]
            context = manager.format_conversation_context(history)
            # Returns: ['User: Hello | AI: Hi there!', 'User: How are you? | AI: I am doing well!']
        """
        formatted_messages = []
        
        for msg in message_history:
            user_part = msg.get('user_message', '').strip()
            ai_part = msg.get('ai_response', '').strip()
            
            if user_part and ai_part:
                # Format as "User: ... | AI: ..." for context
                formatted_msg = f"User: {user_part} | AI: {ai_part}"
                formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    def generate_contextual_response(self, user_message: str, 
                                   message_history: Optional[List[Dict[str, Any]]] = None) -> GeminiResponse:
        """
        Generate AI response with full conversation context.
        
        This is the main method for generating responses in the context of
        an ongoing conversation. It formats the message history and includes
        it as context for the AI.
        
        Args:
            user_message: Current user input
            message_history: Previous messages in the conversation
            
        Returns:
            GeminiResponse: AI response with context awareness
            
        Example:
            response = manager.generate_contextual_response(
                "What was that framework you mentioned?",
                message_history=[
                    {'user_message': 'Tell me about web frameworks', 'ai_response': 'FastAPI is excellent...'}
                ]
            )
        """
        # Format message history for context
        conversation_context = []
        if message_history:
            conversation_context = self.format_conversation_context(message_history)
        
        # Generate response with context
        return self.client.generate_response(user_message, conversation_context)


def init_gemini_client() -> GeminiClient:
    """
    Initialize Gemini client with configuration.
    
    Creates a GeminiClient instance using settings from the application
    configuration. This is the main function other parts of the application
    should use to get a configured Gemini client.
    
    Returns:
        GeminiClient: Configured Gemini client
        
    Raises:
        ValueError: If configuration is invalid
        Exception: If client initialization fails
        
    Example:
        client = init_gemini_client()
        response = client.generate_response("Hello, how are you?")
    """
    config = get_config()
    
    try:
        client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.text_model
        )
        
        return client
        
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        raise


def test_gemini_integration() -> None:
    """
    Test complete Gemini integration with conversation context.
    
    This function tests all aspects of the Gemini integration:
    - Client initialization
    - Basic response generation
    - Context-aware conversation
    - Error handling
    
    Useful for verifying that everything works correctly.
    """
    print("\n🧪 Testing Gemini integration...")
    
    try:
        # Test 1: Initialize client
        print("\n1. Initializing Gemini client...")
        client = init_gemini_client()
        
        # Test 2: Test API connection
        print("\n2. Testing API connection...")
        if not client.test_connection():
            raise Exception("API connection test failed")
        
        # Test 3: Generate simple response
        print("\n3. Testing simple response generation...")
        response = client.generate_response("What is Python programming language?")
        print(f"Response: {response.content[:100]}...")
        print(f"Tokens used: {response.total_tokens}")
        
        # Test 4: Test conversation manager
        print("\n4. Testing conversation manager...")
        manager = ConversationManager(client)
        
        # Simulate conversation history
        fake_history = [
            {
                'user_message': 'Tell me about web frameworks',
                'ai_response': 'Web frameworks help you build web applications. FastAPI is a modern Python framework.',
                'timestamp': datetime.now()
            }
        ]
        
        # Generate contextual response
        contextual_response = manager.generate_contextual_response(
            "What makes FastAPI different from Flask?",
            message_history=fake_history
        )
        
        print(f"Contextual response: {contextual_response.content[:100]}...")
        print(f"Safety check: {'✅ Safe' if contextual_response.is_safe else '⚠️ Flagged'}")
        
        print("\n✅ All Gemini integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Gemini integration test failed: {e}")
        raise


# Development utility - run this file directly to test Gemini integration
if __name__ == "__main__":
    """
    Test Gemini integration when run directly.
    
    Usage:
        python src/services/gemini_client.py
        
    This will test the complete Gemini integration including API connection,
    response generation, and conversation context handling.
    """
    test_gemini_integration()