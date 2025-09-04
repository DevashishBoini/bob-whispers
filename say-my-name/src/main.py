"""
AI Assistant FastAPI Backend

This module provides the HTTP API server for the AI Assistant application.
It exposes conversation services as REST endpoints and handles all HTTP
request/response processing for frontend integration.

Key Features:
- RESTful API endpoints for conversation management
- Request/response validation with Pydantic models
- Comprehensive error handling and HTTP status codes
- API documentation with Swagger/OpenAPI
- CORS support for frontend integration
- Health check and monitoring endpoints

Available Endpoints:
- POST /chat/message - Process user messages and get AI responses
- POST /conversations - Create new conversation threads
- GET /conversations - List all conversations
- GET /conversations/{id}/history - Get conversation message history
- DELETE /conversations/{id} - Delete conversations
- GET /health - Health check endpoint

Usage Example:
    # Start the server
    python src/main.py
    
    # Or use uvicorn directly
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import traceback

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn

# Handle imports for different execution contexts
try:
    # When running from src/ directory, use relative import
    from ..config import get_config
except ImportError:
    try:
        # When running from say-my-name/ directory
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from config import get_config
    except ImportError:
        # When running as module from other contexts
        from src.config import get_config

# from services.conversation_service import ConversationService, init_conversation_service, ConversationResponse ## phase-1 only direct context
from services.enhanced_conversation_service import EnhancedConversationService as ConversationService, init_enhanced_conversation_service as init_conversation_service  ## phase-2 enhanced context
from services.conversation_service import ConversationResponse


# Pydantic models for API request/response validation

class MessageRequest(BaseModel):
    """
    Request model for sending messages to AI assistant.
    
    Validates user input for message processing endpoints.
    
    Attributes:
        conversation_id: ID of conversation to add message to
        message: User's input message
        input_type: Type of input ('text' or 'voice')
    """
    
    conversation_id: str = Field(
        ...,
        description="Unique identifier for the conversation",
        min_length=8,
        max_length=50
    )
    
    message: str = Field(
        ...,
        description="User's message content",
        min_length=1,
        max_length=5000
    )
    
    input_type: str = Field(
        default="text",
        description="Type of input (text or voice)"
    )
    
    @field_validator('input_type')
    @classmethod
    def validate_input_type(cls, v):
        """Validate input type is supported."""
        if v not in ['text', 'voice']:
            raise ValueError('input_type must be either "text" or "voice"')
        return v
    
    @field_validator('message')
    @classmethod
    def validate_message_not_empty(cls, v):
        """Validate message is not just whitespace."""
        if not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v.strip()


class MessageResponse(BaseModel):
    """
    Response model for AI assistant messages.
    
    Structured response containing AI content and metadata.
    
    Attributes:
        success: Whether the request was processed successfully
        ai_response: The AI assistant's response content
        message_id: Database ID of stored message
        timestamp: When the response was generated
        conversation_id: ID of the conversation
        input_type: Original input type
        token_usage: Number of tokens used
        error_message: Error description if success is False
    """
    
    success: bool = Field(..., description="Whether the request succeeded")
    ai_response: str = Field(..., description="AI assistant's response")
    message_id: Optional[int] = Field(None, description="Database message ID")
    timestamp: Optional[datetime] = Field(None, description="Response timestamp")
    conversation_id: str = Field(..., description="Conversation ID")
    input_type: str = Field(..., description="Input type used")
    token_usage: int = Field(default=0, description="Tokens used for response")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class NewConversationRequest(BaseModel):
    """
    Request model for creating new conversations.
    
    Attributes:
        title: Optional title for the conversation
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    title: Optional[str] = Field(
        None,
        description="Optional conversation title",
        max_length=100
    )


class ConversationInfo(BaseModel):
    """
    Response model for conversation metadata.
    
    Attributes:
        conversation_id: Unique conversation ID
        title: Conversation title
        created_at: When conversation was created
        last_message_at: Last activity timestamp
        message_count: Number of messages in conversation
        is_active: Whether conversation is active
    """
    
    conversation_id: str = Field(..., description="Unique conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_message_at: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Number of messages")
    is_active: bool = Field(..., description="Whether conversation is active")


class ConversationListResponse(BaseModel):
    """
    Response model for conversation list endpoints.
    
    Attributes:
        conversations: List of conversation metadata
        total_count: Total number of conversations
    """
    
    conversations: List[ConversationInfo] = Field(..., description="List of conversations")
    total_count: int = Field(..., description="Total number of conversations")


class MessageHistoryItem(BaseModel):
    """
    Model for individual message history items.
    
    Attributes:
        id: Message ID
        user_message: User's input
        ai_response: AI's response
        timestamp: Message timestamp
        input_type: Type of input used
    """
    
    id: int = Field(..., description="Message ID")
    user_message: str = Field(..., description="User's message")
    ai_response: str = Field(..., description="AI's response")
    timestamp: datetime = Field(..., description="Message timestamp")
    input_type: str = Field(..., description="Input type")


class ConversationHistoryResponse(BaseModel):
    """
    Response model for conversation history endpoints.
    
    Attributes:
        conversation_id: ID of the conversation
        messages: List of message exchanges
        total_messages: Total number of messages
    """
    
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[MessageHistoryItem] = Field(..., description="Message history")
    total_messages: int = Field(..., description="Total message count")


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Service status
        timestamp: Current timestamp
        version: Application version
        database_connected: Database connection status
        ai_service_connected: AI service connection status
    """
    
    status: str = Field(..., description="Overall service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")
    database_connected: bool = Field(..., description="Database connection status")
    ai_service_connected: bool = Field(..., description="AI service status")


class ErrorResponse(BaseModel):
    """
    Standardized error response model.
    
    Attributes:
        error: Error type
        message: Human-readable error message
        details: Additional error details
        timestamp: When error occurred
    """
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Global conversation service instance
conversation_service: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    """
    Dependency injection for conversation service.
    
    Provides the conversation service instance to API endpoints.
    Implements lazy initialization pattern.
    
    Returns:
        ConversationService: Configured conversation service
        
    Raises:
        HTTPException: If service initialization fails
    """
    global conversation_service
    
    if conversation_service is None:
        try:
            conversation_service = init_conversation_service()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize conversation service: {str(e)}"
            )
    
    return conversation_service


# FastAPI application setup
def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Sets up the FastAPI app with all middleware, exception handlers,
    and configuration needed for the AI assistant API.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    config = get_config()
    
    # Create FastAPI app with metadata
    app = FastAPI(
        title="AI Assistant API",
        description="HTTP API for AI Assistant with conversation management",
        version=config.app.version,
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc",  # ReDoc documentation
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Create the FastAPI app
app = create_app()


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    Catches all unhandled exceptions and returns standardized error responses.
    Logs error details for debugging while returning safe error messages to clients.
    
    Args:
        request: The HTTP request that caused the error
        exc: The exception that was raised
        
    Returns:
        JSONResponse: Standardized error response
    """
    print(f"❌ Unhandled exception: {type(exc).__name__}: {str(exc)}")
    print(f"📍 Traceback: {traceback.format_exc()}")
    
    error_response = ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred. Please try again later.",
        details=str(exc) if app.debug else None  # Only show details in debug mode
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    
    Returns basic information about the API and links to documentation.
    
    Returns:
        Dict[str, str]: API information and links
    """
    config = get_config()
    
    return {
        "name": config.app.name,
        "version": config.app.version,
        "description": "AI Assistant API for conversation management",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Checks the status of all system components including database
    connectivity and AI service availability.
    
    Returns:
        HealthCheckResponse: System health status
        
    Raises:
        HTTPException: If critical services are unavailable
    """
    config = get_config()
    
    # Check database connectivity
    database_connected = False
    try:
        service = get_conversation_service()
        # Try to get conversation list to test database
        service.get_conversation_list()
        database_connected = True
    except Exception as e:
        print(f"⚠️ Database health check failed: {e}")
    
    # Check AI service connectivity
    ai_service_connected = False
    try:
        service = get_conversation_service()
        # This is a simple check - in production you might want a more thorough test
        ai_service_connected = True
    except Exception as e:
        print(f"⚠️ AI service health check failed: {e}")
    
    # Determine overall status
    overall_status = "healthy" if (database_connected and ai_service_connected) else "degraded"
    
    if not database_connected or not ai_service_connected:
        # Return 503 if critical services are down
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        status_code = status.HTTP_200_OK
    
    response = HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(),
        version=config.app.version,
        database_connected=database_connected,
        ai_service_connected=ai_service_connected
    )
    
    # Let FastAPI handle the serialization automatically
    if not database_connected or not ai_service_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump()
        )
    
    return response


@app.post("/chat/message", response_model=MessageResponse)
async def process_message(
    request: MessageRequest,
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Process a user message and generate AI response.
    
    Main endpoint for chat functionality. Processes user input,
    generates AI response with conversation context, and stores
    the exchange in the database.
    
    Args:
        request: Message request containing conversation ID and message
        service: Conversation service dependency
        
    Returns:
        MessageResponse: AI response with metadata
        
    Raises:
        HTTPException: If message processing fails
    """
    try:
        # Process message through conversation service
        response = service.process_message(
            conversation_id=request.conversation_id,
            user_message=request.message,
            input_type=request.input_type
        )
        
        # Convert service response to API response
        return MessageResponse(
            success=response.success,
            ai_response=response.ai_content,
            message_id=response.message_id,
            timestamp=response.timestamp,
            conversation_id=response.conversation_id,
            input_type=response.input_type,
            token_usage=response.token_usage,
            error_message=response.error_message
        )
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@app.post("/conversations", response_model=Dict[str, str])
async def create_conversation(
    request: NewConversationRequest,
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Create a new conversation thread.
    
    Creates a new conversation with optional title. The conversation
    will be ready to receive messages immediately.
    
    Args:
        request: New conversation request
        service: Conversation service dependency
        
    Returns:
        Dict[str, str]: Created conversation ID
        
    Raises:
        HTTPException: If conversation creation fails
    """
    try:
        conversation_id = service.start_new_conversation(title=request.title)
        
        return {
            "conversation_id": conversation_id,
            "message": "Conversation created successfully"
        }
        
    except Exception as e:
        print(f"❌ Error creating conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@app.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Get list of all conversations.
    
    Returns all active conversations ordered by most recent activity.
    Suitable for displaying conversation list in UI sidebar.
    
    Args:
        service: Conversation service dependency
        
    Returns:
        ConversationListResponse: List of conversations with metadata
        
    Raises:
        HTTPException: If listing conversations fails
    """
    try:
        conversations_data = service.get_conversation_list()
        
        # Convert to API response format
        conversations = [
            ConversationInfo(**conv_data) for conv_data in conversations_data
        ]
        
        return ConversationListResponse(
            conversations=conversations,
            total_count=len(conversations)
        )
        
    except Exception as e:
        print(f"❌ Error listing conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


@app.get("/conversations/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    conversation_id: str,
    limit: Optional[int] = None,
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Get message history for a conversation.
    
    Returns all messages in a conversation in chronological order.
    Optionally limits the number of recent messages returned.
    
    Args:
        conversation_id: ID of conversation to get history for
        limit: Optional limit on number of messages to return
        service: Conversation service dependency
        
    Returns:
        ConversationHistoryResponse: Message history with metadata
        
    Raises:
        HTTPException: If conversation not found or history retrieval fails
    """
    try:
        history_data = service.get_conversation_history(conversation_id, limit)
        
        # Convert to API response format
        messages = [
            MessageHistoryItem(**msg_data) for msg_data in history_data
        ]
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=messages,
            total_messages=len(messages)
        )
        
    except Exception as e:
        print(f"❌ Error getting conversation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Delete a conversation and all its messages.
    
    Permanently removes a conversation thread and all associated messages.
    This operation cannot be undone.
    
    Args:
        conversation_id: ID of conversation to delete
        service: Conversation service dependency
        
    Returns:
        Dict[str, str]: Confirmation message
        
    Raises:
        HTTPException: If conversation not found or deletion fails
    """
    try:
        success = service.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )
        
        return {
            "message": f"Conversation {conversation_id} deleted successfully"
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"❌ Error deleting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@app.get("/conversations/stats")
async def get_conversation_stats(
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Get conversation statistics.
    
    Returns summary statistics about all conversations including
    total counts and activity metrics.
    
    Args:
        service: Conversation service dependency
        
    Returns:
        Dict[str, Any]: Conversation statistics
    """
    try:
        stats = service.get_conversation_stats()
        return stats
        
    except Exception as e:
        print(f"❌ Error getting conversation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation stats: {str(e)}"
        )


def run_server():
    """
    Run the FastAPI server with configuration.
    
    Starts the server using uvicorn with settings from configuration.
    This is the main entry point for running the API server.
    """
    config = get_config()
    
    print(f"🚀 Starting AI Assistant API server...")
    print(f"📡 Server: http://{config.app.host}:{config.app.port}")
    print(f"📚 Documentation: http://{config.app.host}:{config.app.port}/docs")
    print(f"🔍 Health check: http://{config.app.host}:{config.app.port}/health")
    
    uvicorn.run(
        "main:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.debug,  # Auto-reload in debug mode
        log_level="info" if not config.app.debug else "debug"
    )


# Development utility - run this file directly to start the server
if __name__ == "__main__":
    """
    Start the FastAPI server when run directly.
    
    Usage:
        python src/main.py
        
    This will start the API server with all endpoints available
    for testing and frontend integration.
    """
    run_server()