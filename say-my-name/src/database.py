"""
AI Assistant Database Layer

This module provides database operations for storing and retrieving conversations.
Uses SQLite as the primary database with SQLAlchemy ORM for type safety and easy querying.

Key Features:
- SQLite database with automatic table creation
- Type-safe database operations using SQLAlchemy
- Conversation and message management
- Connection pooling and transaction handling
- Database initialization and migration utilities

Database Schema:
- conversations: Stores conversation threads (like chat sessions)
- messages: Stores individual message exchanges within conversations

Usage Example:
    from database import DatabaseManager, init_database
    
    # Initialize database
    db = init_database()
    
    # Create a conversation
    conv_id = db.create_conversation("Python Learning Session")
    
    # Add messages
    db.add_message(conv_id, "How do I use FastAPI?", "FastAPI is a modern web framework...")

    
    
Database Schema:

conversations
├── conversation_id (PRIMARY KEY) - Unique ID for each chat thread
├── title - Human-readable name like "Python Learning"
├── created_at - When conversation started
├── last_message_at - Most recent activity
├── message_count - Number of message exchanges
└── is_active - Whether conversation is still active

messages
├── id (PRIMARY KEY) - Auto-incrementing message ID
├── conversation_id (FOREIGN KEY) - Links to parent conversation
├── user_message - What the user said
├── ai_response - What the AI responded
├── timestamp - When this exchange happened
└── input_type - 'text' or 'voice' (for Phase 3)



Class Structure :

DatabaseManager (main operations)
├── create_conversation() - Start new chat thread
├── add_message() - Store user/AI exchange
├── get_conversation_history() - Retrieve past messages
├── get_all_conversations() - List all chat threads
└── update_conversation_title() - Rename conversations

ConversationModel (SQLAlchemy ORM)
MessageModel (SQLAlchemy ORM)
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from config import get_config


# SQLAlchemy Base for ORM models
Base = declarative_base()


class ConversationModel(Base):
    """
    SQLAlchemy model for conversation threads.
    
    Represents a conversation thread (like a chat session) that contains
    multiple message exchanges between user and AI assistant.
    
    Attributes:
        conversation_id: Unique identifier for the conversation
        title: Human-readable title (auto-generated from first message)
        created_at: Timestamp when conversation was created
        last_message_at: Timestamp of most recent message
        message_count: Number of messages in this conversation
        is_active: Whether conversation is currently active
    
    Relationships:
        messages: List of all messages in this conversation
    """
    
    __tablename__ = 'conversations'
    
    conversation_id = Column(String, primary_key=True, nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    last_message_at = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationship to messages (one-to-many)
    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<Conversation(id='{self.conversation_id[:8]}...', title='{self.title}', messages={self.message_count})>"


class MessageModel(Base):
    """
    SQLAlchemy model for individual message exchanges.
    
    Represents a single message exchange: user input and AI response.
    Each message belongs to a conversation thread.
    
    Attributes:
        id: Auto-incrementing primary key
        conversation_id: Foreign key to parent conversation
        user_message: The user's input message
        ai_response: The AI assistant's response
        timestamp: When this exchange occurred
        input_type: Type of input ('text' or 'voice')
    
    Relationships:
        conversation: Parent conversation this message belongs to
    """
    
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey('conversations.conversation_id'), nullable=False)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    input_type = Column(String, default='text', nullable=False)  # 'text' or 'voice'
    
    # Relationship to conversation (many-to-one)
    conversation = relationship("ConversationModel", back_populates="messages")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        preview = self.user_message[:50] + "..." if len(self.user_message) > 50 else self.user_message
        return f"<Message(id={self.id}, type='{self.input_type}', preview='{preview}')>"


class DatabaseManager:
    """
    Database manager for handling all database operations.
    
    Provides high-level methods for creating conversations, adding messages,
    and querying conversation history. Handles database connections, 
    transactions, and error handling.
    
    This class is designed to be the main interface for database operations
    throughout the application.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database manager with connection.
        
        Args:
            database_url: SQLAlchemy database URL (e.g., 'sqlite:///path/to/db.sqlite')
            
        Example:
            db = DatabaseManager('sqlite:///data/conversations.db')
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            # SQLite-specific optimizations
            echo=False,  # Set to True to log SQL queries (useful for debugging)
            connect_args={"check_same_thread": False}  # Allow multiple threads
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self) -> None:
        """
        Create database tables if they don't exist.
        
        This method is called during initialization to ensure all required
        tables are present in the database.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            print("✅ Database tables created/verified successfully")
        except SQLAlchemyError as e:
            print(f"❌ Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.
        
        Provides automatic session management with proper cleanup and error handling.
        Ensures that sessions are always closed and transactions are handled correctly.
        
        Yields:
            Session: SQLAlchemy session for database operations
            
        Example:
            with db.get_session() as session:
                conversation = session.query(ConversationModel).first()
                print(conversation.title)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"❌ Database transaction failed: {e}")
            raise
        finally:
            session.close()
    
    def create_conversation(self, title: str) -> str:
        """
        Create a new conversation thread.
        
        Creates a new conversation with a unique ID and initial title.
        The conversation will be marked as active and ready to receive messages.
        
        Args:
            title: Human-readable title for the conversation
            
        Returns:
            str: Unique conversation ID
            
        Raises:
            SQLAlchemyError: If database operation fails
            
        Example:
            conv_id = db.create_conversation("Python Learning Session")
            print(f"Created conversation: {conv_id}")
        """
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        try:
            with self.get_session() as session:
                conversation = ConversationModel(
                    conversation_id=conversation_id,
                    title=title,
                    created_at=datetime.now(timezone.utc),
                    last_message_at=datetime.now(timezone.utc),
                    message_count=0,
                    is_active=True
                )
                
                session.add(conversation)
                session.flush()  # Ensure the conversation is saved
                
                print(f"✅ Created conversation: {title} ({conversation_id})")
                return conversation_id
                
        except SQLAlchemyError as e:
            print(f"❌ Failed to create conversation: {e}")
            raise
    
    def add_message(self, conversation_id: str, user_message: str, 
                   ai_response: str, input_type: str = 'text') -> int:
        """
        Add a message exchange to a conversation.
        
        Stores both the user's input and AI's response as a single message record.
        Updates the parent conversation's metadata (last_message_at, message_count).
        
        Args:
            conversation_id: ID of the conversation to add message to
            user_message: The user's input message
            ai_response: The AI assistant's response
            input_type: Type of input ('text' or 'voice')
            
        Returns:
            int: ID of the created message record
            
        Raises:
            ValueError: If conversation doesn't exist
            SQLAlchemyError: If database operation fails
            
        Example:
            message_id = db.add_message(
                conv_id, 
                "How do I use FastAPI?",
                "FastAPI is a modern web framework for Python...",
                input_type='text'
            )
        """
        try:
            with self.get_session() as session:
                # Verify conversation exists
                conversation = session.query(ConversationModel).filter_by(
                    conversation_id=conversation_id
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation {conversation_id} not found")
                
                # Create message record
                message = MessageModel(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    ai_response=ai_response,
                    timestamp=datetime.now(timezone.utc),
                    input_type=input_type
                )
                
                session.add(message)
                session.flush()  # Get the message ID
                
                # Update conversation metadata
                conversation.last_message_at = datetime.now(timezone.utc)
                conversation.message_count += 1
                
                print(f"✅ Added message to conversation {conversation_id[:8]}...")
                return message.id
                
        except SQLAlchemyError as e:
            print(f"❌ Failed to add message: {e}")
            raise
    
    def get_conversation_history(self, conversation_id: str, 
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get message history for a conversation.
        
        Retrieves all messages in a conversation, ordered by timestamp.
        Optionally limits the number of recent messages returned.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of recent messages to return (None = all)
            
        Returns:
            List[Dict]: List of message dictionaries with keys:
                - id: Message ID
                - user_message: User's input
                - ai_response: AI's response
                - timestamp: When the exchange occurred
                - input_type: Type of input ('text' or 'voice')
                
        Example:
            # Get last 10 messages
            history = db.get_conversation_history(conv_id, limit=10)
            for msg in history:
                print(f"User: {msg['user_message']}")
                print(f"AI: {msg['ai_response']}")
        """
        try:
            with self.get_session() as session:
                query = session.query(MessageModel).filter_by(
                    conversation_id=conversation_id
                ).order_by(MessageModel.timestamp.asc())
                
                if limit:
                    # Get most recent messages (ordered by timestamp DESC, then reverse)
                    query = query.order_by(MessageModel.timestamp.desc()).limit(limit)
                    messages = query.all()
                    messages.reverse()  # Return in chronological order
                else:
                    messages = query.all()
                
                # Convert to dictionaries
                return [
                    {
                        'id': msg.id,
                        'user_message': msg.user_message,
                        'ai_response': msg.ai_response,
                        'timestamp': msg.timestamp,
                        'input_type': msg.input_type
                    }
                    for msg in messages
                ]
                
        except SQLAlchemyError as e:
            print(f"❌ Failed to get conversation history: {e}")
            return []
    
    def get_all_conversations(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of all conversations.
        
        Returns conversation metadata sorted by most recent activity.
        Useful for displaying conversation list in UI sidebar.
        
        Args:
            include_inactive: Whether to include inactive conversations
            
        Returns:
            List[Dict]: List of conversation dictionaries with keys:
                - conversation_id: Unique ID
                - title: Conversation title
                - created_at: Creation timestamp
                - last_message_at: Last activity timestamp
                - message_count: Number of messages
                - is_active: Whether conversation is active
                
        Example:
            conversations = db.get_all_conversations()
            for conv in conversations:
                print(f"{conv['title']} - {conv['message_count']} messages")
        """
        try:
            with self.get_session() as session:
                query = session.query(ConversationModel)
                
                if not include_inactive:
                    query = query.filter_by(is_active=True)
                
                # Order by most recent activity
                conversations = query.order_by(
                    ConversationModel.last_message_at.desc()
                ).all()
                
                return [
                    {
                        'conversation_id': conv.conversation_id,
                        'title': conv.title,
                        'created_at': conv.created_at,
                        'last_message_at': conv.last_message_at,
                        'message_count': conv.message_count,
                        'is_active': conv.is_active
                    }
                    for conv in conversations
                ]
                
        except SQLAlchemyError as e:
            print(f"❌ Failed to get conversations: {e}")
            return []
    
    def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """
        Update the title of a conversation.
        
        Useful for renaming conversations or setting auto-generated titles
        based on the first message content.
        
        Args:
            conversation_id: ID of conversation to update
            new_title: New title for the conversation
            
        Returns:
            bool: True if update was successful, False otherwise
            
        Example:
            success = db.update_conversation_title(conv_id, "FastAPI Learning Session")
            if success:
                print("Title updated successfully")
        """
        try:
            with self.get_session() as session:
                conversation = session.query(ConversationModel).filter_by(
                    conversation_id=conversation_id
                ).first()
                
                if not conversation:
                    print(f"❌ Conversation {conversation_id} not found")
                    return False
                
                conversation.title = new_title
                print(f"✅ Updated conversation title to: {new_title}")
                return True
                
        except SQLAlchemyError as e:
            print(f"❌ Failed to update conversation title: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Permanently removes a conversation thread and all associated messages.
        Use with caution as this operation cannot be undone.
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
            
        Example:
            if db.delete_conversation(old_conv_id):
                print("Conversation deleted successfully")
        """
        try:
            with self.get_session() as session:
                conversation = session.query(ConversationModel).filter_by(
                    conversation_id=conversation_id
                ).first()
                
                if not conversation:
                    print(f"❌ Conversation {conversation_id} not found")
                    return False
                
                # Delete conversation (messages will be deleted due to cascade)
                session.delete(conversation)
                print(f"✅ Deleted conversation: {conversation.title}")
                return True
                
        except SQLAlchemyError as e:
            print(f"❌ Failed to delete conversation: {e}")
            return False


def init_database() -> DatabaseManager:
    """
    Initialize database manager with configuration.
    
    Creates a DatabaseManager instance using settings from the configuration.
    This is the main function other parts of the application should use.
    
    Returns:
        DatabaseManager: Configured database manager instance
        
    Raises:
        SQLAlchemyError: If database initialization fails
        
    Example:
        db = init_database()
        conv_id = db.create_conversation("My First Chat")
    """
    config = get_config()
    
    # Create SQLite database URL
    db_path = config.database.path
    database_url = f"sqlite:///{db_path}"
    
    try:
        print(f"🔗 Connecting to database: {db_path}")
        db_manager = DatabaseManager(database_url)
        print("✅ Database initialized successfully")
        return db_manager
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        raise


def test_database_operations() -> None:
    """
    Test basic database operations.
    
    This function tests all major database operations to ensure
    everything is working correctly. Useful for debugging and verification.
    
    Example operations tested:
    - Creating conversations
    - Adding messages
    - Retrieving history
    - Updating titles
    """
    print("\n🧪 Testing database operations...")
    
    try:
        # Initialize database
        db = init_database()
        
        # Test 1: Create conversation
        print("\n1. Creating test conversation...")
        conv_id = db.create_conversation("Test Conversation")
        
        # Test 2: Add messages
        print("\n2. Adding test messages...")
        msg_id_1 = db.add_message(
            conv_id, 
            "Hello, how are you?", 
            "I'm doing well, thank you! How can I help you today?",
            input_type='text'
        )
        
        msg_id_2 = db.add_message(
            conv_id,
            "Can you explain Python?",
            "Python is a high-level programming language known for its simplicity and readability.",
            input_type='text'
        )
        
        # Test 3: Get conversation history
        print("\n3. Retrieving conversation history...")
        history = db.get_conversation_history(conv_id)
        print(f"Retrieved {len(history)} messages")
        
        for i, msg in enumerate(history, 1):
            print(f"  Message {i}: {msg['user_message'][:30]}...")
        
        # Test 4: Get all conversations
        print("\n4. Getting all conversations...")
        conversations = db.get_all_conversations()
        print(f"Found {len(conversations)} conversations")
        
        # Test 5: Update conversation title
        print("\n5. Updating conversation title...")
        success = db.update_conversation_title(conv_id, "Updated Test Conversation")
        print(f"Title update: {'✅ Success' if success else '❌ Failed'}")
        
        print("\n✅ All database tests passed!")
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        raise


# Development utility - run this file directly to test database
if __name__ == "__main__":
    """
    Test database functionality when run directly.
    
    Usage:
        python src/database.py
        
    This will initialize the database and run all test operations,
    helping you verify that the database layer is working correctly.
    """
    test_database_operations()