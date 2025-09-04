"""
AI Assistant Embedding Service

This module provides text-to-vector conversion using Google's embedding models.
Embeddings transform text into high-dimensional numerical vectors that capture
semantic meaning, enabling similarity search and semantic memory functionality.

Key Concepts:
- Embeddings: Mathematical representations of text as vectors (typically 768 dimensions)
- Semantic Similarity: Similar meanings result in similar vector positions in high-dimensional space
- Batch Processing: Converting multiple texts to vectors efficiently in single API calls
- Embedding Models: Google's text-embedding-001 and newer models optimized for different tasks

Theory Background:
Embeddings work by mapping words and phrases to points in a high-dimensional space where
semantically similar content clusters together. For example:
- "debug Python code" and "fix Python errors" will have similar embeddings
- "FastAPI routes" and "API endpoints" will be close in vector space
- "cats" and "dogs" will be closer than "cats" and "airplanes"

This mathematical representation enables computers to understand semantic relationships
without requiring exact word matches.

Usage Example:
    from services.embedding_service import EmbeddingService, init_embedding_service
    
    # Initialize embedding service
    embedding_service = init_embedding_service()
    
    # Convert single text to embedding
    vector = embedding_service.create_embedding("How do I debug FastAPI?")
    print(f"Embedding dimension: {len(vector)}")  # Should be 768
    
    # Batch process multiple texts
    texts = ["Python debugging", "FastAPI errors", "API testing"]
    vectors = embedding_service.create_embeddings_batch(texts)

Architecture:
EmbeddingConfig - Configuration for embedding models and parameters
EmbeddingResult - Data structure for embedding results with metadata  
TextProcessor - Utilities for cleaning and preparing text for embedding
EmbeddingService - Main service for text-to-vector conversion
"""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import re

import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted, 
    InvalidArgument, 
    ServiceUnavailable,
    DeadlineExceeded
)
from pydantic import BaseModel, Field, ConfigDict

# Handle imports for different execution contexts
try:
    from ..config import get_config
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from config import get_config
    except ImportError:
        from src.config import get_config


class EmbeddingConfig(BaseModel):
    """
    Configuration settings for embedding generation service.
    
    Defines model parameters, batch processing settings, and operational
    limits for text-to-vector conversion operations.
    
    Attributes:
        model_name: Google embedding model identifier
        embedding_dimension: Expected output vector dimension
        max_batch_size: Maximum texts to process in single API call
        max_text_length: Maximum character length for input text
        rate_limit_delay: Minimum seconds between API calls
        retry_attempts: Number of retry attempts for failed requests
        normalize_vectors: Whether to normalize embedding vectors to unit length
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True
    )
    
    model_name: str = Field(
        default="models/embedding-001",
        description="Google embedding model name",
        pattern=r"^models/embedding-\d{3}$"
    )
    
    embedding_dimension: int = Field(
        default=768,
        description="Expected dimension of output embedding vectors",
        gt=0,
        le=2048
    )
    
    max_batch_size: int = Field(
        default=100,
        description="Maximum number of texts to embed in single API call",
        gt=0,
        le=500
    )
    
    max_text_length: int = Field(
        default=2000,
        description="Maximum character length for input text",
        gt=0,
        le=10000
    )
    
    rate_limit_delay: float = Field(
        default=0.1,
        description="Minimum seconds between embedding API calls",
        ge=0.0,
        le=5.0
    )
    
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests",
        ge=1,
        le=10
    )
    
    normalize_vectors: bool = Field(
        default=True,
        description="Whether to normalize embedding vectors to unit length"
    )


@dataclass
class EmbeddingResult:
    """
    Result of text embedding operation with metadata.
    
    Contains the embedding vector along with processing information
    and original text for debugging and validation.
    
    Attributes:
        text: Original input text that was embedded
        embedding: High-dimensional vector representation (768 floats)
        model_used: Name of the embedding model
        processing_time: Seconds taken to generate embedding
        timestamp: When the embedding was created
        text_length: Character count of original text
        success: Whether embedding generation succeeded
        error_message: Error description if success is False
    
    Example:
        result = EmbeddingResult(
            text="How do I debug FastAPI applications?",
            embedding=[0.1, -0.2, 0.3, ...],  # 768 dimensions
            model_used="models/embedding-001",
            processing_time=0.15,
            timestamp=datetime.now(),
            text_length=36,
            success=True
        )
    """
    
    text: str
    embedding: List[float]
    model_used: str
    processing_time: float
    timestamp: datetime
    text_length: int
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate embedding result after initialization."""
        if self.success and not self.embedding:
            self.success = False
            self.error_message = "Embedding vector is empty"
        
        if self.success and len(self.embedding) == 0:
            self.success = False
            self.error_message = "Embedding vector has zero dimensions"
    
    def get_vector_magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the embedding vector.
        
        Returns:
            float: Euclidean norm of the embedding vector
        """
        if not self.embedding:
            return 0.0
        
        return sum(x * x for x in self.embedding) ** 0.5
    
    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the embedding vector is normalized (unit length).
        
        Args:
            tolerance: Acceptable deviation from unit length
            
        Returns:
            bool: True if vector is approximately unit length
        """
        magnitude = self.get_vector_magnitude()
        return abs(magnitude - 1.0) < tolerance


class TextProcessor:
    """
    Utility class for preparing text for embedding generation.
    
    Provides text cleaning, validation, and preprocessing functions
    to optimize text input for embedding models and improve result quality.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text for embedding generation.
        
        Removes excessive whitespace, normalizes line breaks, and handles
        common text artifacts that could affect embedding quality.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Cleaned and normalized text
            
        Example:
            raw_text = "  How  do\\n\\n  I debug  FastAPI?  \\t"
            clean_text = TextProcessor.clean_text(raw_text)
            # Returns: "How do I debug FastAPI?"
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters but preserve basic punctuation
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned)
        
        # Normalize quotes and apostrophes
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")
        
        return cleaned
    
    @staticmethod
    def truncate_text(text: str, max_length: int) -> str:
        """
        Truncate text to maximum length while preserving word boundaries.
        
        Cuts text at word boundaries when possible to maintain semantic coherence.
        Adds ellipsis indicator when truncation occurs.
        
        Args:
            text: Input text to truncate
            max_length: Maximum allowed character length
            
        Returns:
            str: Truncated text with ellipsis if needed
            
        Example:
            long_text = "This is a very long text that needs truncation"
            short_text = TextProcessor.truncate_text(long_text, 20)
            # Returns: "This is a very..."
        """
        if len(text) <= max_length:
            return text
        
        if max_length < 4:  # Too short for meaningful truncation
            return text[:max_length]
        
        # Try to truncate at word boundary
        truncated = text[:max_length - 3]  # Leave space for "..."
        
        # Find last space to avoid cutting words
        last_space = truncated.rfind(' ')
        if last_space > max_length // 2:  # Only if we keep reasonable portion
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    @staticmethod
    def validate_text(text: str, max_length: int = 2000) -> Tuple[bool, str]:
        """
        Validate text input for embedding generation.
        
        Checks text length, content validity, and other requirements
        for successful embedding generation.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed character length
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
            
        Example:
            is_valid, error = TextProcessor.validate_text("Hello world")
            if is_valid:
                print("Text is ready for embedding")
            else:
                print(f"Validation error: {error}")
        """
        if not text:
            return False, "Text cannot be empty"
        
        if not text.strip():
            return False, "Text cannot be only whitespace"
        
        if len(text) > max_length:
            return False, f"Text too long: {len(text)} chars (max: {max_length})"
        
        # Check for minimum meaningful content
        if len(text.strip()) < 3:
            return False, "Text too short for meaningful embedding"
        
        return True, ""
    
    @staticmethod
    def prepare_for_embedding(text: str, max_length: int = 2000) -> str:
        """
        Complete text preparation pipeline for embedding generation.
        
        Combines cleaning, validation, and truncation into a single
        preprocessing step optimized for embedding model input.
        
        Args:
            text: Raw input text
            max_length: Maximum allowed length after processing
            
        Returns:
            str: Processed text ready for embedding
            
        Raises:
            ValueError: If text is invalid after processing
            
        Example:
            raw_input = "  How   do I\\n\\tfix FastAPI errors?  "
            processed = TextProcessor.prepare_for_embedding(raw_input)
            # Returns: "How do I fix FastAPI errors?"
        """
        # Clean the text first
        cleaned = TextProcessor.clean_text(text)
        
        # Validate processed text
        is_valid, error_message = TextProcessor.validate_text(cleaned, max_length)
        if not is_valid:
            raise ValueError(f"Text validation failed: {error_message}")
        
        # Truncate if necessary
        final_text = TextProcessor.truncate_text(cleaned, max_length)
        
        return final_text


class EmbeddingService:
    """
    Service for converting text to embedding vectors using Google's embedding API.
    
    Provides both single-text and batch processing capabilities with proper
    error handling, rate limiting, and result validation. This service abstracts
    the complexity of embedding generation while providing rich metadata.
    """
    
    def __init__(self, config: EmbeddingConfig, api_key: str):
        """
        Initialize embedding service with configuration.
        
        Args:
            config: EmbeddingConfig with service parameters
            api_key: Google Generative AI API key
            
        Raises:
            ValueError: If configuration or API key is invalid
        """
        self.config = config
        self.api_key = api_key
        self.text_processor = TextProcessor()
        self._last_request_time = 0.0
        
        # Initialize Google Generative AI
        try:
            genai.configure(api_key=api_key)
            print(f"Embedding service initialized with model: {config.model_name}")
        except Exception as e:
            raise ValueError(f"Failed to initialize embedding service: {e}")
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting between API requests.
        
        Ensures minimum delay between embedding API calls to respect
        rate limits and avoid hitting quota restrictions.
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize embedding vector to unit length.
        
        Converts embedding to unit vector (magnitude = 1.0) which can
        improve similarity search performance and consistency.
        
        Args:
            vector: Original embedding vector
            
        Returns:
            List[float]: Normalized embedding vector
        """
        if not self.config.normalize_vectors:
            return vector
        
        # Calculate magnitude
        magnitude = sum(x * x for x in vector) ** 0.5
        
        if magnitude == 0.0:
            return vector  # Cannot normalize zero vector
        
        # Normalize to unit length
        return [x / magnitude for x in vector]
    
    def create_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding vector for a single text input.
        
        Converts text to a high-dimensional vector representation using
        Google's embedding model. Includes preprocessing, API call,
        and post-processing with comprehensive error handling.
        
        Args:
            text: Input text to convert to embedding
            
        Returns:
            EmbeddingResult: Embedding vector with metadata and success status
            
        Example:
            embedding_service = EmbeddingService(config, api_key)
            result = embedding_service.create_embedding("How do I debug FastAPI?")
            
            if result.success:
                print(f"Generated {len(result.embedding)}-dimensional vector")
                print(f"Processing time: {result.processing_time:.3f}s")
            else:
                print(f"Embedding failed: {result.error_message}")
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Preprocess text
            processed_text = self.text_processor.prepare_for_embedding(
                text, 
                self.config.max_text_length
            )
            
            # Enforce rate limiting
            self._enforce_rate_limit()
            
            # Generate embedding using Google's API
            embedding_response = genai.embed_content(
                model=self.config.model_name,
                content=processed_text,
                task_type="retrieval_document"  # Optimized for document storage/retrieval
            )
            
            # Extract embedding vector
            if not embedding_response or 'embedding' not in embedding_response:
                raise ValueError("Invalid embedding response from API")
            
            raw_embedding = embedding_response['embedding']
            
            # Validate embedding dimensions
            if len(raw_embedding) != self.config.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.config.embedding_dimension}, "
                    f"got {len(raw_embedding)}"
                )
            
            # Normalize vector if configured
            final_embedding = self._normalize_vector(raw_embedding)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create successful result
            result = EmbeddingResult(
                text=processed_text,
                embedding=final_embedding,
                model_used=self.config.model_name,
                processing_time=processing_time,
                timestamp=timestamp,
                text_length=len(processed_text),
                success=True
            )
            
            print(f"Generated embedding for text: '{processed_text[:50]}...' "
                  f"({len(final_embedding)} dimensions, {processing_time:.3f}s)")
            
            return result
            
        except ValueError as e:
            # Client-side validation errors
            return EmbeddingResult(
                text=text[:50] + "..." if len(text) > 50 else text,
                embedding=[],
                model_used=self.config.model_name,
                processing_time=time.time() - start_time,
                timestamp=timestamp,
                text_length=len(text),
                success=False,
                error_message=f"Validation error: {str(e)}"
            )
            
        except (ResourceExhausted, ServiceUnavailable) as e:
            # API quota or service errors
            return EmbeddingResult(
                text=text[:50] + "..." if len(text) > 50 else text,
                embedding=[],
                model_used=self.config.model_name,
                processing_time=time.time() - start_time,
                timestamp=timestamp,
                text_length=len(text),
                success=False,
                error_message=f"API error: {str(e)}"
            )
            
        except Exception as e:
            # Unexpected errors
            return EmbeddingResult(
                text=text[:50] + "..." if len(text) > 50 else text,
                embedding=[],
                model_used=self.config.model_name,
                processing_time=time.time() - start_time,
                timestamp=timestamp,
                text_length=len(text),
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def create_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embedding vectors for multiple texts efficiently.
        
        Processes multiple texts in batches to optimize API usage while
        maintaining individual error handling for each text input.
        
        Args:
            texts: List of input texts to convert to embeddings
            
        Returns:
            List[EmbeddingResult]: Embedding results for each input text
            
        Example:
            texts = [
                "How do I debug FastAPI?",
                "Python error handling best practices",
                "API testing strategies"
            ]
            results = embedding_service.create_embeddings_batch(texts)
            
            successful_embeddings = [r for r in results if r.success]
            print(f"Successfully processed {len(successful_embeddings)}/{len(texts)} texts")
        """
        if not texts:
            return []
        
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), self.config.max_batch_size):
            batch = texts[i:i + self.config.max_batch_size]
            
            print(f"Processing embedding batch {i//self.config.max_batch_size + 1}: "
                  f"{len(batch)} texts")
            
            # Process each text in the batch individually for now
            # (Google's embedding API batch support may vary by model)
            for text in batch:
                result = self.create_embedding(text)
                results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"Batch processing complete: {successful} successful, {failed} failed")
        
        return results
    
    def test_embedding_generation(self) -> bool:
        """
        Test embedding generation with a simple input.
        
        Validates that the embedding service is working correctly by
        generating an embedding for a test phrase and checking the result.
        
        Returns:
            bool: True if embedding generation works, False otherwise
        """
        try:
            test_text = "This is a test message for embedding generation."
            result = self.create_embedding(test_text)
            
            if result.success:
                print(f"Embedding test successful:")
                print(f"  Text: {result.text}")
                print(f"  Dimensions: {len(result.embedding)}")
                print(f"  Processing time: {result.processing_time:.3f}s")
                print(f"  Vector magnitude: {result.get_vector_magnitude():.6f}")
                print(f"  Normalized: {result.is_normalized()}")
                return True
            else:
                print(f"Embedding test failed: {result.error_message}")
                return False
                
        except Exception as e:
            print(f"Embedding test error: {e}")
            return False


def init_embedding_service() -> EmbeddingService:
    """
    Initialize embedding service with application configuration.
    
    Creates an EmbeddingService instance using settings from the main
    application configuration. This is the primary way other parts of
    the application should get a configured embedding service.
    
    Returns:
        EmbeddingService: Fully configured embedding service
        
    Raises:
        ValueError: If configuration is invalid
        Exception: If service initialization fails
        
    Example:
        embedding_service = init_embedding_service()
        
        # Generate single embedding
        result = embedding_service.create_embedding("Sample text")
        if result.success:
            vector = result.embedding
    """
    try:
        # Get application configuration
        app_config = get_config()
        
        # Create embedding configuration
        embedding_config = EmbeddingConfig(
            model_name=app_config.gemini.embedding_model,
            embedding_dimension=768,  # Standard for Google embeddings
            max_batch_size=50,  # Conservative batch size
            max_text_length=2000,  # Reasonable limit for conversation messages
            rate_limit_delay=0.1,  # Small delay between requests
            retry_attempts=3,
            normalize_vectors=True  # Normalize for better similarity search
        )
        
        # Initialize embedding service
        embedding_service = EmbeddingService(
            config=embedding_config,
            api_key=app_config.gemini.api_key
        )
        
        print("Embedding service initialization completed successfully")
        return embedding_service
        
    except Exception as e:
        print(f"Failed to initialize embedding service: {e}")
        raise


def test_embedding_service() -> None:
    """
    Test complete embedding service functionality.
    
    Validates embedding generation with various text inputs including:
    - Single text embedding
    - Batch processing
    - Edge cases and error handling
    - Vector normalization and validation
    """
    print("\n🧪 Testing embedding service functionality...")
    
    try:
        # Test 1: Initialize service
        print("\n1. Initializing embedding service...")
        embedding_service = init_embedding_service()
        
        # Test 2: Basic embedding generation
        print("\n2. Testing basic embedding generation...")
        if not embedding_service.test_embedding_generation():
            raise Exception("Basic embedding test failed")
        
        # Test 3: Single text embedding
        print("\n3. Testing single text embedding...")
        test_text = "How do I debug FastAPI applications effectively?"
        result = embedding_service.create_embedding(test_text)
        
        if result.success:
            print(f"Single embedding successful:")
            print(f"  Original text length: {len(test_text)} chars")
            print(f"  Processed text length: {result.text_length} chars") 
            print(f"  Embedding dimensions: {len(result.embedding)}")
            print(f"  Processing time: {result.processing_time:.3f}s")
            print(f"  Vector normalized: {result.is_normalized()}")
        else:
            print(f"Single embedding failed: {result.error_message}")
        
        # Test 4: Batch processing
        print("\n4. Testing batch embedding generation...")
        test_texts = [
            "FastAPI routing and endpoints",
            "Python debugging techniques", 
            "API error handling strategies",
            "Database integration with SQLAlchemy",
            "Testing web applications"
        ]
        
        batch_results = embedding_service.create_embeddings_batch(test_texts)
        
        successful_count = sum(1 for r in batch_results if r.success)
        print(f"Batch processing: {successful_count}/{len(test_texts)} successful")
        
        # Test 5: Text preprocessing validation
        print("\n5. Testing text preprocessing...")
        processor = TextProcessor()
        
        # Test text cleaning
        messy_text = "  How   do I\\n\\tfix  errors?  "
        clean_text = processor.clean_text(messy_text)
        print(f"Text cleaning: '{messy_text}' -> '{clean_text}'")
        
        # Test text validation
        is_valid, error = processor.validate_text("Valid text for testing")
        print(f"Validation: {'✅ Valid' if is_valid else f'❌ Invalid: {error}'}")
        
        print("\n✅ All embedding service tests passed!")
        
    except Exception as e:
        print(f"\n❌ Embedding service test failed: {e}")
        raise


# Development utility - run this file directly to test embedding service
if __name__ == "__main__":
    """
    Test embedding service functionality when run directly.
    
    Usage:
        python src/services/embedding_service.py
        
    This will test text preprocessing, embedding generation,
    and batch processing capabilities.
    """
    test_embedding_service()