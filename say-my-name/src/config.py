"""
AI Assistant Configuration Management System

This module provides centralized configuration management for the AI Assistant application.
It uses Pydantic v2 for type safety and automatic environment variable loading with validation.

Key Features:
- Type-safe configuration with Pydantic v2 models
- Automatic environment variable loading from .env files
- Configuration validation with helpful error messages
- Organized settings by functional area (database, AI, logging, etc.)
- Default values for development environment

Usage Example:
    from config import get_config
    
    config = get_config()
    api_key = config.gemini.api_key
    db_path = config.database.path

    
Class Structure :
    
Settings (main container)
├── GeminiConfig - AI API settings only
├── DatabaseConfig - Storage settings only  
├── AppConfig - Server settings only
└── LoggingConfig - Logging settings only

"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import BaseModel, Field, ConfigDict, field_validator
from dotenv import load_dotenv


class GeminiConfig(BaseModel):
    """
    Google Gemini API configuration settings.
    
    Handles all Gemini-related API keys, model names, and rate limiting settings.
    This class is separated to keep AI-related configuration organized.
    
    Attributes:
        api_key: Google Gemini API key for authentication
        text_model: Model name for text conversations  
        audio_model: Model name for audio processing (speech-to-text)
        embedding_model: Model name for generating embeddings (Phase 2)
        rpm_limit: Requests per minute limit (free tier: 10)
        rpd_limit: Requests per day limit (free tier: 250)
    """
    
    model_config = ConfigDict(
        # Allow environment variable loading with prefix
        env_prefix='GEMINI_',
        # Case insensitive environment variables
        case_sensitive=False,
        # Validate assignment to catch errors early
        validate_assignment=True
    )
    
    api_key: str = Field(
        ..., 
        description="Google Gemini API key from https://aistudio.google.com",
        min_length=10
    )
    
    text_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for text conversations"
    )
    
    audio_model: str = Field(
        default="gemini-2.5-flash", 
        description="Gemini model for audio processing"
    )
    
    embedding_model: str = Field(
        default="models/embedding-001",
        description="Gemini model for generating embeddings"
    )
    
    rpm_limit: int = Field(
        default=10,
        description="Requests per minute limit for free tier",
        ge=1,  # Greater than or equal to 1
        le=1000  # Reasonable upper bound
    )
    
    rpd_limit: int = Field(
        default=250,
        description="Requests per day limit for free tier", 
        ge=1,
        le=10000
    )
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """
        Validate that API key is not a placeholder.
        
        Args:
            v: The API key string to validate
            
        Returns:
            str: The validated API key
            
        Raises:
            ValueError: If API key appears to be a placeholder
        """
        if v in ['your_api_key_here', 'your_gemini_api_key_here', '']:
            raise ValueError(
                "Please set a real Gemini API key in your .env file. "
                "Get one from https://aistudio.google.com/app/apikey"
            )
        return v


class DatabaseConfig(BaseModel):
    """
    Database configuration for SQLite storage.
    
    Manages file paths and database connection settings.
    Ensures database directory exists and handles path resolution.
    
    Attributes:
        path: Path to SQLite database file
        chromadb_path: Path to ChromaDB vector database directory (Phase 2)
        auto_create_dirs: Whether to automatically create directories
    """
    
    model_config = ConfigDict(
        env_prefix='DATABASE_',
        case_sensitive=False
    )
    
    path: Path = Field(
        default=Path("data/conversations.db"),
        description="Path to SQLite database file"
    )
    
    chromadb_path: Path = Field(
        default=Path("data/chromadb"),
        description="Path to ChromaDB vector database directory"
    )
    
    auto_create_dirs: bool = Field(
        default=True,
        description="Automatically create database directories if they don't exist"
    )
    
    def model_post_init(self, __context) -> None:
        """
        Post-initialization hook to create directories.
        
        This method runs after the model is initialized and creates
        the necessary directories for database storage.
        """
        if self.auto_create_dirs:
            # Create SQLite database directory
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create ChromaDB directory  
            self.chromadb_path.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    """
    General application configuration settings.
    
    Contains environment, server, and application metadata settings.
    These are the core settings that affect how the application runs.
    
    Attributes:
        env: Application environment (development/production)
        name: Application name for logging and display
        version: Application version
        debug: Enable debug mode for development
        host: Server host address
        port: Server port number
    """
    
    model_config = ConfigDict(
        env_prefix='APP_',
        case_sensitive=False
    )
    
    env: str = Field(
        default="development",
        description="Application environment"
    )
    
    name: str = Field(
        default="AI Assistant",
        description="Application name"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode for development"
    )
    
    host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host address"
    )
    
    port: int = Field(
        default=8000,
        description="FastAPI server port number",
        ge=1024,  # Avoid privileged ports
        le=65535  # Valid port range
    )


class LoggingConfig(BaseModel):
    """
    Logging configuration settings.
    
    Controls how the application logs information for debugging and monitoring.
    Supports both console and file logging with configurable levels.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_path: Path to log file
        enable_file_logging: Whether to write logs to file
    """
    
    model_config = ConfigDict(
        env_prefix='LOG_',
        case_sensitive=False
    )
    
    level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    file_path: Path = Field(
        default=Path("logs/app.log"),
        description="Path to log file"
    )
    
    enable_file_logging: bool = Field(
        default=True,
        description="Enable file logging"
    )
    
    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """
        Validate logging level is valid.
        
        Args:
            v: The logging level string
            
        Returns:
            str: The validated logging level
            
        Raises:
            ValueError: If logging level is invalid
        """
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        
        return v_upper
    
    def model_post_init(self, __context) -> None:
        """Create logs directory if it doesn't exist."""
        if self.enable_file_logging:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """
    Main application settings container.
    
    Combines all configuration sections into a single settings object.
    This is the main class that other parts of the application will import.
    
    Usage:
        settings = Settings()
        api_key = settings.gemini.api_key
        db_path = settings.database.path
    
    Attributes:
        gemini: Google Gemini API configuration
        database: Database storage configuration
        app: General application settings
        logging: Logging configuration
    """
    
    # Nested configuration sections
    gemini: GeminiConfig
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    app: AppConfig = Field(default_factory=AppConfig) 
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @classmethod
    def load_from_env(cls) -> 'Settings':
        """
        Load configuration from environment variables.
        
        This method loads the .env file and creates a Settings object
        with all configuration values properly typed and validated.
        
        Returns:
            Settings: Fully configured settings object
            
        Raises:
            ValidationError: If required environment variables are missing
            FileNotFoundError: If .env file is required but not found
            
        Example:
            # Loads from .env file in project root
            config = Settings.load_from_env()
            print(f"Using database: {config.database.path}")
        """
        # Load environment variables from .env file
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print("⚠️  Warning: .env file not found. Using environment variables only.")
        
        # Create settings with validation
        try:
            return cls(
                gemini=GeminiConfig(
                    api_key=os.getenv('GEMINI_API_KEY', ''),
                    text_model=os.getenv('GEMINI_TEXT_MODEL', 'gemini-2.5-flash'),
                    audio_model=os.getenv('GEMINI_AUDIO_MODEL', 'gemini-2.5-flash'),
                    embedding_model=os.getenv('GEMINI_EMBEDDING_MODEL', 'models/embedding-001'),
                    rpm_limit=int(os.getenv('GEMINI_RPM_LIMIT', '10')),
                    rpd_limit=int(os.getenv('GEMINI_RPD_LIMIT', '250'))
                ),
                database=DatabaseConfig(),
                app=AppConfig(),
                logging=LoggingConfig()
            )
        except Exception as e:
            print(f"❌ Configuration Error: {e}")
            print("💡 Make sure your .env file is properly configured.")
            raise


@lru_cache()
def get_config() -> Settings:
    """
    Get application configuration (cached).
    
    This function uses LRU cache to ensure configuration is loaded only once
    and reused across the application. This improves performance and ensures
    consistent configuration throughout the app lifecycle.
    
    Returns:
        Settings: Cached configuration object
        
    Example:
        # This will load config once and cache it
        config = get_config()
        
        # Subsequent calls return the same cached instance
        config2 = get_config()  # Same object as config
        assert config is config2  # True
    """
    return Settings.load_from_env()


def print_config_summary() -> None:
    """
    Print a summary of current configuration for debugging.
    
    This utility function helps during development to verify that
    configuration is loaded correctly. It masks sensitive information
    like API keys for security.
    
    Example Output:
        🤖 AI Assistant Configuration
        ================================
        Environment: development
        Debug Mode: True
        Database: data/conversations.db
        Gemini Model: gemini-2.5-flash
        API Key: ****...ab12 (masked)
    """
    try:
        config = get_config()
        
        print("\n🤖 AI Assistant Configuration")
        print("=" * 40)
        print(f"Environment: {config.app.env}")
        print(f"Debug Mode: {config.app.debug}")
        print(f"Server: {config.app.host}:{config.app.port}")
        print(f"Database: {config.database.path}")
        print(f"Gemini Model: {config.gemini.text_model}")
        
        # Mask API key for security (show only first 4 and last 4 chars)
        api_key = config.gemini.api_key
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        print(f"API Key: {masked_key} (masked)")
        
        print(f"Log Level: {config.logging.level}")
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")


# Development utility - run this file directly to test configuration
if __name__ == "__main__":
    """
    Test configuration loading when run directly.
    
    Usage:
        python src/config.py
        
    This will load the configuration and print a summary,
    helping you verify that your .env file is set up correctly.
    """
    print("🔧 Testing configuration loading...")
    
    try:
        config = get_config()
        print("✅ Configuration loaded successfully!")
        print_config_summary()
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure .env file exists in project root")
        print("2. Check that GEMINI_API_KEY is set in .env")
        print("3. Verify .env file format (KEY=value, no quotes)")
        
        exit(1)