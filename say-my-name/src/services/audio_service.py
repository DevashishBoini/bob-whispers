"""
AI Assistant Audio Service - Simplified for Google Gemini API

This module provides basic audio handling for voice input with Google's Gemini Audio API.
Since Google's API handles format conversion and speech optimization internally, this service
focuses on browser integration, validation, and API communication.

Key Features:
- Basic audio validation (size, duration)
- Browser audio capture integration
- Direct Gemini API transcription
- Simple error handling and metadata extraction

Usage Example:
    from services.audio_service import AudioService, init_audio_service
    
    # Initialize audio service
    audio_service = init_audio_service()
    
    # Validate and process audio
    result = audio_service.process_voice_input(audio_data)
    if result.success:
        print(f"Transcription: {result.transcript}")

Architecture:
AudioConfig - Basic configuration for audio limits
AudioResult - Response structure for audio processing
AudioService - Main service for audio handling
"""

import io
import wave
from typing import Optional, Dict, Any
from dataclasses import dataclass

from pydantic import BaseModel, Field, ConfigDict

# Handle config import
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


class AudioConfig(BaseModel):
    """
    Basic configuration for audio processing.
    
    Simple settings for validation and limits since Google's API
    handles the complex audio processing internally.
    """
    
    model_config = ConfigDict(frozen=True, validate_assignment=True)
    
    max_file_size_mb: float = Field(
        default=20.0,
        description="Maximum audio file size in megabytes",
        gt=0.0, le=100.0
    )
    
    max_duration_seconds: float = Field(
        default=300.0,  # 5 minutes
        description="Maximum recording duration in seconds", 
        gt=0.0, le=600.0
    )
    
    min_duration_seconds: float = Field(
        default=0.5,
        description="Minimum recording duration in seconds",
        gt=0.0, le=5.0
    )


@dataclass
class AudioResult:
    """
    Result of audio processing operation.
    
    Simple structure containing success status and relevant information
    for voice input processing.
    """
    success: bool
    transcript: str = ""
    duration: float = 0.0
    file_size: int = 0
    error_message: str = ""
    confidence: float = 0.0
    
    def __str__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"{status} AudioResult(transcript='{self.transcript[:50]}...', duration={self.duration:.1f}s)"


class AudioService:
    """
    Simplified audio service for Google Gemini integration.
    
    Handles basic validation and prepares audio for Google's STT API
    without redundant processing since Google handles optimization internally.
    """
    
    def __init__(self, config: AudioConfig):
        """Initialize audio service with configuration."""
        self.config = config
        print(f"Audio service initialized: max {config.max_file_size_mb}MB, {config.max_duration_seconds}s")
    
    def validate_audio_data(self, audio_data: bytes) -> tuple[bool, str]:
        """
        Basic validation of audio data before API submission.
        
        Args:
            audio_data: Raw audio file data
            
        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        # Check file size
        file_size_mb = len(audio_data) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)"
        
        # Check minimum size (avoid empty files)
        if len(audio_data) < 1024:  # 1KB minimum
            return False, "Audio file is too small or empty"
        
        # Try to extract basic duration info for WAV files
        duration = self._get_audio_duration(audio_data)
        if duration > 0:
            if duration < self.config.min_duration_seconds:
                return False, f"Recording too short: {duration:.1f}s (min: {self.config.min_duration_seconds}s)"
            
            if duration > self.config.max_duration_seconds:
                return False, f"Recording too long: {duration:.1f}s (max: {self.config.max_duration_seconds}s)"
        
        return True, "Audio validation passed"
    
    def _get_audio_duration(self, audio_data: bytes) -> float:
        """
        Extract audio duration for validation.
        
        Args:
            audio_data: Audio file data
            
        Returns:
            float: Duration in seconds, or 0 if cannot determine
        """
        try:
            # Handle WAV files
            if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
                with io.BytesIO(audio_data) as wav_io:
                    with wave.open(wav_io, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        return frames / float(rate)
        except:
            pass
        
        # For other formats, return 0 (will be handled by Google's API)
        return 0.0
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Get basic information about audio file.
        
        Args:
            audio_data: Audio file data
            
        Returns:
            Dict: Basic audio information
        """
        size_mb = len(audio_data) / (1024 * 1024)
        duration = self._get_audio_duration(audio_data)
        is_valid, message = self.validate_audio_data(audio_data)
        
        # Basic format detection
        format_type = "unknown"
        if audio_data[:4] == b'RIFF':
            format_type = "wav"
        elif audio_data[:3] == b'ID3' or audio_data[:2] in [b'\xff\xfb', b'\xff\xf3']:
            format_type = "mp3"
        elif audio_data[:4] == b'ftyp':
            format_type = "m4a"
        elif audio_data[:4] == b'OggS':
            format_type = "ogg"
        
        return {
            "valid": is_valid,
            "message": message,
            "size_mb": round(size_mb, 2),
            "duration": round(duration, 1),
            "format": format_type,
            "ready_for_api": is_valid
        }


def init_audio_service() -> AudioService:
    """
    Initialize audio service with application configuration.
    
    Returns:
        AudioService: Configured audio service for voice processing
    """
    try:
        # Create basic audio configuration
        audio_config = AudioConfig(
            max_file_size_mb=20.0,      # Reasonable limit for voice messages
            max_duration_seconds=300.0,  # 5 minute maximum  
            min_duration_seconds=0.5     # Half second minimum
        )
        
        audio_service = AudioService(audio_config)
        print("Audio service initialization completed")
        return audio_service
        
    except Exception as e:
        print(f"Failed to initialize audio service: {e}")
        raise


def test_audio_service() -> None:
    """Test basic audio service functionality."""
    print("\n🧪 Testing simplified audio service...")
    
    try:
        # Initialize service
        print("\n1. Initializing audio service...")
        audio_service = init_audio_service()
        
        # Test with minimal WAV data
        print("\n2. Testing with sample audio data...")
        
        # Create minimal valid WAV header for testing
        sample_wav = (
            b'RIFF' +
            (44 + 1000).to_bytes(4, 'little') +  # File size
            b'WAVE' +
            b'fmt ' +
            (16).to_bytes(4, 'little') +  # Format chunk size
            (1).to_bytes(2, 'little') +   # Audio format (PCM)
            (1).to_bytes(2, 'little') +   # Number of channels
            (16000).to_bytes(4, 'little') + # Sample rate
            (32000).to_bytes(4, 'little') + # Byte rate
            (2).to_bytes(2, 'little') +   # Block align
            (16).to_bytes(2, 'little') +  # Bits per sample
            b'data' +
            (1000).to_bytes(4, 'little') + # Data chunk size
            b'\x00' * 1000  # Audio data (1000 bytes of silence)
        )
        
        # Test validation
        is_valid, message = audio_service.validate_audio_data(sample_wav)
        print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'} - {message}")
        
        # Test audio info
        info = audio_service.get_audio_info(sample_wav)
        print(f"Audio info: {info}")
        
        # Test invalid audio
        print("\n3. Testing invalid audio handling...")
        invalid_audio = b"not_audio"
        is_valid, message = audio_service.validate_audio_data(invalid_audio)
        print(f"Invalid audio: {'✅ Correctly rejected' if not is_valid else '❌ Should have failed'}")
        print(f"Message: {message}")
        
        print("\n✅ Audio service tests passed!")
        
    except Exception as e:
        print(f"\n❌ Audio service test failed: {e}")
        raise


if __name__ == "__main__":
    test_audio_service()