"""
Text-to-Speech Service

Provides text-to-speech capabilities using Web Speech API and other TTS providers.
Supports multiple TTS engines with fallback options for reliability.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import tempfile
import hashlib

logger = logging.getLogger(__name__)

class TTSProvider(Enum):
    """Available TTS providers."""
    WEB_SPEECH_API = "web_speech_api"
    BROWSER_NATIVE = "browser_native" 
    OFFLINE_CACHE = "offline_cache"

class TTSVoice(Enum):
    """Available voice types."""
    MALE = "male"
    FEMALE = "female" 
    NEUTRAL = "neutral"

@dataclass
class TTSRequest:
    """TTS generation request."""
    text: str
    voice: TTSVoice = TTSVoice.FEMALE
    rate: float = 1.0  # 0.1 to 2.0
    pitch: float = 1.0  # 0.0 to 2.0
    volume: float = 1.0  # 0.0 to 1.0
    language: str = "en-US"

@dataclass
class TTSResult:
    """TTS generation result."""
    success: bool
    audio_data: Optional[bytes] = None
    audio_url: Optional[str] = None
    speech_commands: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    provider_used: Optional[TTSProvider] = None
    duration_ms: Optional[int] = None

class TTSService:
    """
    Text-to-Speech service with multiple provider support.
    
    Provides voice synthesis capabilities with fallback options:
    1. Web Speech API (browser-based, real-time)
    2. Browser native TTS (client-side JavaScript)
    3. Offline cache (pre-generated common responses)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enable_cache: bool = True):
        """
        Initialize TTS service.
        
        Args:
            cache_dir: Directory for audio cache (optional)
            enable_cache: Whether to enable response caching
        """
        self.cache_dir = cache_dir or Path("data/tts_cache")
        self.enable_cache = enable_cache
        self.common_responses = self._load_common_responses()
        
        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TTS service initialized with cache: {self.enable_cache}")
    
    def generate_speech(self, request: TTSRequest) -> TTSResult:
        """
        Generate speech for the given text.
        
        Args:
            request: TTS generation request
            
        Returns:
            TTSResult with audio data or speech commands
        """
        try:
            # Check cache first
            if self.enable_cache:
                cached_result = self._check_cache(request)
                if cached_result:
                    return cached_result
            
            # Generate Web Speech API commands (primary method)
            speech_commands = self._generate_web_speech_commands(request)
            
            result = TTSResult(
                success=True,
                speech_commands=speech_commands,
                provider_used=TTSProvider.WEB_SPEECH_API
            )
            
            # Cache the result
            if self.enable_cache:
                self._cache_result(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return TTSResult(
                success=False,
                error_message=str(e)
            )
    
    def _generate_web_speech_commands(self, request: TTSRequest) -> Dict[str, Any]:
        """Generate Web Speech API commands for client-side execution."""
        
        # Map our voice types to Web Speech API voices
        voice_mapping = {
            TTSVoice.FEMALE: ["Google UK English Female", "Microsoft Zira", "female"],
            TTSVoice.MALE: ["Google UK English Male", "Microsoft David", "male"], 
            TTSVoice.NEUTRAL: ["Google US English", "Microsoft Mark", "default"]
        }
        
        return {
            "type": "web_speech_synthesis",
            "text": request.text,
            "options": {
                "rate": max(0.1, min(2.0, request.rate)),
                "pitch": max(0.0, min(2.0, request.pitch)), 
                "volume": max(0.0, min(1.0, request.volume)),
                "lang": request.language,
                "voicePreferences": voice_mapping.get(request.voice, voice_mapping[TTSVoice.FEMALE])
            },
            "fallback": {
                "type": "browser_native",
                "text": request.text
            }
        }
    
    def _check_cache(self, request: TTSRequest) -> Optional[TTSResult]:
        """Check if response is cached."""
        cache_key = self._generate_cache_key(request)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                return TTSResult(**cached_data)
            except Exception as e:
                logger.warning(f"Failed to load cached TTS result: {e}")
        
        return None
    
    def _cache_result(self, request: TTSRequest, result: TTSResult):
        """Cache TTS result."""
        try:
            cache_key = self._generate_cache_key(request)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Convert result to dict for JSON serialization
            cache_data = {
                "success": result.success,
                "speech_commands": result.speech_commands,
                "provider_used": result.provider_used.value if result.provider_used else None,
                "duration_ms": result.duration_ms
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache TTS result: {e}")
    
    def _generate_cache_key(self, request: TTSRequest) -> str:
        """Generate cache key for request."""
        content = f"{request.text}_{request.voice.value}_{request.rate}_{request.pitch}_{request.volume}_{request.language}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_common_responses(self) -> Dict[str, str]:
        """Load common AI responses for pre-caching."""
        return {
            "thinking": "Let me think about that...",
            "error": "I'm sorry, I encountered an error processing your request.",
            "clarify": "Could you please clarify what you mean?",
            "welcome": "Hello! How can I help you today?",
            "goodbye": "Goodbye! Have a great day!",
            "processing": "Processing your request...",
            "understood": "I understand.",
            "thanks": "You're welcome!"
        }
    
    def get_supported_voices(self) -> List[Dict[str, str]]:
        """Get list of supported voices."""
        return [
            {"id": "female", "name": "Female Voice", "language": "en-US"},
            {"id": "male", "name": "Male Voice", "language": "en-US"},
            {"id": "neutral", "name": "Neutral Voice", "language": "en-US"}
        ]
    
    def preload_common_responses(self):
        """Pre-generate common responses for faster playback."""
        logger.info("Pre-loading common TTS responses...")
        
        for key, text in self.common_responses.items():
            request = TTSRequest(text=text)
            result = self.generate_speech(request)
            
            if result.success:
                logger.debug(f"Pre-loaded TTS for: {key}")
            else:
                logger.warning(f"Failed to pre-load TTS for: {key}")
        
        logger.info(f"Pre-loaded {len(self.common_responses)} common responses")

def init_tts_service(cache_dir: Optional[Path] = None, enable_cache: bool = True) -> TTSService:
    """
    Initialize TTS service.
    
    Args:
        cache_dir: Directory for audio cache
        enable_cache: Whether to enable caching
        
    Returns:
        Initialized TTS service
    """
    service = TTSService(cache_dir=cache_dir, enable_cache=enable_cache)
    
    # Pre-load common responses in background
    if enable_cache:
        try:
            service.preload_common_responses()
        except Exception as e:
            logger.warning(f"Failed to pre-load common responses: {e}")
    
    logger.info("TTS service initialization completed")
    return service

# Test function
if __name__ == "__main__":
    """Test TTS service functionality."""
    
    # Initialize service
    tts = init_tts_service()
    
    # Test speech generation
    request = TTSRequest(
        text="Hello! This is a test of the text-to-speech system.",
        voice=TTSVoice.FEMALE,
        rate=1.0
    )
    
    result = tts.generate_speech(request)
    
    if result.success:
        print("✅ TTS test successful!")
        print(f"Provider: {result.provider_used}")
        print(f"Speech commands: {json.dumps(result.speech_commands, indent=2)}")
    else:
        print(f"❌ TTS test failed: {result.error_message}")
    
    # Test voice listing
    voices = tts.get_supported_voices()
    print(f"\nSupported voices: {len(voices)}")
    for voice in voices:
        print(f"  - {voice['name']} ({voice['id']})")