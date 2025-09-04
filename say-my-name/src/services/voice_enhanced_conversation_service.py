# """
# Voice-Enhanced Conversation Service - Simplified

# This module extends the enhanced conversation service with voice message processing.
# Voice input is transcribed to text and then processed through the existing conversation
# pipeline exactly like a text message, with no additional complexity.

# Key Features:
# - Voice transcription using Gemini Audio API
# - Direct integration with existing conversation pipeline
# - Voice messages treated identically to text messages after transcription
# - Same storage, same semantic memory, same responses

# Usage Example:
#     service = init_voice_enhanced_conversation_service()
    
#     # Process voice message (returns same ConversationResponse as text)
#     response = service.process_voice_message(conv_id, audio_data)
    
#     # Process text message (unchanged)
#     response = service.process_message(conv_id, "text message", "text")
# """

# from typing import Optional

# # Import existing services
# try:
#     from .enhanced_conversation_service import EnhancedConversationService, init_enhanced_conversation_service
#     from .conversation_service import ConversationResponse
#     from .gemini_audio_client import GeminiAudioClient, init_gemini_audio_client
#     from .audio_service import AudioService, init_audio_service
# except ImportError:
#     try:
#         import sys
#         import os
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#         from services.enhanced_conversation_service import EnhancedConversationService, init_enhanced_conversation_service
#         from services.conversation_service import ConversationResponse
#         from services.gemini_audio_client import GeminiAudioClient, init_gemini_audio_client
#         from services.audio_service import AudioService, init_audio_service
#     except ImportError:
#         from src.services.enhanced_conversation_service import EnhancedConversationService, init_enhanced_conversation_service
#         from src.services.conversation_service import ConversationResponse
#         from src.services.gemini_audio_client import GeminiAudioClient, init_gemini_audio_client
#         from src.services.audio_service import AudioService, init_audio_service


# class VoiceEnhancedConversationService:
#     """
#     Enhanced conversation service with voice message processing.
    
#     Simple extension that adds voice transcription capability while
#     using the existing conversation pipeline for all processing.
#     """
    
#     def __init__(self, enhanced_service: EnhancedConversationService, 
#                  audio_client: GeminiAudioClient, audio_service: AudioService):
#         """
#         Initialize voice-enhanced conversation service.
        
#         Args:
#             enhanced_service: Existing enhanced conversation service
#             audio_client: Gemini audio client for transcription
#             audio_service: Audio service for validation
#         """
#         self.enhanced_service = enhanced_service
#         self.audio_client = audio_client
#         self.audio_service = audio_service
        
#         print("Voice-enhanced conversation service initialized")
    
#     def process_voice_message(self, conversation_id: str, audio_data: bytes) -> ConversationResponse:
#         """
#         Process voice message by transcribing to text and using existing pipeline.
        
#         Args:
#             conversation_id: ID of conversation
#             audio_data: Raw audio file data
            
#         Returns:
#             ConversationResponse: Same response format as text messages
#         """
#         try:
#             print(f"Processing voice message in conversation {conversation_id[:8]}...")
            
#             # Step 1: Validate audio
#             is_valid, validation_message = self.audio_service.validate_audio_data(audio_data)
#             if not is_valid:
#                 return ConversationResponse(
#                     conversation_id=conversation_id,
#                     user_message="[Invalid audio]",
#                     ai_content="",
#                     success=False,
#                     error_message=f"Audio validation failed: {validation_message}",
#                     input_type="voice"
#                 )
            
#             # Step 2: Transcribe audio to text
#             print("Transcribing audio to text...")
#             transcription_result = self.audio_client.transcribe_audio(audio_data)
            
#             if not transcription_result.success:
#                 return ConversationResponse(
#                     conversation_id=conversation_id,
#                     user_message="[Transcription failed]",
#                     ai_content="",
#                     success=False,
#                     error_message=f"Transcription failed: {transcription_result.error_message}",
#                     input_type="voice"
#                 )
            
#             # Step 3: Process transcribed text through existing pipeline
#             transcribed_text = transcription_result.transcript
#             print(f"Transcribed: '{transcribed_text[:50]}...'")
            
#             return self.enhanced_service.process_message(
#                 conversation_id=conversation_id,
#                 user_message=transcribed_text,
#                 input_type="voice"
#             )
            
#         except Exception as e:
#             print(f"Voice processing error: {e}")
#             return ConversationResponse(
#                 conversation_id=conversation_id,
#                 user_message="[Voice processing error]",
#                 ai_content="",
#                 success=False,
#                 error_message=f"Voice processing failed: {str(e)}",
#                 input_type="voice"
#             )
    
#     # Delegate all other methods to enhanced service
#     def process_message(self, conversation_id: str, user_message: str, 
#                        input_type: str = 'text') -> ConversationResponse:
#         """Process text message (delegates to enhanced service)."""
#         return self.enhanced_service.process_message(conversation_id, user_message, input_type)
    
#     def start_new_conversation(self, title: Optional[str] = None) -> str:
#         """Start new conversation (delegates to enhanced service)."""
#         return self.enhanced_service.start_new_conversation(title)
    
#     def get_conversation_list(self):
#         """Get conversation list (delegates to enhanced service)."""
#         return self.enhanced_service.get_conversation_list()
    
#     def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None):
#         """Get conversation history (delegates to enhanced service)."""
#         return self.enhanced_service.get_conversation_history(conversation_id, limit)
    
#     def delete_conversation(self, conversation_id: str) -> bool:
#         """Delete conversation (delegates to enhanced service)."""
#         return self.enhanced_service.delete_conversation(conversation_id)
    
#     def get_enhanced_stats(self):
#         """Get enhanced statistics (delegates to enhanced service)."""
#         stats = self.enhanced_service.get_enhanced_stats()
#         stats["voice_processing_enabled"] = True
#         return stats
    
#     def cleanup(self):
#         """Clean up resources."""
#         self.enhanced_service.cleanup()


# def init_voice_enhanced_conversation_service() -> VoiceEnhancedConversationService:
#     """
#     Initialize voice-enhanced conversation service with all dependencies.
    
#     Returns:
#         VoiceEnhancedConversationService: Service with voice capabilities
#     """
#     try:
#         print("Initializing voice-enhanced conversation service...")
        
#         # Initialize dependencies
#         enhanced_service = init_enhanced_conversation_service()
#         audio_client = init_gemini_audio_client()
#         audio_service = init_audio_service()
        
#         # Create simplified voice service
#         voice_service = VoiceEnhancedConversationService(
#             enhanced_service, audio_client, audio_service
#         )
        
#         print("Voice-enhanced conversation service ready")
#         return voice_service
        
#     except Exception as e:
#         print(f"Failed to initialize voice service: {e}")
#         raise


# def test_voice_service():
#     """Test voice service functionality."""
#     print("\nTesting simplified voice service...")
    
#     try:
#         # Initialize service
#         service = init_voice_enhanced_conversation_service()
        
#         # Create conversation
#         conv_id = service.start_new_conversation("Voice Test")
#         print(f"Created conversation: {conv_id}")
        
#         # Test text message (should work normally)
#         text_response = service.process_message(
#             conv_id, "This is a text message", "text"
#         )
#         print(f"Text message: {'✅' if text_response.success else '❌'}")
        
#         # Test voice with dummy data (will fail transcription gracefully)
#         dummy_audio = b"dummy_audio" * 1000  # Make it pass size validation
#         voice_response = service.process_voice_message(conv_id, dummy_audio)
#         print(f"Voice message handling: {'✅' if not voice_response.success else 'Expected failure'}")
        
#         # Cleanup
#         service.cleanup()
        
#         print("✅ Simplified voice service tests passed")
        
#     except Exception as e:
#         print(f"❌ Voice service test failed: {e}")


# if __name__ == "__main__":
#     test_voice_service()


"""
Voice-Enhanced Conversation Service - With TTS Support

This module extends the enhanced conversation service with voice message processing
and text-to-speech response capabilities for full voice conversations.

Key Features:
- Voice transcription using Gemini Audio API
- Text-to-speech response generation
- Direct integration with existing conversation pipeline
- Voice messages treated identically to text messages after transcription
- Same storage, same semantic memory, same responses
- Optional voice responses with TTS controls

Usage Example:
    service = init_voice_enhanced_conversation_service()
    
    # Process voice message with optional TTS response
    response = service.process_voice_message(conv_id, audio_data, enable_tts=True)
    
    # Process text message (unchanged)
    response = service.process_message(conv_id, "text message", "text")
"""

import sys
import os
from typing import Optional

# Import existing services
try:
    from .enhanced_conversation_service import EnhancedConversationService, init_enhanced_conversation_service
    from .conversation_service import ConversationResponse
    from .gemini_audio_client import GeminiAudioClient, init_gemini_audio_client
    from .audio_service import AudioService, init_audio_service
    from .tts_service import TTSService, TTSRequest, TTSVoice, init_tts_service
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from services.enhanced_conversation_service import EnhancedConversationService, init_enhanced_conversation_service
        from services.conversation_service import ConversationResponse
        from services.gemini_audio_client import GeminiAudioClient, init_gemini_audio_client
        from services.audio_service import AudioService, init_audio_service
        from services.tts_service import TTSService, TTSRequest, TTSVoice, init_tts_service
    except ImportError:
        from src.services.enhanced_conversation_service import EnhancedConversationService, init_enhanced_conversation_service
        from src.services.conversation_service import ConversationResponse
        from src.services.gemini_audio_client import GeminiAudioClient, init_gemini_audio_client
        from src.services.audio_service import AudioService, init_audio_service
        from src.services.tts_service import TTSService, TTSRequest, TTSVoice, init_tts_service

class VoiceEnhancedConversationResponse:
    """Enhanced conversation response with TTS support."""
    
    def __init__(self, base_response: ConversationResponse, tts_commands: Optional[dict] = None):
        # Copy all attributes from base response
        self.conversation_id = base_response.conversation_id
        self.user_message = base_response.user_message
        self.ai_content = base_response.ai_content
        self.success = base_response.success
        self.error_message = base_response.error_message
        self.input_type = getattr(base_response, 'input_type', 'text')
        self.memories_used = getattr(base_response, 'memories_used', [])
        self.context_summary = getattr(base_response, 'context_summary', '')
        
        # Add TTS capabilities
        self.tts_commands = tts_commands
        self.has_voice_response = tts_commands is not None

class VoiceEnhancedConversationService:
    """
    Enhanced conversation service with voice message processing and TTS responses.
    
    Extends EnhancedConversationService with:
    - Voice input transcription
    - Text-to-speech response generation
    - Full voice conversation capabilities
    """
    
    def __init__(self, enhanced_service: EnhancedConversationService, 
                 audio_client: GeminiAudioClient, audio_service: AudioService,
                 tts_service: TTSService):
        """
        Initialize voice-enhanced conversation service.
        
        Args:
            enhanced_service: Existing enhanced conversation service
            audio_client: Gemini audio client for transcription
            audio_service: Audio service for validation
            tts_service: TTS service for voice responses
        """
        self.enhanced_service = enhanced_service
        self.audio_client = audio_client
        self.audio_service = audio_service
        self.tts_service = tts_service
        
        # TTS settings
        self.default_voice = TTSVoice.FEMALE
        self.default_rate = 1.0
        self.default_pitch = 1.0
        self.default_volume = 0.8
        
        print("Voice-enhanced conversation service initialized with TTS support")
    
    def process_voice_message(self, conversation_id: str, audio_data: bytes, 
                            enable_tts: bool = True, voice: TTSVoice = None) -> VoiceEnhancedConversationResponse:
        """
        Process voice message by transcribing to text and using existing pipeline.
        
        Args:
            conversation_id: ID of conversation
            audio_data: Raw audio file data
            enable_tts: Whether to generate TTS response
            voice: Voice type for TTS response
            
        Returns:
            VoiceEnhancedConversationResponse: Response with optional TTS commands
        """
        try:
            print(f"Processing voice message in conversation {conversation_id[:8]}...")
            
            # Step 1: Validate audio
            is_valid, validation_message = self.audio_service.validate_audio_data(audio_data)
            if not is_valid:
                return self._create_error_response(
                    conversation_id, 
                    f"Audio validation failed: {validation_message}",
                    input_type="voice"
                )
            
            # Step 2: Transcribe audio to text
            print("Transcribing audio to text...")
            transcription_result = self.audio_client.transcribe_audio(audio_data)
            
            if not transcription_result.success:
                return self._create_error_response(
                    conversation_id,
                    f"Transcription failed: {transcription_result.error_message}",
                    input_type="voice"
                )
            
            transcribed_text = transcription_result.transcript
            print(f"Transcribed: '{transcribed_text[:50]}...'")
            
            # Step 3: Process as text message through existing pipeline
            base_response = self.enhanced_service.process_message_with_memory(
                conversation_id=conversation_id,
                user_message=transcribed_text,
                input_type="voice"
            )
            
            # Step 4: Generate TTS response if requested and successful
            tts_commands = None
            if enable_tts and base_response.success and base_response.ai_content:
                tts_commands = self._generate_tts_response(
                    base_response.ai_content, 
                    voice or self.default_voice
                )
            
            return VoiceEnhancedConversationResponse(base_response, tts_commands)
            
        except Exception as e:
            print(f"Error processing voice message: {e}")
            return self._create_error_response(
                conversation_id,
                f"Voice processing failed: {str(e)}",
                input_type="voice"
            )
    
    def process_message_with_tts(self, conversation_id: str, user_message: str, 
                               input_type: str = "text", enable_tts: bool = True,
                               voice: TTSVoice = None) -> VoiceEnhancedConversationResponse:
        """
        Process text message with optional TTS response.
        
        Args:
            conversation_id: ID of conversation
            user_message: User's message text
            input_type: Type of input ("text", "voice", etc.)
            enable_tts: Whether to generate TTS response
            voice: Voice type for TTS response
            
        Returns:
            VoiceEnhancedConversationResponse: Response with optional TTS commands
        """
        try:
            # Process through existing pipeline
            base_response = self.enhanced_service.process_message_with_memory(
                conversation_id=conversation_id,
                user_message=user_message,
                input_type=input_type
            )
            
            # Generate TTS response if requested and successful
            tts_commands = None
            if enable_tts and base_response.success and base_response.ai_content:
                tts_commands = self._generate_tts_response(
                    base_response.ai_content,
                    voice or self.default_voice
                )
            
            return VoiceEnhancedConversationResponse(base_response, tts_commands)
            
        except Exception as e:
            return self._create_error_response(
                conversation_id,
                f"Message processing failed: {str(e)}",
                input_type=input_type
            )
    
    def _generate_tts_response(self, ai_content: str, voice: TTSVoice) -> Optional[dict]:
        """Generate TTS commands for AI response."""
        try:
            print(f"Generating TTS response with {voice.value} voice...")
            
            tts_request = TTSRequest(
                text=ai_content,
                voice=voice,
                rate=self.default_rate,
                pitch=self.default_pitch,
                volume=self.default_volume
            )
            
            tts_result = self.tts_service.generate_speech(tts_request)
            
            if tts_result.success:
                print(f"✅ TTS generated using {tts_result.provider_used.value}")
                return tts_result.speech_commands
            else:
                print(f"⚠️ TTS generation failed: {tts_result.error_message}")
                return None
                
        except Exception as e:
            print(f"Error generating TTS response: {e}")
            return None
    
    def _create_error_response(self, conversation_id: str, error_message: str, 
                             input_type: str = "text") -> VoiceEnhancedConversationResponse:
        """Create error response."""
        error_response = ConversationResponse(
            conversation_id=conversation_id,
            user_message="[Error]",
            ai_content="",
            success=False,
            error_message=error_message,
            input_type=input_type
        )
        return VoiceEnhancedConversationResponse(error_response)
    
    def set_tts_settings(self, voice: TTSVoice = None, rate: float = None, 
                        pitch: float = None, volume: float = None):
        """Update default TTS settings."""
        if voice is not None:
            self.default_voice = voice
        if rate is not None:
            self.default_rate = max(0.1, min(2.0, rate))
        if pitch is not None:
            self.default_pitch = max(0.0, min(2.0, pitch))
        if volume is not None:
            self.default_volume = max(0.0, min(1.0, volume))
        
        print(f"Updated TTS settings: voice={self.default_voice.value}, rate={self.default_rate}, pitch={self.default_pitch}, volume={self.default_volume}")
    
    def get_supported_voices(self):
        """Get supported TTS voices."""
        return self.tts_service.get_supported_voices()
    
    # Delegate all other methods to enhanced service
    def start_new_conversation(self, title: Optional[str] = None):
        """Start a new conversation."""
        return self.enhanced_service.start_new_conversation(title)
    
    def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None):
        """Get conversation message history."""
        return self.enhanced_service.get_conversation_history(conversation_id, limit)
    
    def get_conversation_list(self):
        """List all conversations."""
        return self.enhanced_service.get_conversation_list()
    
    def list_conversations(self):
        """List all conversations (alias for get_conversation_list)."""
        return self.enhanced_service.get_conversation_list()
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        return self.enhanced_service.delete_conversation(conversation_id)
    
    def get_conversation_stats(self):
        """Get conversation statistics."""
        stats = self.enhanced_service.get_enhanced_stats()
        stats["voice_processing_enabled"] = True
        stats["tts_enabled"] = True
        return stats
    
    def get_enhanced_stats(self):
        """Get enhanced statistics."""
        stats = self.enhanced_service.get_enhanced_stats()
        stats["voice_processing_enabled"] = True
        stats["tts_enabled"] = True
        return stats
    
    def process_message(self, conversation_id: str, user_message: str, input_type: str = 'text') -> ConversationResponse:
        """Process text message (backward compatibility)."""
        # For backward compatibility, convert to regular ConversationResponse
        voice_response = self.process_message_with_tts(
            conversation_id=conversation_id,
            user_message=user_message,
            input_type=input_type,
            enable_tts=False  # Disable TTS for backward compatibility
        )
        
        # Convert VoiceEnhancedConversationResponse to ConversationResponse
        return ConversationResponse(
            conversation_id=voice_response.conversation_id,
            user_message=voice_response.user_message,
            ai_content=voice_response.ai_content,
            success=voice_response.success,
            error_message=voice_response.error_message,
            input_type=getattr(voice_response, 'input_type', 'text')
        )
    
    def cleanup(self):
        """Clean up resources."""
        self.enhanced_service.cleanup()
        print("Voice-enhanced conversation service cleanup completed")

def init_voice_enhanced_conversation_service():
    """
    Initialize voice-enhanced conversation service with TTS support.
    
    Returns:
        VoiceEnhancedConversationService: Fully initialized service
    """
    print("Initializing voice-enhanced conversation service with TTS...")
    
    try:
        # Initialize base enhanced service
        enhanced_service = init_enhanced_conversation_service()
        
        # Initialize audio services
        print("Initializing audio services...")
        gemini_audio_client = init_gemini_audio_client()
        audio_service = init_audio_service()
        
        # Initialize TTS service
        print("Initializing TTS service...")
        tts_service = init_tts_service()
        
        # Create voice-enhanced service
        service = VoiceEnhancedConversationService(
            enhanced_service=enhanced_service,
            audio_client=gemini_audio_client,
            audio_service=audio_service,
            tts_service=tts_service
        )
        
        print("Voice-enhanced conversation service with TTS ready")
        return service
        
    except Exception as e:
        print(f"Failed to initialize voice-enhanced conversation service: {e}")
        raise

def test_voice_enhanced_service():
    """Test the voice-enhanced conversation service with TTS."""
    print("Testing voice-enhanced conversation service with TTS support...")
    
    try:
        # Initialize service
        service = init_voice_enhanced_conversation_service()
        
        # Create test conversation
        conv_id = service.start_new_conversation("TTS Test")
        print(f"Created conversation: {conv_id}")
        
        # Test text message with TTS
        print("\nTesting text message with TTS...")
        response = service.process_message_with_tts(
            conversation_id=conv_id,
            user_message="Hello! Can you introduce yourself?",
            enable_tts=True,
            voice=TTSVoice.FEMALE
        )
        
        if response.success:
            print(f"✅ Text + TTS successful!")
            print(f"AI: {response.ai_content[:100]}...")
            if response.has_voice_response and response.tts_commands:
                print(f"🔊 TTS Commands: {response.tts_commands['type']}")
            else:
                print("⚠️ No TTS commands generated")
        else:
            print(f"❌ Failed: {response.error_message}")
        
        # Test voice settings
        print("\nTesting TTS settings...")
        service.set_tts_settings(voice=TTSVoice.MALE, rate=1.2, volume=0.9)
        
        voices = service.get_supported_voices()
        print(f"Supported voices: {len(voices)}")
        
        # Cleanup
        service.cleanup()
        print("✅ Voice-enhanced service test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    """Run voice-enhanced service tests."""
    test_voice_enhanced_service()