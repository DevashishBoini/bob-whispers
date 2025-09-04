#!/usr/bin/env python3
"""
Real Audio Testing Script

This script allows testing the audio service with actual audio files.
You can record audio using your system's recorder or browser and test
the complete audio processing pipeline.

Usage:
    python test_real_audio.py path/to/audio/file.wav
    
Or for interactive testing:
    python test_real_audio.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.audio_service import init_audio_service
from services.gemini_audio_client import init_gemini_audio_client


def test_audio_file(file_path: str):
    """Test audio processing with a real audio file."""
    print(f"\n🎤 Testing with audio file: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ Audio file not found: {file_path}")
        return False
    
    try:
        # Read the audio file
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"📁 File size: {len(audio_data) / 1024:.1f} KB")
        
        # Test 1: Audio service validation
        print("\n1. Testing audio service...")
        audio_service = init_audio_service()
        
        # Validate audio
        is_valid, message = audio_service.validate_audio_data(audio_data)
        print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'} - {message}")
        
        # Get audio info
        audio_info = audio_service.get_audio_info(audio_data)
        print(f"Audio Info:")
        for key, value in audio_info.items():
            print(f"  {key}: {value}")
        
        if not is_valid:
            return False
        
        # Test 2: Gemini Audio Client
        print("\n2. Testing Gemini audio transcription...")
        audio_client = init_gemini_audio_client()
        
        # Attempt transcription
        transcription_result = audio_client.transcribe_audio(audio_data)
        
        if transcription_result.success:
            print(f"✅ Transcription successful!")
            print(f"   Transcript: '{transcription_result.transcript}'")
            print(f"   Confidence: {transcription_result.confidence}")
            print(f"   Processing time: {transcription_result.processing_time:.2f}s")
        else:
            print(f"❌ Transcription failed: {transcription_result.error_message}")
        
        return transcription_result.success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def interactive_test():
    """Interactive testing mode."""
    print("\n🎙️ Interactive Audio Testing")
    print("=" * 50)
    print("To test with real audio, you have several options:")
    print("")
    print("1. Record audio using your system:")
    print("   - Windows: Voice Recorder app")
    print("   - Mac: QuickTime Player > File > New Audio Recording")
    print("   - Linux: Record with `arecord test.wav`")
    print("")
    print("2. Record using browser:")
    print("   - Go to https://online-voice-recorder.com")
    print("   - Record a short message (2-10 seconds)")
    print("   - Download as WAV file")
    print("")
    print("3. Use existing audio file:")
    print("   - Any WAV, MP3, M4A file")
    print("")
    
    while True:
        file_path = input("Enter path to audio file (or 'quit' to exit): ").strip()
        
        if file_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if file_path:
            test_audio_file(file_path)
            print("\n" + "=" * 50)
        else:
            print("Please enter a file path.")


def main():
    """Main testing function."""
    if len(sys.argv) > 1:
        # Test with provided file
        audio_file = sys.argv[1]
        success = test_audio_file(audio_file)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        interactive_test()


if __name__ == "__main__":
    main()