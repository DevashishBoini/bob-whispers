#!/usr/bin/env python3
"""
Gemini Audio Client Test Script

This script tests the Gemini audio client integration with proper import paths.

Usage:
    python test_gemini_audio_client.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.gemini_audio_client import init_gemini_audio_client


def test_gemini_audio_integration():
    """Test Gemini audio integration functionality."""
    print("\n🧪 Testing Gemini audio integration...")
    
    try:
        # Test 1: Initialize audio client
        print("\n1. Initializing Gemini audio client...")
        audio_client = init_gemini_audio_client()
        
        # Test 2: Test base functionality
        print("\n2. Testing base text functionality...")
        text_response = audio_client.generate_response("Hello, this is a test message.")
        if text_response.content:
            print(f"✅ Text generation working: '{text_response.content[:50]}...'")
        else:
            print("❌ Text generation failed")
        
        # Test 3: Test audio transcription readiness
        print("\n3. Testing audio transcription readiness...")
        audio_ready = audio_client.test_audio_transcription()
        
        # Test 4: Test with dummy audio data (will fail gracefully)
        print("\n4. Testing audio transcription error handling...")
        dummy_audio = b"fake_audio_data"
        transcription_result = audio_client.transcribe_audio(dummy_audio)
        
        if not transcription_result.success:
            print(f"✅ Error handling working: {transcription_result.error_message}")
        else:
            print("❌ Should have failed with dummy data")
        
        print("\n✅ Gemini audio integration tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Gemini audio integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    success = test_gemini_audio_integration()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()