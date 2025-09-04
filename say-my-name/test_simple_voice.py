#!/usr/bin/env python3
"""Simple test for voice integration."""

import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from services.voice_enhanced_conversation_service import init_voice_enhanced_conversation_service

def test_with_audio(audio_file):
    if not Path(audio_file).exists():
        print(f"Audio file not found: {audio_file}")
        return
    
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    
    print(f"Testing voice integration with {audio_file} ({len(audio_data)/1024:.1f}KB)")
    
    # Initialize service
    service = init_voice_enhanced_conversation_service()
    
    # Create conversation
    conv_id = service.start_new_conversation("Voice Test")
    
    # Process voice message
    print("Processing voice message...")
    response = service.process_voice_message(conv_id, audio_data)
    
    if response.success:
        print(f"✅ Success!")
        print(f"User (voice): {response.user_message}")
        print(f"AI: {response.ai_content[:100]}...")
    else:
        print(f"❌ Failed: {response.error_message}")
    
    service.cleanup()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_with_audio(sys.argv[1])
    else:
        print("Usage: python test_simple_voice.py audio_file.m4a")