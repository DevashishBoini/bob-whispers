#!/usr/bin/env python3
"""Test the voice-enabled API endpoints."""

import requests
import json
import time
from pathlib import Path

def test_voice_api():
    """Test the voice API integration."""
    print("🧪 Testing Voice API Integration")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test API health first
    print('🔍 Testing API health...')
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code == 200:
            print('✅ API is healthy')
        else:
            print('❌ API health check failed')
            return False
    except requests.exceptions.RequestException as e:
        print(f'❌ API not reachable: {e}')
        print('Make sure to start the server first: python src/main.py')
        return False

    # Create a conversation first
    print('\n📝 Creating test conversation...')
    try:
        conv_response = requests.post(f'{base_url}/conversations', 
                                    json={'title': 'Voice API Test'})
        conv_data = conv_response.json()
        conv_id = conv_data['conversation_id']
        print(f'✅ Created conversation: {conv_id}')
    except Exception as e:
        print(f'❌ Failed to create conversation: {e}')
        return False

    # Test text message with TTS
    print('\n💬 Testing text message with TTS...')
    try:
        text_response = requests.post(f'{base_url}/chat/message',
                                    json={
                                        'conversation_id': conv_id,
                                        'message': 'Hello! Can you introduce yourself briefly?',
                                        'enable_tts': True
                                    })
        text_data = text_response.json()
        if text_response.status_code == 200 and text_data['success']:
            print(f'✅ Text + TTS successful!')
            print(f'   AI: {text_data["ai_response"][:100]}...')
            print(f'   Has TTS: {text_data["has_voice_response"]}')
            if text_data["tts_commands"]:
                print(f'   TTS Type: {text_data["tts_commands"]["type"]}')
        else:
            print(f'❌ Text message failed: {text_data}')
    except Exception as e:
        print(f'❌ Text message test failed: {e}')
    
    # Test voice message upload
    print('\n🎤 Testing voice message upload...')
    audio_file = Path("Gachibowli.m4a")
    if audio_file.exists():
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio_file': ('test.m4a', f, 'audio/mp4')}
                data = {
                    'conversation_id': conv_id,
                    'enable_tts': 'true'
                }
                voice_response = requests.post(f'{base_url}/chat/voice',
                                             files=files,
                                             data=data)
                voice_data = voice_response.json()
                
                if voice_response.status_code == 200 and voice_data['success']:
                    print(f'✅ Voice message successful!')
                    print(f'   User (voice): {voice_data.get("user_message", "N/A")}')
                    print(f'   AI: {voice_data["ai_response"][:100]}...')
                    print(f'   Has TTS: {voice_data["has_voice_response"]}')
                else:
                    print(f'❌ Voice message failed: {voice_data}')
        except Exception as e:
            print(f'❌ Voice message test failed: {e}')
    else:
        print('⚠️ No audio file found, skipping voice upload test')
    
    # Test conversation stats
    print('\n📊 Testing conversation stats...')
    try:
        stats_response = requests.get(f'{base_url}/conversations/stats')
        stats_data = stats_response.json()
        print(f'✅ Stats retrieved:')
        print(f'   Voice enabled: {stats_data.get("voice_processing_enabled", False)}')
        print(f'   TTS enabled: {stats_data.get("tts_enabled", False)}')
    except Exception as e:
        print(f'❌ Stats test failed: {e}')
    
    print('\n✅ Voice API integration tests completed!')
    return True

if __name__ == "__main__":
    test_voice_api()
