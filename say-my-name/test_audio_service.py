#!/usr/bin/env python3
"""
Audio Service Test with Real Audio Files

This script tests the audio service with actual audio files instead of
synthetic data. It validates the complete audio processing pipeline.

Usage:
    python test_audio_service.py path/to/audio/file.wav
    
Or for interactive testing:
    python test_audio_service.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.audio_service import init_audio_service


def test_audio_service_with_file(file_path: str):
    """Test audio service with a real audio file."""
    print(f"\n🎵 Testing audio service with file: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ Audio file not found: {file_path}")
        return False
    
    try:
        # Read the audio file
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        
        print(f"📁 File size: {len(audio_data)} bytes ({len(audio_data) / 1024:.1f} KB)")
        
        # Initialize audio service
        print("\n1. Initializing audio service...")
        audio_service = init_audio_service()
        
        # Test validation
        print("\n2. Testing audio validation...")
        is_valid, message = audio_service.validate_audio_data(audio_data)
        print(f"Validation: {'✅ Valid' if is_valid else '❌ Invalid'} - {message}")
        
        # Test audio info extraction
        print("\n3. Testing audio info extraction...")
        audio_info = audio_service.get_audio_info(audio_data)
        print("Audio Information:")
        for key, value in audio_info.items():
            print(f"  {key}: {value}")
        
        # Test with various file sizes (create truncated versions)
        print("\n4. Testing with different file sizes...")
        
        # Test very small file (should fail)
        tiny_audio = audio_data[:100]  # Just 100 bytes
        is_valid, message = audio_service.validate_audio_data(tiny_audio)
        print(f"Tiny file (100 bytes): {'❌ Correctly rejected' if not is_valid else '✅ Unexpectedly accepted'}")
        print(f"  Message: {message}")
        
        # Test empty file (should fail)
        empty_audio = b""
        is_valid, message = audio_service.validate_audio_data(empty_audio)
        print(f"Empty file: {'❌ Correctly rejected' if not is_valid else '✅ Unexpectedly accepted'}")
        print(f"  Message: {message}")
        
        # Test file info for different sizes
        print("\n5. Testing audio info with different data sizes...")
        for size, label in [(1000, "1KB"), (10000, "10KB"), (len(audio_data), "Full")]:
            if len(audio_data) >= size:
                test_data = audio_data[:size]
                info = audio_service.get_audio_info(test_data)
                print(f"  {label} sample: {info['size_mb']:.3f}MB, {info['duration']}s, {info['format']}, valid: {info['valid']}")
        
        return audio_info['valid']
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_audio_service_comprehensive():
    """Comprehensive test of audio service without requiring files."""
    print("\n🧪 Testing audio service comprehensively...")
    
    try:
        # Test 1: Initialize service
        print("\n1. Initializing audio service...")
        audio_service = init_audio_service()
        
        # Test 2: Test various invalid inputs
        print("\n2. Testing invalid inputs...")
        
        invalid_cases = [
            (b"", "Empty data"),
            (b"x" * 100, "Too small (100 bytes)"),
            (b"not_audio_data", "Invalid format"),
            (b"x" * (25 * 1024 * 1024), "Too large (25MB)"),  # Over 20MB limit
        ]
        
        for test_data, description in invalid_cases:
            is_valid, message = audio_service.validate_audio_data(test_data)
            print(f"  {description}: {'❌ Correctly rejected' if not is_valid else '✅ Unexpectedly accepted'}")
            print(f"    Message: {message}")
        
        # Test 3: Test audio info with various formats
        print("\n3. Testing format detection...")
        
        # Create dummy headers for different formats
        format_tests = [
            (b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 1000, "WAV format"),
            (b"ID3" + b"\x00" * 1000, "MP3 format"),
            (b"\x00" * 4 + b"ftyp" + b"M4A " + b"\x00" * 1000, "M4A format"),
            (b"OggS" + b"\x00" * 1000, "OGG format"),
            (b"unknown_header" + b"\x00" * 1000, "Unknown format"),
        ]
        
        for test_data, description in format_tests:
            info = audio_service.get_audio_info(test_data)
            print(f"  {description}: detected as '{info['format']}', size: {info['size_mb']:.3f}MB")
        
        print("\n✅ Comprehensive audio service tests completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Comprehensive test failed: {e}")
        return False


def interactive_test():
    """Interactive testing mode."""
    print("\n🎙️ Audio Service Interactive Testing")
    print("=" * 50)
    print("This will test the audio service with real audio files.")
    print("")
    print("To get audio files for testing:")
    print("1. Use your existing file: Gachibowli.m4a")
    print("2. Record new audio with system tools")
    print("3. Download sample audio from internet")
    print("")
    
    while True:
        print("\nOptions:")
        print("1. Test with audio file")
        print("2. Run comprehensive tests (no file needed)")
        print("3. Quit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter path to audio file: ").strip()
            if file_path:
                test_audio_service_with_file(file_path)
        elif choice == "2":
            test_audio_service_comprehensive()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main testing function."""
    if len(sys.argv) > 1:
        # Test with provided file
        audio_file = sys.argv[1]
        success = test_audio_service_with_file(audio_file)
        
        # Also run comprehensive tests
        print("\n" + "=" * 60)
        comprehensive_success = test_audio_service_comprehensive()
        
        sys.exit(0 if (success and comprehensive_success) else 1)
    else:
        # Interactive mode
        interactive_test()


if __name__ == "__main__":
    main()