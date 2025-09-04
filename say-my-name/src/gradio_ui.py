"""
AI Assistant Gradio Frontend Interface

This module provides a Claude-like chat interface using Gradio that connects
to the FastAPI backend. Features conversation management, real-time chat,
voice input with TTS responses, and a professional user interface.

Key Features:
- Claude-like interface with sidebar and main chat area
- Multiple conversation thread management
- Real-time message processing with loading indicators
- Voice input with automatic transcription
- Text-to-speech responses (TTS enabled by default)
- Manual TTS controls with voice selection
- Conversation history loading and display
- Responsive design for desktop and mobile
- Error handling with user-friendly messages
- Auto-scrolling chat interface

Usage Example:
    # Start the interface
    python src/gradio_ui.py
    
    # Or import and launch programmatically
    from gradio_ui import create_chat_interface
    interface = create_chat_interface()
    interface.launch()


Class Structure:

APIClient (HTTP communication)
├── _make_request() - Generic HTTP request handler
├── create_conversation() - POST /conversations
├── send_message() - POST /chat/message
├── get_conversations() - GET /conversations
└── get_conversation_history() - GET /conversations/{id}/history

ConversationManager (UI state management)
├── refresh_conversations() - Update conversation list
├── switch_conversation() - Change active conversation
├── send_message() - Process user messages
└── create_new_conversation() - Create and switch to new chat

create_chat_interface() (UI layout)
├── Sidebar with conversation management
├── Main chat area with message history
├── Message input with send functionality  
└── Event handlers for all interactions
"""

import json
import requests
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import time

import gradio as gr

from config import get_config


class APIClient:
    """
    Client for communicating with the FastAPI backend.
    
    Handles all HTTP requests to the backend API with proper error handling
    and response processing. Provides a clean interface for UI components.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client with base URL.
        
        Args:
            base_url: Base URL of the FastAPI backend
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = 30  # 30 second timeout for requests
        
        print(f"🔗 API Client initialized for: {self.base_url}")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make HTTP request to backend API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            data: Request data for POST requests
            
        Returns:
            Dict[str, Any]: Response data
            
        Raises:
            Exception: If request fails or API returns error
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(
                    url,
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
            elif method == "DELETE":
                response = requests.delete(url, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise Exception("❌ Cannot connect to backend API. Make sure the FastAPI server is running.")
        except requests.exceptions.Timeout:
            raise Exception("⏱️ Request timeout. The server is taking too long to respond.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception("❌ Resource not found.")
            elif e.response.status_code == 500:
                raise Exception("❌ Server error. Please try again later.")
            else:
                raise Exception(f"❌ API error: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"❌ Unexpected error: {str(e)}")
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation thread.
        
        Args:
            title: Optional conversation title
            
        Returns:
            str: New conversation ID
        """
        data = {"title": title} if title else {}
        response = self._make_request("POST", "/conversations", data)
        return response["conversation_id"]
    
    def get_conversations(self) -> List[Dict[str, Any]]:
        """
        Get list of all conversations.
        
        Returns:
            List[Dict]: List of conversation metadata
        """
        response = self._make_request("GET", "/conversations")
        return response.get("conversations", [])
    
    def send_message(self, conversation_id: str, message: str, input_type: str = "text", 
                    enable_tts: bool = False) -> Dict[str, Any]:
        """
        Send message to AI assistant with optional TTS.
        
        Args:
            conversation_id: Target conversation ID
            message: User's message
            input_type: Type of input (text or voice)
            enable_tts: Whether to enable TTS response
            
        Returns:
            Dict[str, Any]: AI response data
        """
        data = {
            "conversation_id": conversation_id,
            "message": message,
            "input_type": input_type,
            "enable_tts": enable_tts
        }
        return self._make_request("POST", "/chat/message", data)
    
    def send_voice_message(self, conversation_id: str, audio_file_path: str, 
                          enable_tts: bool = True) -> Dict[str, Any]:
        """
        Send voice message to AI assistant.
        
        Args:
            conversation_id: Target conversation ID
            audio_file_path: Path to audio file
            enable_tts: Whether to enable TTS response
            
        Returns:
            Dict[str, Any]: AI response data
        """
        try:
            print(f"🔍 Sending voice message: {audio_file_path}")
            
            # Check if file exists
            import os
            if not os.path.exists(audio_file_path):
                raise Exception(f"Audio file not found: {audio_file_path}")
            
            with open(audio_file_path, 'rb') as f:
                files = {'audio_file': ('voice.mp3', f, 'audio/mp3')}
                data = {
                    'conversation_id': conversation_id,
                    'enable_tts': str(enable_tts).lower()
                }
                
                url = f"{self.base_url}/chat/voice"
                print(f"🌐 Making request to: {url}")
                
                response = requests.post(url, files=files, data=data, timeout=self.timeout)
                
                print(f"📡 Response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"❌ HTTP Error: {response.text}")
                
                response.raise_for_status()
                result = response.json()
                print(f"✅ API response received: {result.get('success', False)}")
                return result
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            raise Exception(f"❌ Network error: {str(e)}")
        except Exception as e:
            print(f"❌ Voice message error: {e}")
            raise Exception(f"❌ Voice message failed: {str(e)}")
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get message history for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List[Dict]: List of message exchanges
        """
        response = self._make_request("GET", f"/conversations/{conversation_id}/history")
        return response.get("messages", [])
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            self._make_request("DELETE", f"/conversations/{conversation_id}")
            return True
        except Exception:
            return False
    
    def health_check(self) -> bool:
        """
        Check if the backend API is healthy.
        
        Returns:
            bool: True if API is healthy
        """
        try:
            response = self._make_request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception:
            return False


class ConversationManager:
    """
    Manages conversation state and operations for the UI.
    
    Handles conversation switching, message history, and UI state management.
    Provides a clean interface between the UI components and the API client.
    """
    
    def __init__(self, api_client: APIClient):
        """
        Initialize conversation manager.
        
        Args:
            api_client: API client for backend communication
        """
        self.api = api_client
        self.current_conversation_id: Optional[str] = None
        self.conversations_cache: List[Dict[str, Any]] = []
        
    def refresh_conversations(self) -> List[Tuple[str, str]]:
        """
        Refresh conversation list from backend.
        
        Returns:
            List[Tuple[str, str]]: List of (title, conversation_id) pairs for dropdown
        """
        try:
            self.conversations_cache = self.api.get_conversations()
            
            # Format for Gradio dropdown: (display_name, value)
            if self.conversations_cache:
                return [(conv["title"], conv["conversation_id"]) for conv in self.conversations_cache]
            else:
                return [("No conversations yet", "")]
                
        except Exception as e:
            print(f"❌ Error refreshing conversations: {e}")
            return [("Error loading conversations", "")]
    
    def create_new_conversation(self, title: Optional[str] = None) -> Tuple[str, str]:
        """
        Create a new conversation and switch to it.
        
        Args:
            title: Optional conversation title
            
        Returns:
            Tuple[str, str]: (new_conversation_id, success_message)
        """
        try:
            conv_id = self.api.create_conversation(title)
            self.current_conversation_id = conv_id
            
            return conv_id, f"✅ Created new conversation: {conv_id[:8]}..."
            
        except Exception as e:
            return "", f"❌ Failed to create conversation: {str(e)}"
    
    def switch_conversation(self, conversation_id: str) -> List[List[str]]:
        """
        Switch to a different conversation and load its history.
        
        Args:
            conversation_id: Target conversation ID
            
        Returns:
            List[List[str]]: Chat history formatted for Gradio chatbot
        """
        if not conversation_id or conversation_id == "":
            return []
        
        try:
            self.current_conversation_id = conversation_id
            history = self.api.get_conversation_history(conversation_id)
            
            # Format history for Gradio chatbot: [[user_message, ai_response], ...]
            chat_history = []
            for msg in history:
                chat_history.append([msg["user_message"], msg["ai_response"]])
            
            print(f"🔄 Switched to conversation {conversation_id[:8]}... ({len(history)} messages)")
            return chat_history
            
        except Exception as e:
            print(f"❌ Error switching conversation: {e}")
            return [[f"❌ Error loading conversation: {str(e)}", ""]]
    
    def send_message(self, message: str, enable_tts: bool = False) -> Tuple[List[List[str]], str]:
        """
        Send message in current conversation.
        
        Args:
            message: User's message
            enable_tts: Whether to enable TTS response
            
        Returns:
            Tuple[List[List[str]], str]: (updated_chat_history, status_message)
        """
        if not self.current_conversation_id:
            return [], "❌ No conversation selected. Please create a new conversation first."

        if not message.strip():
            return [], "❌ Message cannot be empty."

        try:
            # Send message to API
            response = self.api.send_message(
                self.current_conversation_id, 
                message.strip(), 
                input_type="text",
                enable_tts=enable_tts
            )

            if response["success"]:
                # Get updated conversation history
                updated_history = self.switch_conversation(self.current_conversation_id)
                tts_status = " (with TTS)" if response.get("has_voice_response") else ""
                return updated_history, f"✅ Message sent successfully{tts_status}"
            else:
                error_msg = response.get("error_message", "Unknown error")
                return [], f"❌ Failed to send message: {error_msg}"
                
        except Exception as e:
            return [], f"❌ Error sending message: {str(e)}"
    
    def send_voice_message(self, audio_file_path: str, enable_tts: bool = True) -> Tuple[List[List[str]], str]:
        """
        Send voice message in current conversation.
        
        Args:
            audio_file_path: Path to audio file
            enable_tts: Whether to enable TTS response
            
        Returns:
            Tuple[List[List[str]], str]: (updated_chat_history, status_message)
        """
        if not self.current_conversation_id:
            return [], "❌ No conversation selected. Please create a new conversation first."

        if not audio_file_path:
            return [], "❌ Please select an audio file."

        print(f"🎤 Processing voice file: {audio_file_path}")
        
        try:
            # Send voice message to API
            response = self.api.send_voice_message(
                self.current_conversation_id,
                audio_file_path,
                enable_tts=enable_tts
            )
            
            print(f"📡 API response success: {response.get('success', False)}")

            if response["success"]:
                # Get updated conversation history
                updated_history = self.switch_conversation(self.current_conversation_id)
                tts_status = " (with TTS)" if response.get("has_voice_response") else ""
                return updated_history, f"✅ Voice message processed{tts_status}"
            else:
                error_msg = response.get("error_message", "Unknown error")
                print(f"❌ API error: {error_msg}")
                return [], f"❌ Failed to process voice message: {error_msg}"
                
        except Exception as e:
            print(f"❌ Exception in send_voice_message: {e}")
            return [], f"❌ Error processing voice message: {str(e)}"
def create_chat_interface() -> gr.Blocks:
    """
    Create the main chat interface using Gradio.
    
    Creates a Claude-like interface with conversation management sidebar
    and main chat area. Handles all user interactions and API communication.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Initialize components
    config = get_config()
    api_client = APIClient(f"http://{config.app.host}:{config.app.port}")
    conv_manager = ConversationManager(api_client)
    
    # Custom CSS and JavaScript for voice features
    custom_css = """
    /* Main container styling */
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto;
    }
    
    /* Sidebar styling */
    .sidebar {
        background-color: transparent;
        border-right: 1px solid #e9ecef;
        padding: 15px;
        min-height: 600px;
    }
    
    /* Chat area styling */
    .chat-container {
        background-color: transparent;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Voice input styling */
    .voice-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Button styling */
    .primary-button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    
    .voice-button {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    
    .secondary-button {
        background-color: #6c757d !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 6px 12px !important;
    }
    
    /* Status message styling */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Input sections */
    .input-section {
        margin-bottom: 15px;
        padding: 10px;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        background-color: #f8f9fa;
    }
    
    /* TTS Button styling */
    .tts-button {
        background-color: #17a2b8 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
        font-size: 12px !important;
        margin-left: 8px !important;
    }
    
    .tts-button:hover {
        background-color: #138496 !important;
    }
    """
    
    # JavaScript for TTS functionality
    tts_javascript = """
    <script>
    // Initialize speech synthesis voices
    let voicesLoaded = false;
    let availableVoices = [];
    
    function loadVoices() {
        availableVoices = speechSynthesis.getVoices();
        voicesLoaded = true;
        console.log('Available voices:', availableVoices.length);
        availableVoices.forEach((voice, index) => {
            console.log(`Voice ${index}: ${voice.name} (${voice.lang})`);
        });
    }
    
    // Load voices when available
    if ('speechSynthesis' in window) {
        speechSynthesis.onvoiceschanged = loadVoices;
        // Also try to load immediately
        setTimeout(loadVoices, 100);
    }
    
    // Enhanced TTS function
    function speakTextEnhanced(text, voiceType = 'female') {
        return new Promise((resolve, reject) => {
            if (!voicesLoaded) {
                loadVoices();
                setTimeout(() => speakTextEnhanced(text, voiceType), 500);
                return;
            }
            
            if (!('speechSynthesis' in window)) {
                reject('Speech synthesis not supported');
                return;
            }
            
            // Stop any current speech
            speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Select voice
            let selectedVoice = availableVoices[0]; // Default fallback
            
            if (voiceType === 'female') {
                selectedVoice = availableVoices.find(voice => 
                    voice.name.toLowerCase().includes('female') ||
                    voice.name.includes('Zira') ||
                    voice.name.includes('Samantha') ||
                    voice.name.includes('Karen') ||
                    voice.name.includes('Victoria') ||
                    voice.name.includes('Susan')
                ) || selectedVoice;
            } else if (voiceType === 'male') {
                selectedVoice = availableVoices.find(voice => 
                    voice.name.toLowerCase().includes('male') ||
                    voice.name.includes('David') ||
                    voice.name.includes('Daniel') ||
                    voice.name.includes('Alex') ||
                    voice.name.includes('Tom')
                ) || selectedVoice;
            }
            
            if (selectedVoice) {
                utterance.voice = selectedVoice;
                console.log('Using voice:', selectedVoice.name);
            }
            
            // Speech settings
            utterance.rate = 0.8;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            utterance.lang = 'en-US';
            
            // Event handlers
            utterance.onstart = () => {
                console.log('Speech started');
                resolve('Speaking...');
            };
            
            utterance.onend = () => {
                console.log('Speech ended');
            };
            
            utterance.onerror = (event) => {
                console.error('Speech error:', event.error);
                reject(event.error);
            };
            
            // Speak
            speechSynthesis.speak(utterance);
        });
    }
    
    // Test TTS on page load
    window.addEventListener('load', function() {
        setTimeout(() => {
            console.log('TTS Test: Voices available:', speechSynthesis.getVoices().length);
        }, 1000);
    });
    </script>
    """
    
    # Create the interface
    with gr.Blocks(
        title="AI Assistant with Voice",
        theme=gr.themes.Soft(),
        css=custom_css,
        head=tts_javascript
    ) as interface:
        
        # Header
        gr.Markdown("# 🤖 AI Assistant with Voice & TTS")
        gr.Markdown("*Your intelligent conversation partner with voice input, text-to-speech responses, and persistent memory*")
        
        # Check API health on startup
        with gr.Row():
            health_status = gr.Markdown("🔍 Checking API connection...")
        
        # Main interface layout
        with gr.Row():
            # Left sidebar - Conversation management
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.Markdown("### 💬 Conversations")
                
                # Conversation list dropdown
                conversation_dropdown = gr.Dropdown(
                    label="Select Conversation",
                    choices=[("No conversations yet", "")],
                    value="",
                    interactive=True
                )
                
                # New conversation section
                gr.Markdown("**Start New Chat:**")
                new_conv_title = gr.Textbox(
                    label="Title (optional)",
                    placeholder="e.g., Python Learning",
                    lines=1
                )
                
                new_conv_btn = gr.Button(
                    "📝 New Conversation",
                    elem_classes=["primary-button"],
                    variant="primary"
                )
                
                # Conversation management
                refresh_btn = gr.Button(
                    "🔄 Refresh List",
                    elem_classes=["secondary-button"]
                )
                
                delete_btn = gr.Button(
                    "🗑️ Delete Current",
                    elem_classes=["secondary-button"],
                    variant="stop"
                )
                
                # Status messages
                sidebar_status = gr.Markdown("", elem_classes=["status-success"])
            
            # Right main area - Chat interface
            with gr.Column(scale=3, elem_classes=["chat-container"]):
                # Chat history display
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=450,
                    show_label=False,
                    container=True,
                    bubble_full_width=False
                )
                
                # TTS Control Section
                with gr.Row():
                    with gr.Column(scale=3):
                        tts_text_input = gr.Textbox(
                            label="Text to Read Out",
                            placeholder="Enter text to be read aloud, or it will auto-fill with the last AI response...",
                            lines=2,
                            show_label=True
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            read_text_btn = gr.Button(
                                "🔊 Read Text",
                                elem_classes=["tts-button"],
                                variant="secondary"
                            )
                            stop_reading_btn = gr.Button(
                                "⏹️ Stop",
                                elem_classes=["secondary-button"],
                                variant="stop"
                            )
                        
                        voice_type_dropdown = gr.Dropdown(
                            label="Voice",
                            choices=[("Female", "female"), ("Male", "male"), ("Default", "default")],
                            value="female",
                            show_label=True
                        )
                
                # Voice input section
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("**🎤 Voice Input**")
                        audio_input = gr.Audio(
                            label="Record or Upload Audio",
                            sources=["microphone", "upload"],
                            type="filepath",
                            show_label=False
                        )
                        
                        with gr.Row():
                            voice_send_btn = gr.Button(
                                "🎤 Send Voice",
                                elem_classes=["voice-button"],
                                variant="primary",
                                interactive=True
                            )
                
                # Text message input area
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("**💬 Text Input**")
                        message_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here... (Press Enter to send)",
                            lines=1,
                            show_label=False
                        )
                    
                    with gr.Column(scale=1):
                        send_btn = gr.Button(
                            "📤 Send",
                            elem_classes=["primary-button"],
                            variant="primary"
                        )
                
                # Chat status
                chat_status = gr.Markdown("", elem_classes=["status-success"])
        
        # Event handlers
        
        def check_api_health():
            """Check if the backend API is running."""
            try:
                if api_client.health_check():
                    return "✅ **API Connected** - Ready to chat!"
                else:
                    return "⚠️ **API Warning** - Service may be degraded"
            except Exception as e:
                return f"❌ **API Error** - {str(e)}"
        
        def refresh_conversation_list():
            """Refresh the conversation dropdown list."""
            try:
                choices = conv_manager.refresh_conversations()
                return gr.Dropdown(choices=choices), "✅ Conversation list refreshed"
            except Exception as e:
                return gr.Dropdown(choices=[("Error", "")]), f"❌ Error: {str(e)}"
        
        def create_conversation(title):
            """Create new conversation and update UI."""
            try:
                conv_id, message = conv_manager.create_new_conversation(title)
                if conv_id:
                    # Refresh dropdown and select new conversation
                    choices = conv_manager.refresh_conversations()
                    return (
                        gr.Dropdown(choices=choices, value=conv_id),  # Update dropdown
                        [],  # Clear chat history
                        "",  # Clear title input
                        message,  # Status message
                        ""   # Clear chat status
                    )
                else:
                    return (
                        gr.Dropdown(),  # No change to dropdown
                        [],  # No change to chat
                        title,  # Keep title
                        message,  # Error message
                        ""   # Clear chat status
                    )
            except Exception as e:
                return (
                    gr.Dropdown(), [], title, 
                    f"❌ Error creating conversation: {str(e)}", ""
                )
        
        def switch_conversation(conversation_id):
            """Switch to selected conversation."""
            try:
                if conversation_id and conversation_id != "":
                    chat_history = conv_manager.switch_conversation(conversation_id)
                    return chat_history, f"✅ Switched to conversation {conversation_id[:8]}..."
                else:
                    return [], "ℹ️ Please select a conversation"
            except Exception as e:
                return [], f"❌ Error switching conversation: {str(e)}"
        
        def send_message(message, current_history):
            """Send user message and get AI response."""
            if not message.strip():
                return current_history, "", "❌ Please enter a message"

            try:
                # Send message to API with TTS enabled by default
                api_response_history, status = conv_manager.send_message(message, enable_tts=True)

                # Return final updated state with AI response
                return api_response_history, "", status

            except Exception as e:
                # Keep user message visible even if error occurs
                error_history = current_history + [[message, f"❌ Error: {str(e)}"]]
                error_message = f"❌ Error: {str(e)}"
                print(f"Message processing error: {e}")
                return error_history, "", error_message
        
        def send_voice_message(audio_path, current_history):
            """Send voice message and get AI response."""
            if not audio_path:
                return current_history, "❌ Please record or upload an audio file"

            try:
                # Show processing status immediately
                processing_history = current_history + [["[Processing voice message...]", None]]
                
                # Send voice message to API with TTS enabled by default
                api_response_history, status = conv_manager.send_voice_message(audio_path, enable_tts=True)

                # Return final updated state
                return api_response_history, status

            except Exception as e:
                error_message = f"❌ Voice error: {str(e)}"
                print(f"Voice processing error: {e}")
                return current_history, error_message
        
        def read_text_aloud(text, voice_type):
            """Trigger text-to-speech for given text."""
            if not text.strip():
                return "⚠️ No text to read"
            
            # Return JavaScript code that will be executed by Gradio
            js_code = f"""
            () => {{
                const text = `{text.replace('`', '\\`').replace('\\', '\\\\').replace('"', '\\"')}`;
                const voiceType = '{voice_type}';
                
                // Stop any current speech
                if (window.speechSynthesis.speaking) {{
                    window.speechSynthesis.cancel();
                }}
                
                if ('speechSynthesis' in window) {{
                    const utterance = new SpeechSynthesisUtterance(text);
                    
                    // Get available voices
                    const voices = window.speechSynthesis.getVoices();
                    
                    // Select voice based on preference
                    let selectedVoice = null;
                    if (voiceType === 'female') {{
                        selectedVoice = voices.find(voice => 
                            voice.name.toLowerCase().includes('female') || 
                            voice.name.includes('Zira') ||
                            voice.name.includes('Samantha') ||
                            voice.name.includes('Karen') ||
                            voice.name.includes('Victoria')
                        );
                    }} else if (voiceType === 'male') {{
                        selectedVoice = voices.find(voice => 
                            voice.name.toLowerCase().includes('male') || 
                            voice.name.includes('David') ||
                            voice.name.includes('Daniel') ||
                            voice.name.includes('Alex')
                        );
                    }}
                    
                    // Fallback to first available voice
                    if (!selectedVoice && voices.length > 0) {{
                        selectedVoice = voices[0];
                    }}
                    
                    if (selectedVoice) {{
                        utterance.voice = selectedVoice;
                    }}
                    
                    // Speech settings
                    utterance.rate = 0.9;
                    utterance.pitch = 1.0;
                    utterance.volume = 0.8;
                    
                    // Speak the text
                    window.speechSynthesis.speak(utterance);
                    
                    return "🔊 Speaking: " + text.substring(0, 50) + (text.length > 50 ? "..." : "");
                }} else {{
                    return "❌ Speech synthesis not supported in your browser";
                }}
            }}
            """
            
            return f"🔊 Reading: {text[:50]}..." if len(text) > 50 else f"🔊 Reading: {text}", js_code
        
        def stop_text_reading():
            """Stop current text-to-speech."""
            js_code = """
            () => {
                if (window.speechSynthesis.speaking) {
                    window.speechSynthesis.cancel();
                    return "⏹️ Stopped reading";
                } else {
                    return "ℹ️ No speech to stop";
                }
            }
            """
            return "⏹️ Stopping...", js_code
        
        def auto_fill_last_response(chat_history):
            """Auto-fill the TTS input with the last AI response."""
            if chat_history and len(chat_history) > 0:
                last_exchange = chat_history[-1]
                if len(last_exchange) > 1 and last_exchange[1]:  # Check if AI response exists
                    return last_exchange[1]  # Return AI response
            return ""
        
        def delete_current_conversation(current_conv_id):
            """Delete the currently selected conversation."""
            try:
                if not current_conv_id:
                    return (
                        gr.Dropdown(),  # No change
                        [],  # Clear chat
                        "⚠️ No conversation selected to delete"
                    )
                
                success = api_client.delete_conversation(current_conv_id)
                if success:
                    # Refresh conversation list and clear selection
                    choices = conv_manager.refresh_conversations()
                    conv_manager.current_conversation_id = None
                    
                    return (
                        gr.Dropdown(choices=choices, value=""),
                        [],  # Clear chat history
                        f"✅ Conversation {current_conv_id[:8]}... deleted"
                    )
                else:
                    return (
                        gr.Dropdown(),  # No change
                        gr.Chatbot(),   # No change
                        "❌ Failed to delete conversation"
                    )
                    
            except Exception as e:
                return (
                    gr.Dropdown(), gr.Chatbot(), 
                    f"❌ Error deleting conversation: {str(e)}"
                )
        
        # Bind event handlers
        
        # Initialize API health check
        interface.load(
            check_api_health,
            outputs=[health_status]
        )
        
        # Initialize conversation list
        interface.load(
            refresh_conversation_list,
            outputs=[conversation_dropdown, sidebar_status]
        )
        
        # New conversation button
        new_conv_btn.click(
            create_conversation,
            inputs=[new_conv_title],
            outputs=[conversation_dropdown, chatbot, new_conv_title, sidebar_status, chat_status]
        )
        
        # Refresh conversations button
        refresh_btn.click(
            refresh_conversation_list,
            outputs=[conversation_dropdown, sidebar_status]
        )
        
        # Conversation dropdown change
        conversation_dropdown.change(
            switch_conversation,
            inputs=[conversation_dropdown],
            outputs=[chatbot, chat_status]
        )
        
        # Send message button and Enter key
        send_btn.click(
            send_message,
            inputs=[message_input, chatbot],
            outputs=[chatbot, message_input, chat_status]
        )

        message_input.submit(
            send_message,
            inputs=[message_input, chatbot],
            outputs=[chatbot, message_input, chat_status]
        )
        
        # Voice message button
        voice_send_btn.click(
            send_voice_message,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, chat_status]
        )
        
        # TTS Controls with JavaScript
        read_text_btn.click(
            fn=None,
            js="""
            (text, voice_type) => {
                if (!text.trim()) {
                    return "⚠️ No text to read";
                }
                
                // Stop any current speech
                if (window.speechSynthesis.speaking) {
                    window.speechSynthesis.cancel();
                }
                
                if ('speechSynthesis' in window) {
                    const utterance = new SpeechSynthesisUtterance(text);
                    
                    // Get available voices
                    const voices = window.speechSynthesis.getVoices();
                    
                    // Select voice based on preference
                    let selectedVoice = null;
                    if (voice_type === 'female') {
                        selectedVoice = voices.find(voice => 
                            voice.name.toLowerCase().includes('female') || 
                            voice.name.includes('Zira') ||
                            voice.name.includes('Samantha') ||
                            voice.name.includes('Karen') ||
                            voice.name.includes('Victoria')
                        );
                    } else if (voice_type === 'male') {
                        selectedVoice = voices.find(voice => 
                            voice.name.toLowerCase().includes('male') || 
                            voice.name.includes('David') ||
                            voice.name.includes('Daniel') ||
                            voice.name.includes('Alex')
                        );
                    }
                    
                    // Fallback to first available voice
                    if (!selectedVoice && voices.length > 0) {
                        selectedVoice = voices[0];
                    }
                    
                    if (selectedVoice) {
                        utterance.voice = selectedVoice;
                    }
                    
                    // Speech settings
                    utterance.rate = 0.9;
                    utterance.pitch = 1.0;
                    utterance.volume = 0.8;
                    
                    // Event handlers
                    utterance.onstart = function() {
                        console.log('Started speaking');
                    };
                    
                    utterance.onend = function() {
                        console.log('Finished speaking');
                    };
                    
                    utterance.onerror = function(event) {
                        console.error('Speech synthesis error:', event.error);
                    };
                    
                    // Speak the text
                    window.speechSynthesis.speak(utterance);
                    
                    return "🔊 Speaking: " + text.substring(0, 50) + (text.length > 50 ? "..." : "");
                } else {
                    return "❌ Speech synthesis not supported in your browser";
                }
            }
            """,
            inputs=[tts_text_input, voice_type_dropdown],
            outputs=[chat_status]
        )
        
        stop_reading_btn.click(
            fn=None,
            js="""
            () => {
                if (window.speechSynthesis.speaking) {
                    window.speechSynthesis.cancel();
                    return "⏹️ Stopped reading";
                } else {
                    return "ℹ️ No speech to stop";
                }
            }
            """,
            outputs=[chat_status]
        )
        
        # Auto-fill TTS input when chat updates
        chatbot.change(
            auto_fill_last_response,
            inputs=[chatbot],
            outputs=[tts_text_input]
        )
        
        # Delete conversation button
        delete_btn.click(
            delete_current_conversation,
            inputs=[conversation_dropdown],
            outputs=[conversation_dropdown, chatbot, sidebar_status]
        )
    
    return interface


def launch_interface():
    """
    Launch the Gradio interface with configuration.
    
    Starts the Gradio server with settings from configuration.
    This is the main entry point for running the UI.
    """
    config = get_config()
    
    print(f"🚀 Starting AI Assistant UI...")
    print(f"🌐 Frontend: http://localhost:7860")
    print(f"🔗 Backend API: http://{config.app.host}:{config.app.port}")
    print(f"📚 API Docs: http://{config.app.host}:{config.app.port}/docs")
    
    # Create and launch interface
    interface = create_chat_interface()
    
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard Gradio port
        share=True,           # Set to True to create public link
        debug=config.app.debug,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    """
    Launch the Gradio UI when run directly.
    
    Usage:
        python src/gradio_ui.py
        
    Make sure the FastAPI backend is running first:
        python src/main.py
    """
    
    print("🤖 AI Assistant - Gradio Frontend")
    print("=" * 50)
    print("⚠️  Make sure FastAPI backend is running first!")
    print("   Run: python src/main.py")
    print("=" * 50)
    
    launch_interface()