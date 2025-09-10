import speech_recognition as sr
import pyttsx3
import re
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import threading
import queue

class NLPInterface:
    """
    Natural Language Processing interface for voice commands and chatbot
    """
    
    def __init__(self, config_file='ai_config.json'):
        self.config = self._load_config(config_file)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self._setup_tts()
        
        # Command processing
        self.command_queue = queue.Queue()
        self.is_listening = False
        
        # Intent patterns
        self.intent_patterns = self._load_intent_patterns()
        
        # Response templates
        self.response_templates = self._load_response_templates()
        
    def _load_config(self, config_file: str) -> dict:
        """Load NLP configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'voice': {
                    'enabled': True,
                    'language': 'en-US',
                    'voice_rate': 200,
                    'voice_volume': 0.8
                },
                'commands': {
                    'wake_words': ['hey lab', 'smart lab', 'lab assistant'],
                    'timeout': 5
                }
            }
    
    def _setup_tts(self):
        """Setup text-to-speech engine"""
        voices = self.tts_engine.getProperty('voices')
        if voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        
        self.tts_engine.setProperty('rate', self.config['voice']['voice_rate'])
        self.tts_engine.setProperty('volume', self.config['voice']['voice_volume'])
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent recognition patterns"""
        return {
            'status_check': [
                r'what.*status',
                r'how.*lab',
                r'is.*occupied',
                r'anyone.*there',
                r'current.*occupancy'
            ],
            'start_monitoring': [
                r'start.*monitoring',
                r'begin.*detection',
                r'turn.*on.*camera',
                r'activate.*system'
            ],
            'stop_monitoring': [
                r'stop.*monitoring',
                r'end.*detection',
                r'turn.*off.*camera',
                r'deactivate.*system'
            ],
            'energy_info': [
                r'energy.*usage',
                r'power.*consumption',
                r'how.*much.*energy',
                r'energy.*report'
            ],
            'schedule_info': [
                r'when.*busy',
                r'busy.*times',
                r'occupancy.*prediction',
                r'next.*class'
            ],
            'settings_change': [
                r'change.*settings',
                r'adjust.*detection',
                r'modify.*zone',
                r'update.*config'
            ],
            'help': [
                r'help',
                r'what.*can.*do',
                r'commands',
                r'how.*to.*use'
            ]
        }
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different intents"""
        return {
            'status_check': [
                "The lab is currently {status} with {count} people detected.",
                "Current occupancy: {count} people. Status: {status}.",
                "The lab has {count} people. Detection is {status}."
            ],
            'start_monitoring': [
                "Starting monitoring system. Camera activated.",
                "Detection system is now active.",
                "Monitoring started successfully."
            ],
            'stop_monitoring': [
                "Stopping monitoring system. Camera deactivated.",
                "Detection system is now inactive.",
                "Monitoring stopped successfully."
            ],
            'energy_info': [
                "Current energy consumption is {energy} watts.",
                "Energy usage: {energy} watts. Active zones: {zones}.",
                "The lab is consuming {energy} watts of power."
            ],
            'schedule_info': [
                "Peak hours are typically {peak_hours}.",
                "The lab is usually busiest during {peak_hours}.",
                "Expected occupancy: {prediction} people in the next hour."
            ],
            'settings_change': [
                "Settings updated successfully.",
                "Configuration changed as requested.",
                "Settings have been modified."
            ],
            'help': [
                "I can help you with lab monitoring, energy usage, and system control.",
                "Available commands: status check, start/stop monitoring, energy info, and more.",
                "Just ask me about the lab status or system controls."
            ],
            'error': [
                "I didn't understand that command. Please try again.",
                "Could you please rephrase that?",
                "I'm not sure what you mean. Can you be more specific?"
            ]
        }
    
    def start_voice_listening(self):
        """Start continuous voice listening"""
        self.is_listening = True
        threading.Thread(target=self._listen_continuously, daemon=True).start()
        print("🎤 Voice interface activated. Say 'Hey Lab' to start commands.")
    
    def stop_voice_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        print("🔇 Voice interface deactivated.")
    
    def _listen_continuously(self):
        """Continuous voice listening loop"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        while self.is_listening:
            try:
                with self.microphone as source:
                    print("🎤 Listening...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                # Recognize speech
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"🗣️ Heard: {text}")
                    
                    # Check for wake words
                    if self._contains_wake_word(text):
                        print("👂 Wake word detected!")
                        self._process_voice_command(text)
                        
                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    print(f"❌ Speech recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                pass  # Timeout, continue listening
    
    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains wake words"""
        text_lower = text.lower()
        wake_words = self.config['commands']['wake_words']
        
        for wake_word in wake_words:
            if wake_word in text_lower:
                return True
        return False
    
    def _process_voice_command(self, text: str) -> Optional[str]:
        """Process voice command and return response"""
        # Remove wake words
        text_clean = text.lower()
        for wake_word in self.config['commands']['wake_words']:
            text_clean = text_clean.replace(wake_word, '').strip()
        
        # Recognize intent
        intent = self._recognize_intent(text_clean)
        
        # Generate response
        response = self._generate_response(intent, text_clean)
        
        # Speak response
        if response:
            self.speak(response)
            return response
        
        return None
    
    def _recognize_intent(self, text: str) -> str:
        """Recognize intent from text"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        
        return 'error'
    
    def _generate_response(self, intent: str, text: str) -> str:
        """Generate response based on intent"""
        templates = self.response_templates.get(intent, self.response_templates['error'])
        
        # Select random template
        import random
        template = random.choice(templates)
        
        # Fill in placeholders
        if intent == 'status_check':
            # This would be filled with actual data from the system
            return template.format(status="occupied", count="2")
        elif intent == 'energy_info':
            return template.format(energy="75", zones="1")
        elif intent == 'schedule_info':
            return template.format(peak_hours="9 AM to 5 PM", prediction="3")
        else:
            return template
    
    def speak(self, text: str):
        """Convert text to speech"""
        if self.config['voice']['enabled']:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
    
    def process_text_command(self, text: str) -> Dict[str, Any]:
        """Process text command and return structured response"""
        intent = self._recognize_intent(text.lower())
        
        response = {
            'intent': intent,
            'text': text,
            'response': self._generate_response(intent, text),
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.8
        }
        
        return response
    
    def get_available_commands(self) -> List[Dict[str, str]]:
        """Get list of available voice commands"""
        commands = [
            {
                'command': 'Hey Lab, what\'s the status?',
                'description': 'Check current lab occupancy and status',
                'intent': 'status_check'
            },
            {
                'command': 'Hey Lab, start monitoring',
                'description': 'Start the detection system',
                'intent': 'start_monitoring'
            },
            {
                'command': 'Hey Lab, stop monitoring',
                'description': 'Stop the detection system',
                'intent': 'stop_monitoring'
            },
            {
                'command': 'Hey Lab, energy usage',
                'description': 'Get current energy consumption info',
                'intent': 'energy_info'
            },
            {
                'command': 'Hey Lab, when is it busy?',
                'description': 'Get occupancy predictions',
                'intent': 'schedule_info'
            },
            {
                'command': 'Hey Lab, help',
                'description': 'Get help and available commands',
                'intent': 'help'
            }
        ]
        
        return commands
    
    def create_chatbot_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Create chatbot response based on user input and context"""
        intent = self._recognize_intent(user_input.lower())
        
        # Enhanced response generation with context
        if intent == 'status_check':
            occupancy = context.get('occupancy_count', 0)
            status = "occupied" if occupancy > 0 else "empty"
            return f"The lab is currently {status} with {occupancy} people detected."
        
        elif intent == 'energy_info':
            energy = context.get('energy_consumption', 0)
            zones = context.get('active_zones', 0)
            return f"Current energy consumption is {energy} watts across {zones} active zones."
        
        elif intent == 'schedule_info':
            predictions = context.get('occupancy_predictions', [])
            if predictions:
                next_hour = predictions[0] if predictions else {'predicted_occupancy': 0}
                return f"Expected occupancy in the next hour: {next_hour['predicted_occupancy']} people."
            return "Peak hours are typically 9 AM to 5 PM."
        
        else:
            return self._generate_response(intent, user_input)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = 0.7
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = -0.7
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {
            'time': [],
            'numbers': [],
            'locations': [],
            'actions': []
        }
        
        # Extract time patterns
        time_patterns = [
            r'\d{1,2}:\d{2}',  # HH:MM
            r'\d{1,2}\s*(am|pm)',  # 3pm, 10am
            r'(morning|afternoon|evening|night)',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text.lower())
            entities['time'].extend(matches)
        
        # Extract numbers
        number_pattern = r'\d+'
        numbers = re.findall(number_pattern, text)
        entities['numbers'].extend(numbers)
        
        # Extract locations
        location_words = ['lab', 'room', 'zone', 'area', 'section']
        for word in location_words:
            if word in text.lower():
                entities['locations'].append(word)
        
        # Extract actions
        action_words = ['start', 'stop', 'monitor', 'detect', 'check', 'show']
        for word in action_words:
            if word in text.lower():
                entities['actions'].append(word)
        
        return entities
    
    def cleanup(self):
        """Cleanup NLP resources"""
        self.stop_voice_listening()
        if hasattr(self, 'tts_engine'):
            self.tts_engine.stop()
