import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def detect_language(text: str, locale: str = "en-US") -> str:
    """
    Detect language from text content and locale information
    
    Args:
        text: User input text
        locale: Alexa request locale (e.g., 'en-US', 'hi-IN')
    
    Returns:
        Language code ('en' or 'hi')
    """
    try:
        # First check locale
        if locale:
            if locale.startswith("hi"):
                return "hi"
            elif locale.startswith("en"):
                return "en"
        
        # Check for Hindi characters (Devanagari script)
        if text and re.search(r'[\u0900-\u097F]', text):
            logger.debug(f"Hindi script detected in text: {text}")
            return "hi"
        
        # Check for common Hindi transliterated words
        hindi_transliterated_words = [
            'namaste', 'kaise', 'kya', 'hai', 'haan', 'nahi', 'aap', 'main', 
            'mera', 'tera', 'uska', 'yahan', 'wahan', 'kahan', 'kab', 'kyun',
            'achha', 'theek', 'dhanyawad', 'shukriya', 'alvida'
        ]
        
        if text:
            text_lower = text.lower()
            for word in hindi_transliterated_words:
                if word in text_lower:
                    logger.debug(f"Hindi transliterated word '{word}' detected")
                    return "hi"
        
        # Default to English
        return "en"
        
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        return "en"  # Default fallback

def get_system_prompt(language: str) -> str:
    """
    Get language-specific system prompt for the AI assistant
    
    Args:
        language: Language code ('en' or 'hi')
    
    Returns:
        System prompt string
    """
    prompts = {
        "en": """You are a helpful and friendly AI assistant for Alexa. 
Respond naturally and conversationally in English. Keep responses concise and suitable for voice interaction. 
Be helpful, informative, and engaging. If asked about your capabilities, mention that you can help with 
questions, provide information, and have conversations in both English and Hindi.""",
        
        "hi": """आप Alexa के लिए एक सहायक और मित्रवत AI सहायक हैं। 
हिंदी में प्राकृतिक और बातचीत के अंदाज में जवाब दें। आवाज़ की बातचीत के लिए उपयुक्त संक्षिप्त उत्तर दें। 
सहायक, जानकारीपूर्ण और आकर्षक बनें। यदि आपकी क्षमताओं के बारे में पूछा जाए, तो बताएं कि आप 
सवालों का जवाब दे सकते हैं, जानकारी प्रदान कर सकते हैं, और हिंदी और अंग्रेजी दोनों में बातचीत कर सकते हैं।"""
    }
    
    return prompts.get(language, prompts["en"])

def get_error_messages(language: str) -> Dict[str, str]:
    """
    Get localized error messages for different error scenarios
    
    Args:
        language: Language code ('en' or 'hi')
    
    Returns:
        Dictionary of error messages
    """
    messages = {
        "en": {
            "service_unavailable": "The AI service is currently unavailable. Please try again later.",
            "request_error": "Sorry, I couldn't understand your request. Please try again.",
            "unexpected_error": "An unexpected error occurred. Please try again.",
            "no_query": "I didn't hear a question. What would you like to know?",
            "processing_error": "I'm having trouble processing your request right now. Please try again."
        },
        "hi": {
            "service_unavailable": "AI सेवा अभी उपलब्ध नहीं है। कृपया बाद में पुनः प्रयास करें।",
            "request_error": "माफ़ करें, मैं आपके अनुरोध को समझ नहीं सका। कृपया पुनः प्रयास करें।",
            "unexpected_error": "एक अप्रत्याशित त्रुटि हुई है। कृपया पुनः प्रयास करें।",
            "no_query": "मैंने कोई सवाल नहीं सुना। आप क्या जानना चाहते हैं?",
            "processing_error": "मुझे अभी आपके अनुरोध को संसाधित करने में परेशानी हो रही है। कृपया पुनः प्रयास करें।"
        }
    }
    
    return messages.get(language, messages["en"])

def format_response_for_voice(text: str, language: str) -> str:
    """
    Format response text to be more suitable for voice output
    
    Args:
        text: Response text
        language: Language code
    
    Returns:
        Formatted text suitable for voice synthesis
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Limit length for voice interaction (Alexa has limits)
    max_length = 300 if language == "en" else 250  # Hindi might need slightly less
    
    if len(text) > max_length:
        # Try to cut at sentence boundary
        sentences = re.split(r'[।.!?]', text)
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence) > max_length:
                break
            truncated += sentence + ("।" if language == "hi" else ".")
        
        if truncated:
            text = truncated
        else:
            # Hard truncate if no sentence boundary found
            text = text[:max_length] + ("..." if language == "en" else "...")
    
    return text

def get_language_name(language_code: str) -> str:
    """
    Get human-readable language name from code
    
    Args:
        language_code: Language code ('en' or 'hi')
    
    Returns:
        Human-readable language name
    """
    names = {
        "en": "English",
        "hi": "Hindi / हिंदी"
    }
    return names.get(language_code, "Unknown")
