import os
import logging
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from language_utils import detect_language, get_system_prompt, get_error_messages

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SESSION_SECRET", "alexa-multilingual-skill-secret")

# Get API tokens from environment variables
HF_TOKEN = os.getenv("HF_TOKEN") # SECURITY: no hardcoding
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API endpoints
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def query_openai_api(messages):
    """Query OpenAI API for chat completions using messages array"""
    if not OPENAI_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
														 
											   
		  
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None

def query_huggingface_api(messages):
    """Query HuggingFace API directly with HTTP requests using chat messages array"""
    if not HF_TOKEN:
        return None
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
								  
	
    payload = {
        "inputs": messages
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "").strip()
                # If the model returns an echoed prompt, try to trim it out
                user_content = messages[-1]["content"]
                if generated_text.startswith(user_content):
                    generated_text = generated_text[len(user_content):].strip()
                return generated_text
        else:
            logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        logger.error(f"Error calling HuggingFace API: {e}")
        return None

def get_fallback_response(user_query, language):
    """Generate fallback responses when APIs are unavailable"""
    prompt_lower = user_query.lower()
    
    if language == "hi":
        responses = {
            "hello": "नमस्ते! मैं आपकी सहायता के लिए यहाँ हूँ।",
            "how are you": "मैं ठीक हूँ, धन्यवाद! आपका दिन कैसा है?",
            "weather": "मुझे मौसम की जानकारी नहीं है, लेकिन मैं अन्य सवालों में मदद कर सकता हूँ।",
            "time": "मुझे वर्तमान समय नहीं पता, लेकिन मैं अन्य प्रश्नों का उत्तर दे सकता हूँ।",
            "name": "मैं एक AI सहायक हूँ जो हिंदी और अंग्रेजी में बात कर सकता हूँ।",
            "help": "मैं सवालों के जवाब दे सकता हूँ और बातचीत कर सकता हूँ। कुछ और पूछिए!"
        }
        default = "यह दिलचस्प सवाल है। मैं इसके बारे मे��� और जानना चाहूँगा। क्या आप कुछ और बता सकते हैं?"
    else:
        responses = {
            "hello": "Hello! I'm here to help you.",
            "how are you": "I'm doing well, thank you! How is your day going?",
            "weather": "I don't have access to weather information, but I can help with other questions.",
            "time": "I don't know the current time, but I can answer other questions.",
            "name": "I'm an AI assistant that can communicate in both Hindi and English.",
            "help": "I can answer questions and have conversations. What would you like to know?"
        }
        default = "That's an interesting question. I'd like to learn more about that. Can you tell me more?"
    
    for key, response in responses.items():
        if key in prompt_lower:
            return response
	
    return default

def query_ai_api(messages, user_query, detected_language="en"):
    """Try OpenAI first, fallback to HuggingFace, then local fallback, using message history array"""
    # Try OpenAI first
    result = query_openai_api(messages)
    if result:
        return result
    
    # Fallback to HuggingFace
    result = query_huggingface_api(messages)
    if result:
        return result
    
    # Final fallback - always return something
    return get_fallback_response(user_query, detected_language)

def build_chat_messages_from_history(history, user_query, system_prompt=None):
    """
    Convert chat history and user's current Alexa question into chat-completions 'messages' format.
    Optionally prepend a system prompt (for models like OpenAI/Mistral).
	
																 
											  
																  
	
																					  
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add previous turns as user/assistant pairs
    for turn in history:
        if turn.get("user"):
            messages.append({"role": "user", "content": turn["user"]})
        if turn.get("assistant"):
            messages.append({"role": "assistant", "content": turn["assistant"]})

    # Add the latest user turn
    messages.append({"role": "user", "content": user_query})
    return messages

# In-memory history storage per Alexa session with language context
chat_history = {}

@app.route("/", methods=["GET"])
def index():
    """Home page to display application status"""
    return render_template("index.html")

@app.route("/alexa", methods=["POST"])
def alexa_webhook():
    """Enhanced Alexa webhook with Hindi and English language support, using chat-completions format"""
    try:
        payload = request.get_json()
        logger.debug(f"Received Alexa payload: {payload}")
        
        # Extract session information
        session_id = payload.get("session", {}).get("sessionId", "default")
        
        # Extract user query and locale information
        request_data = payload.get("request", {})
        intent_data = request_data.get("intent", {})
        slots = intent_data.get("slots", {})
        
        # Get user query from slots
        user_query = None
        if "query" in slots and slots["query"].get("value"):
            user_query = slots["query"]["value"]
        elif "message" in slots and slots["message"].get("value"):
            user_query = slots["message"]["value"]
        else:
            # Try to extract from other common slot names
            for slot_name, slot_data in slots.items():
                if slot_data.get("value"):
                    user_query = slot_data["value"]
                    break
        
        if not user_query:
            logger.error("No user query found in request")
            return create_alexa_response("I didn't understand your request. Please try again.", "en")
        
        # Detect language from locale or content
        request_locale = payload.get("request", {}).get("locale", "en-US")
        detected_language = detect_language(user_query, request_locale)
        
        logger.info(f"Session: {session_id}, Language: {detected_language}, Query: {user_query}")
        
        # Check if any AI API token is available
        if not HF_TOKEN and not OPENAI_API_KEY:
            error_messages = get_error_messages(detected_language)
            return create_alexa_response(error_messages["service_unavailable"], detected_language)
        
        # Get or create session history with language context
        if session_id not in chat_history:
            chat_history[session_id] = {
                "messages": [],
                "primary_language": detected_language
            }
        
        session_data = chat_history[session_id]
        
        # Update primary language if consistently using a different language
        if detected_language != session_data["primary_language"]:
            session_data["primary_language"] = detected_language
            logger.info(f"Updated primary language for session {session_id} to {detected_language}")
        
        # Build chat-completions messages from history (last 3 turns for model context)
        history = session_data["messages"]
						 
								 
														
																  
												  
															  
		
											 
        system_prompt = get_system_prompt(detected_language)
        messages = build_chat_messages_from_history(history[-3:], user_query, system_prompt)
        
        # Call AI API with fallback system using chat-format messages
        ai_reply = query_ai_api(messages, user_query, detected_language)
        
        if ai_reply is None:
            error_messages = get_error_messages(detected_language)
            return create_alexa_response(error_messages["service_unavailable"], detected_language)
        
        # Clean up the response (remove any "Assistant:" prefix if present)
        if ai_reply.startswith("Assistant:"):
            ai_reply = ai_reply[10:].strip()
        
        logger.info(f"AI Response ({detected_language}): {ai_reply}")
        
        # Update session history
        history.append({"user": user_query, "assistant": ai_reply, "language": detected_language})
        session_data["messages"] = history
        
        # Return Alexa-compatible response
        return create_alexa_response(ai_reply, detected_language)
        
    except KeyError as ke:
        logger.error(f"Missing key in request payload: {ke}")
        detected_language = detect_language("", request.headers.get("Accept-Language", "en-US"))
        error_messages = get_error_messages(detected_language)
        return create_alexa_response(error_messages["request_error"], detected_language)
        
    except Exception as e:
        logger.exception("Unexpected error in Alexa webhook")
        detected_language = detect_language("", request.headers.get("Accept-Language", "en-US"))
        error_messages = get_error_messages(detected_language)
        return create_alexa_response(error_messages["unexpected_error"], detected_language)

def create_alexa_response(text, language="en", end_session=True):
    """Create Alexa-compatible JSON response"""
    return jsonify({
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "PlainText",
                "text": text
            },
            "shouldEndSession": end_session
        },
        "sessionAttributes": {
            "language": language
        }
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "huggingface_api": "connected" if HF_TOKEN else "disconnected",
        "openai_api": "connected" if OPENAI_API_KEY else "disconnected",
        "ai_service": "available" if (HF_TOKEN or OPENAI_API_KEY) else "unavailable",
        "supported_languages": ["en", "hi"]
    }
    return jsonify(status)

@app.route("/sessions", methods=["GET"])
def get_sessions():
    """Debug endpoint to view active sessions"""
    session_summary = {}
    for session_id, data in chat_history.items():
        session_summary[session_id] = {
            "primary_language": data["primary_language"],
            "message_count": len(data["messages"]),
            "last_activity": data["messages"][-1] if data["messages"] else None
        }
    return jsonify(session_summary)

if __name__ == "__main__":
    logger.info("Starting Enhanced Alexa Skill with Hindi and English support")
    app.run(host="0.0.0.0", port=5000, debug=True)
