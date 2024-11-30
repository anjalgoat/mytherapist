import streamlit as st
import sys
import os
from datetime import datetime
import uuid
import logging
from dotenv import load_dotenv

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now the imports should work
from app.agents.coordinator import CoordinatorAgent
from app.models.message import Message

def load_config():
    """Load configuration from either Streamlit secrets or .env file"""
    config = {}
    
    # First try loading from .env file for local development
    load_dotenv()
    config["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config["MODEL_NAME"] = os.getenv("MODEL_NAME", "mixtral-8x7b-32768")
    config["MAX_HISTORY"] = int(os.getenv("MAX_HISTORY", "10"))
    config["CRISIS_THRESHOLD"] = float(os.getenv("CRISIS_THRESHOLD", "0.7"))
    
    # If we're in Streamlit Cloud, override with secrets
    try:
        if hasattr(st, 'secrets'):
            config["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", config["GROQ_API_KEY"])
            config["MODEL_NAME"] = st.secrets.get("MODEL_NAME", config["MODEL_NAME"])
            config["MAX_HISTORY"] = int(st.secrets.get("MAX_HISTORY", config["MAX_HISTORY"]))
            config["CRISIS_THRESHOLD"] = float(st.secrets.get("CRISIS_THRESHOLD", config["CRISIS_THRESHOLD"]))
    except Exception as e:
        logger.warning(f"Could not load Streamlit secrets: {e}")
    
    if not config["GROQ_API_KEY"]:
        raise ValueError("GROQ_API_KEY must be set in either .env file or Streamlit secrets")
        
    return config

def init_session_state():
    """Initialize session state with better error handling"""
    if 'initialized' not in st.session_state:
        try:
            # Load configuration
            st.session_state.config = load_config()
            logger.info("Configuration loaded successfully")
            
            # Initialize messages list
            st.session_state.messages = []
            
            # Initialize coordinator
            logger.info("Initializing coordinator...")
            st.session_state.coordinator = CoordinatorAgent()
            logger.info("Coordinator initialized successfully")
            
            # Initialize conversation state
            st.session_state.conversation_state = None
            
            # Mark as initialized
            st.session_state.initialized = True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            st.error("Failed to initialize the application:")
            st.error(str(e))
            return False
    return True

def main():
    st.title("Therapeutic Chatbot")
    st.write("Share your thoughts and feelings in a safe, supportive space.")
    
    # Initialize session state
    if not init_session_state():
        st.stop()
    
    # Debug info in sidebar
    with st.sidebar:
        if st.checkbox("Show Debug Info"):
            st.write("Configuration Status:")
            safe_config = {k: '***' if 'KEY' in k else v 
                         for k, v in st.session_state.config.items()}
            st.json(safe_config)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message.sender):
            st.markdown(message.content)
            if message.sender == "assistant":
                if message.metadata and message.metadata.get('therapeutic_intent'):
                    st.caption(f"Therapeutic approach: {message.metadata['therapeutic_intent']}")
                if message.metadata and message.metadata.get('crisis'):
                    st.error("⚠️ Crisis Support Resources ⚠️")
                    st.info("Emergency: 911 | Crisis Hotline: 988")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        user_message = Message(
            id=str(uuid.uuid4()),
            content=prompt,
            sender="user",
            timestamp=datetime.now().timestamp(),
            metadata={}
        )
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    import asyncio
                    response_message, new_state = asyncio.run(
                        st.session_state.coordinator.process_message(
                            user_message,
                            st.session_state.conversation_state
                        )
                    )
                    
                    # Update state and messages
                    st.session_state.conversation_state = new_state
                    st.session_state.messages.append(response_message)
                    
                    # Display response
                    st.markdown(response_message.content)
                    
                    # Display metadata if available
                    if response_message.metadata:
                        if response_message.metadata.get('therapeutic_intent'):
                            st.caption(f"Therapeutic approach: {response_message.metadata['therapeutic_intent']}")
                        if response_message.metadata.get('crisis'):
                            st.error("⚠️ Crisis Support Resources ⚠️")
                            st.info("Emergency: 911 | Crisis Hotline: 988")
                            
                except Exception as e:
                    logger.error(f"Error generating response: {e}", exc_info=True)
                    st.error("Failed to generate response. Please try again.")
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()