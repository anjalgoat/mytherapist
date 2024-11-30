import streamlit as st
import sys
import os
from datetime import datetime
import uuid

# Step 3: Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import directly from local directories
from agents.coordinator import CoordinatorAgent
from models.message import Message

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = CoordinatorAgent()
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = None

def main():
    st.title("Therapeutic Chatbot")
    st.write("Share your thoughts and feelings in a safe, supportive space.")
    
    init_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        is_user = message.sender == "user"
        with st.chat_message("user" if is_user else "assistant"):
            st.write(message.content)
            if not is_user and message.metadata and message.metadata.get('therapeuticIntent'):
                st.caption(f"Therapeutic approach: {message.metadata['therapeuticIntent']}")
                
            if not is_user and message.metadata and message.metadata.get('crisis'):
                st.error("⚠️ Crisis Support Resources ⚠️")
                st.info("Emergency: 911 | Crisis Hotline: 988")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Create a unique ID for the message
        msg_id = str(uuid.uuid4())
        
        user_message = Message(
            id=msg_id,
            content=prompt,
            sender="user",
            timestamp=datetime.now().timestamp(),
            metadata={}
        )
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.empty():
                st.write("Thinking...")
                try:
                    import asyncio
                    # The process_message returns a tuple of (response, state)
                    response_message, new_state = asyncio.run(
                        st.session_state.coordinator.process_message(
                            user_message,
                            st.session_state.conversation_state
                        )
                    )
                    
                    st.session_state.conversation_state = new_state
                    st.session_state.messages.append(response_message)
                    
                    st.write(response_message.content)
                    
                    if response_message.metadata and response_message.metadata.get('therapeuticIntent'):
                        st.caption(f"Therapeutic approach: {response_message.metadata['therapeuticIntent']}")
                    
                    if response_message.metadata and response_message.metadata.get('crisis'):
                        st.error("⚠️ Crisis Support Resources ⚠️")
                        st.info("Emergency: 911 | Crisis Hotline: 988")
                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")

if __name__ == "__main__":
    main()