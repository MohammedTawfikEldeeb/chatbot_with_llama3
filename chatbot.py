import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
MAX_HISTORY = int(os.getenv("MAX_HISTORY_LENGTH", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

# Initialize Groq client
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

# Custom CSS for clean UI
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 20px;}
    .user-message {background: #f0f8ff; border-radius: 15px; padding: 10px;}
    .assistant-message {background: #f5f5f5; border-radius: 15px; padding: 10px;}
    .typing-cursor {display: inline-block; width: 8px; height: 16px; background: #333; animation: blink 1s infinite;}
    @keyframes blink {50% {opacity: 0;}}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown(f'<div style="background: #4CAF50; color: white; padding: 4px 10px; border-radius: 12px; display: inline-block;">Llama 3 70B</div>', unsafe_allow_html=True)
    
    TEMPERATURE = st.slider("Creativity", 0.0, 1.0, TEMPERATURE, 0.1)
    MAX_HISTORY = st.slider("Context Memory", 1, 10, MAX_HISTORY)
    
    if st.button("🧹 Clear History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("💬 Llama 3 Chat")

# Display all historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Message Llama..."):
    # Store pending message (not yet in full history)
    st.session_state.pending_message = {"role": "user", "content": prompt}
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Include pending message in context
        context_messages = st.session_state.messages[-MAX_HISTORY:] + [st.session_state.pending_message]
        
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=context_messages,
            temperature=TEMPERATURE,
            stream=True
        )
        
        for chunk in stream:
            text_chunk = chunk.choices[0].delta.content or ""
            full_response += text_chunk
            response_placeholder.markdown(full_response + '<span class="typing-cursor"></span>', unsafe_allow_html=True)
        
        response_placeholder.markdown(full_response)
    
    # Only add to full history AFTER complete response
    st.session_state.messages.append(st.session_state.pending_message)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.pending_message = None
