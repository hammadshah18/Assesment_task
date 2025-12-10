import streamlit as st
import json
import os
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


parser = JsonOutputParser()

prompt_template = """
You are a customer support AI. For the given customer message, return a STRICT JSON object with exactly these three fields:

1. category: one of the following:
["Complaint", "Refund/Return", "Sales Inquiry", "Delivery Question", "Account/Technical Issue", "General Query", "Spam"]

2. sentiment: one of:
["Positive", "Neutral", "Negative"]

3. reply: a short, professional, 1-2 sentence customer service reply appropriate for the message.

Return ONLY valid JSON. No explanation, no text outside the JSON.

Customer message: {msg}
"""

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(page_title="AI Customer Support", page_icon="", layout="wide")

# Sidebar
with st.sidebar:
    st.title("AI Customer Support")
    st.markdown("---")
    st.markdown("### About")
    st.info("This AI assistant analyzes customer messages and provides:\n\n 1: Category Classification\n\n 2: Sentiment Analysis\n\n 3: Auto-generated Reply")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


st.title("               Customer Message Analyzer")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


message = st.chat_input("Enter customer message here...")

if message:
    
    st.session_state.messages.append({"role": "user", "content": message})
    
    
    with st.chat_message("user"):
        st.markdown(message)
    
    
    with st.spinner("Analyzing message..."):
        
        final_prompt = prompt_template.format(msg=message)

        
        output = model.invoke([
            SystemMessage(content="You strictly respond in JSON."),
            HumanMessage(content=final_prompt),
        ])

        raw_text = output.content.strip()
        
        
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.replace("```", "").strip()

        try:
            result = json.loads(raw_text)
        except Exception as e:
            st.error("Model returned invalid JSON. Try again.")
            st.write("Raw response:", raw_text)
            st.stop()
    

    with st.chat_message("assistant"):
        
        response_text = f"""**📋 Category:** {result['category']}

**😊 Sentiment:** {result['sentiment']}

**✉️ Suggested Reply:**

{result['reply']}"""
        
      
        message_placeholder = st.empty()
        full_response = ""
        
        for char in response_text:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        
        message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
