import streamlit as st
import os
from PIL import Image

# Import the compiled LangGraph orchestrator from your previous script
# Make sure dynamic_agent.py is in the same folder!
from dynamic_agent import app 

# Configure the Streamlit page
st.set_page_config(page_title="Radiology AI Orchestrator", layout="wide", page_icon="🩺")
st.title("🩺 Multi-Agent Radiology Assistant")

# --- Sidebar: Clinical Image Upload ---
with st.sidebar:
    st.header("Clinical Inputs")
    st.markdown("Upload a patient's Chest X-ray to allow the Vision Agent to extract findings.")
    
    uploaded_file = st.file_uploader("Upload X-Ray (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    
    image_path = ""
    if uploaded_file is not None:
        # Save the byte stream to a temporary local file for the Vision Tool
        image_path = "temp_clinical_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.image(uploaded_file, caption="Current Case Image", use_container_width=True)
        st.success("Image successfully loaded into agent memory.")

# --- Main View: Chat Interface ---
st.header("Case Discussion")

# Initialize session state to keep track of the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. Please upload a study in the sidebar or ask a clinical question."}
    ]

# Render the existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("E.g., 'Review this image and provide a DDx' or 'What defines a tension pneumothorax?'"):
    
    # 1. Display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Trigger the LangGraph Orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing the case..."):
            
            # Construct the initial state to pass into the graph
            initial_state = {
                "patient_query": prompt,
                "image_path": image_path, # Passes the local path if an image exists
                "vision_findings": "",
                "literature_context": "",
                "final_report": ""
            }
            
            # Invoke the LangGraph workflow
            result = app.invoke(initial_state)
            final_answer = result["final_report"]
            
            # Display the final synthesized report
            st.markdown(final_answer)
            
    # 3. Save the agent's response to history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})