import streamlit as st
import openai
import re
from typing import Dict, List, Tuple

# Initialize OpenAI (you'll need to set your API key)
def init_openai():
    if 'OPENAI_API_KEY' not in st.secrets:
        st.error("OpenAI API key not found. Please add it to your secrets.")
        return False
    openai.api_key = st.secrets['OPENAI_API_KEY']
    return True

def initialize_session_state():
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'name': '',
            'age': '',
            'location': '',
            'diagnosis': '',
            'concern': '',
            'target': ''
        }
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'conversation_complete' not in st.session_state:
        st.session_state.conversation_complete = False

def extract_information(text: str) -> Dict[str, str]:
    """
    Use LLM to extract relevant information from user's text
    """
    system_prompt = """
    You are an AI medical assistant. Extract the following information from the user's text:
    - name
    - age
    - location
    - diagnosis
    - concern
    - target
    
    Return only a JSON object with the found fields. If a field is not found, leave it empty.
    Example: {"name": "John", "age": "25", "location": "", "diagnosis": "", "concern": "", "target": ""}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
        )
        return eval(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error in extracting information: {str(e)}")
        return {}

def generate_next_question(current_data: Dict[str, str], conversation_history: List[Tuple[str, str]]) -> str:
    """
    Use LLM to generate the next appropriate question based on missing information
    """
    system_prompt = """
    You are an AI medical assistant having a conversation with a patient. 
    Generate the next question based on the missing information.
    
    Required fields:
    - name
    - age
    - location
    - diagnosis
    - concern
    - target
    
    Rules:
    1. Ask for one piece of missing information at a time
    2. Be conversational and natural
    3. Reference previously provided information when relevant
    4. If all information is complete, indicate with: "COMPLETION: All information collected"
    5. Make the questions contextual based on previous responses
    
    Previous conversation and current data will be provided. Generate only the next question.
    """
    
    # Prepare the conversation context
    context = "Current information:\n"
    for field, value in current_data.items():
        context += f"{field}: {value}\n"
    
    context += "\nConversation history:\n"
    for speaker, text in conversation_history:
        context += f"{speaker}: {text}\n"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in generating question: {str(e)}")
        return "I apologize, but I'm having trouble generating the next question. Could you please provide any missing information?"

def process_conversation_input(user_input: str):
    # Add user's response to conversation history
    st.session_state.conversation_history.append(("user", user_input))
    
    # Extract information from user's response
    extracted_info = extract_information(user_input)
    
    # Update form data with any new information
    for field, value in extracted_info.items():
        if value and not st.session_state.form_data[field]:
            st.session_state.form_data[field] = value
    
    # Generate next question based on missing information
    next_question = generate_next_question(
        st.session_state.form_data,
        st.session_state.conversation_history
    )
    
    # Check if all information is collected
    if next_question.startswith("COMPLETION:"):
        st.session_state.conversation_complete = True
        st.session_state.conversation_history.append(
            ("system", "Thank you! I've collected all the necessary information.")
        )
    else:
        st.session_state.conversation_history.append(("system", next_question))

def main():
    st.title("Medical Information System")
    
    # Initialize OpenAI
    if not init_openai():
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Mode selection
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Enter by Field", use_container_width=True):
            st.session_state.mode = "field"
    with col2:
        if st.button("Enter by Conversation", use_container_width=True):
            st.session_state.mode = "conversation"
            if not st.session_state.conversation_history:
                st.session_state.conversation_history.append(
                    ("system", "Hello! I'm here to help collect your information. Could you please tell me your name?")
                )
    with col3:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            initialize_session_state()
    
    st.divider()
    
    # Form Mode
    if st.session_state.mode == "field":
        st.subheader("Enter Information by Field")
        with st.form("medical_form"):
            st.text_input("Name", key="name", value=st.session_state.form_data['name'])
            st.number_input("Age", key="age", value=int(st.session_state.form_data['age']) if st.session_state.form_data['age'].isdigit() else 0, min_value=0, max_value=150)
            st.text_input("Location", key="location", value=st.session_state.form_data['location'])
            st.text_input("Diagnosis", key="diagnosis", value=st.session_state.form_data['diagnosis'])
            st.text_area("Concern", key="concern", value=st.session_state.form_data['concern'])
            st.text_area("Target", key="target", value=st.session_state.form_data['target'])
            
            if st.form_submit_button("Submit"):
                st.session_state.form_data = {
                    'name': st.session_state.name,
                    'age': str(st.session_state.age),
                    'location': st.session_state.location,
                    'diagnosis': st.session_state.diagnosis,
                    'concern': st.session_state.concern,
                    'target': st.session_state.target
                }
                st.success("Information submitted successfully!")
    
    # Conversation Mode
    elif st.session_state.mode == "conversation":
        st.subheader("Conversation Mode")
        
        # Display conversation history in a chat-like interface
        for speaker, message in st.session_state.conversation_history:
            if speaker == "system":
                st.markdown(f"🤖 **Assistant**: {message}")
            else:
                st.markdown(f"👤 **You**: {message}")
        
        # Input field for conversation
        if not st.session_state.conversation_complete:
            user_input = st.text_input("Your response", key="conversation_input")
            if st.button("Send") and user_input:
                process_conversation_input(user_input)
                st.experimental_rerun()
    
    # Display collected information
    if any(st.session_state.form_data.values()):
        st.divider()
        st.subheader("Collected Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Name:", st.session_state.form_data['name'])
            st.write("Age:", st.session_state.form_data['age'])
            st.write("Location:", st.session_state.form_data['location'])
        with col2:
            st.write("Diagnosis:", st.session_state.form_data['diagnosis'])
            st.write("Concern:", st.session_state.form_data['concern'])
            st.write("Target:", st.session_state.form_data['target'])

if __name__ == "__main__":
    main()
