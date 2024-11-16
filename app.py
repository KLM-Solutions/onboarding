import streamlit as st
from openai import OpenAI
import json
from typing import Dict, List, Tuple

# Initialize OpenAI client
def init_openai():
    if 'OPENAI_API_KEY' not in st.secrets:
        st.error("OpenAI API key not found. Please add it to your secrets.")
        return None
    return OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

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

def extract_information(client: OpenAI, text: str) -> Dict[str, str]:
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error in extracting information: {str(e)}")
        return {}

def generate_next_question(client: OpenAI, current_data: Dict[str, str], conversation_history: List[Tuple[str, str]]) -> str:
    """
    Enhanced version with better handling of repeated responses
    """
    system_prompt = """
    You are an AI medical assistant having a conversation with a patient. 
    Analyze the conversation history and current situation carefully.
    
    Required fields:
    - name
    - age
    - location
    - diagnosis
    - concern
    - target
    
    Special handling rules:
    1. If the user repeats the same answer more than twice, respond with a different approach:
       - Explain why you need the information
       - Provide an example of how to answer
       - Use a more specific question format
    2. If you detect the conversation is stuck:
       - Acknowledge the situation
       - Provide clear examples of expected response
       - Offer multiple choice options if appropriate
    3. For age specifically:
       - If user keeps repeating name/other info, say: "I have your name as [name]. For age, please just type a number, like '25' or '30'."
    
    Generate the next appropriate response based on the conversation flow.
    """
    
    # Count repeated responses
    user_responses = [msg[1] for msg in conversation_history if msg[0] == "user"]
    last_response = user_responses[-1] if user_responses else ""
    repeat_count = sum(1 for r in user_responses[-3:] if r == last_response)
    
    # Prepare the conversation context with additional metadata
    context = {
        "current_data": current_data,
        "conversation_history": conversation_history,
        "repeated_response_count": repeat_count,
        "last_response": last_response,
        "missing_fields": [field for field, value in current_data.items() if not value]
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(context)}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in generating question: {str(e)}")
        return "I'm having trouble understanding. Could you please provide your age as a number? For example: 25"

def process_conversation_input(client: OpenAI, user_input: str):
    """
    Enhanced version with better input processing
    """
    # Add user's response to conversation history
    st.session_state.conversation_history.append(("user", user_input))
    
    # Check for repeated responses
    user_responses = [msg[1] for msg in st.session_state.conversation_history if msg[0] == "user"]
    repeat_count = sum(1 for r in user_responses[-3:] if r == user_input)
    
    if repeat_count >= 3:
        # If user has repeated the same response 3 times, provide more specific guidance
        if "age" in str(st.session_state.conversation_history[-2][1]).lower():
            next_question = "I notice you're repeating your name. For age, I just need a number. For example, if you're 25 years old, just type '25'. How old are you?"
        else:
            next_question = generate_next_question(client, st.session_state.form_data, st.session_state.conversation_history)
        st.session_state.conversation_history.append(("system", next_question))
        return
    
    # Try to extract information
    extracted_info = extract_information(client, user_input)
    
    # Update form data with any new information
    updated = False
    for field, value in extracted_info.items():
        if value and not st.session_state.form_data[field]:
            st.session_state.form_data[field] = value
            updated = True
    
    # If no new information was extracted, handle accordingly
    if not updated and repeat_count > 1:
        missing_fields = [field for field, value in st.session_state.form_data.items() if not value]
        if missing_fields:
            if "age" in missing_fields:
                next_question = "Let me be more specific: I need your age as a number. For example, if you're 25 years old, just type '25'. What is your age?"
            else:
                next_question = generate_next_question(client, st.session_state.form_data, st.session_state.conversation_history)
            st.session_state.conversation_history.append(("system", next_question))
            return
    
    # Generate next question based on missing information
    next_question = generate_next_question(
        client,
        st.session_state.form_data,
        st.session_state.conversation_history
    )
    
    # Check if all information is collected
    if "COMPLETION:" in next_question:
        st.session_state.conversation_complete = True
        st.session_state.conversation_history.append(
            ("system", "Thank you! I've collected all the necessary information.")
        )
    else:
        st.session_state.conversation_history.append(("system", next_question))
def main():
    st.title("Medical Information System")
    
    # Initialize OpenAI client
    client = init_openai()
    if not client:
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
                process_conversation_input(client, user_input)
                st.rerun()  # Using st.rerun() instead of st.experimental_rerun()
    
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
