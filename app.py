import streamlit as st
from openai import OpenAI
import json
from typing import Dict, List, Tuple

def initialize_session_state():
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'conversation_step' not in st.session_state:
        st.session_state.conversation_step = 1  # 1 for personal info, 2 for medical info
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

def extract_personal_info(client: OpenAI, text: str) -> Dict[str, str]:
    system_prompt = """
    Extract personal information from the text. Look for:
    - name
    - age
    - location
    
    Return only a JSON object with these fields. If a field is not found, leave it empty.
    Example: {"name": "John", "age": "25", "location": "New York"}
    
    Only extract information that is clearly stated. Do not make assumptions.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error in extracting information: {str(e)}")
        return {}

def extract_medical_info(client: OpenAI, text: str) -> Dict[str, str]:
    system_prompt = """
    Extract medical information from the text. Look for:
    - diagnosis
    - concern
    - target
    
    Return only a JSON object with these fields. If a field is not found, leave it empty.
    Example: {"diagnosis": "diabetes", "concern": "high blood sugar", "target": "better sugar control"}
    
    Only extract information that is clearly stated. Do not make assumptions.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error in extracting information: {str(e)}")
        return {}

def check_missing_personal_fields(data: Dict[str, str]) -> List[str]:
    personal_fields = ['name', 'age', 'location']
    return [field for field in personal_fields if not data.get(field)]

def check_missing_medical_fields(data: Dict[str, str]) -> List[str]:
    medical_fields = ['diagnosis', 'concern', 'target']
    return [field for field in medical_fields if not data.get(field)]

def generate_personal_info_prompt(missing_fields: List[str], form_data: Dict[str, str]) -> str:
    if not missing_fields:
        return "COMPLETE"
    
    prompt = "I see you've provided: \n"
    provided_fields = ['name', 'age', 'location']
    for field in provided_fields:
        if form_data.get(field):
            prompt += f"- {field}: {form_data[field]}\n"
    
    prompt += "\nCould you please provide your " + ", ".join(missing_fields) + "?"
    return prompt

def generate_medical_info_prompt(missing_fields: List[str], form_data: Dict[str, str]) -> str:
    if not missing_fields:
        return "COMPLETE"
    
    name = form_data.get('name', 'there')
    prompt = f"Thank you {name}, now I understand:\n"
    prompt += f"- You are {form_data.get('age')} years old\n"
    prompt += f"- Located in {form_data.get('location')}\n\n"
    
    if not any(form_data.get(field) for field in ['diagnosis', 'concern', 'target']):
        return f"Could you tell me about your medical condition? Please include your diagnosis, main concern, and what you'd like to achieve."
    
    prompt += "For your medical information, I still need to know your " + ", ".join(missing_fields)
    return prompt

def process_conversation_input(client: OpenAI, user_input: str):
    # Add user's response to conversation history
    st.session_state.conversation_history.append(("user", user_input))
    
    # Process based on current step
    if st.session_state.conversation_step == 1:
        # Extract personal information
        extracted_info = extract_personal_info(client, user_input)
        for field, value in extracted_info.items():
            if value:
                st.session_state.form_data[field] = value
        
        # Check missing personal fields
        missing_fields = check_missing_personal_fields(st.session_state.form_data)
        
        if missing_fields:
            # Still need more personal info
            next_prompt = generate_personal_info_prompt(missing_fields, st.session_state.form_data)
            st.session_state.conversation_history.append(("system", next_prompt))
        else:
            # Move to medical information
            st.session_state.conversation_step = 2
            next_prompt = generate_medical_info_prompt(
                check_missing_medical_fields(st.session_state.form_data),
                st.session_state.form_data
            )
            st.session_state.conversation_history.append(("system", next_prompt))
    
    else:  # Step 2
        # Extract medical information
        extracted_info = extract_medical_info(client, user_input)
        for field, value in extracted_info.items():
            if value:
                st.session_state.form_data[field] = value
        
        # Check missing medical fields
        missing_fields = check_missing_medical_fields(st.session_state.form_data)
        
        if missing_fields:
            next_prompt = generate_medical_info_prompt(missing_fields, st.session_state.form_data)
            st.session_state.conversation_history.append(("system", next_prompt))
        else:
            st.session_state.conversation_complete = True
            st.session_state.conversation_history.append(
                ("system", "Thank you! I've collected all the necessary information.")
            )

def main():
    st.title("Medical Information System")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    
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
                    ("system", "Tell me about yourself (name, age, location):")
                )
    with col3:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            initialize_session_state()
    
    st.divider()
    
    # Conversation Mode
    if st.session_state.mode == "conversation":
        st.subheader("Conversation Mode")
        
        # Display conversation history
        for speaker, message in st.session_state.conversation_history:
            if speaker == "system":
                st.markdown(f"🤖 **Assistant**: {message}")
            else:
                st.markdown(f"👤 **You**: {message}")
        
        # Input field with appropriate watermark
        if not st.session_state.conversation_complete:
            placeholder = (
                "Type your name, age, and location..."
                if st.session_state.conversation_step == 1
                else "Type your diagnosis, concern, and target..."
            )
            user_input = st.text_input(
                "Your response",
                key="conversation_input",
                placeholder=placeholder
            )
            if st.button("Send") and user_input:
                process_conversation_input(client, user_input)
                st.rerun()
    
    # Display collected information
    if any(st.session_state.form_data.values()):
        st.divider()
        st.subheader("Collected Information")
        
        # Personal Information
        st.write("Personal Information:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Name:", st.session_state.form_data['name'])
        with col2:
            st.write("Age:", st.session_state.form_data['age'])
        with col3:
            st.write("Location:", st.session_state.form_data['location'])
        
        # Medical Information
        if any(st.session_state.form_data.get(field) for field in ['diagnosis', 'concern', 'target']):
            st.write("Medical Information:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Diagnosis:", st.session_state.form_data['diagnosis'])
            with col2:
                st.write("Concern:", st.session_state.form_data['concern'])
            with col3:
                st.write("Target:", st.session_state.form_data['target'])

if __name__ == "__main__":
    main()
