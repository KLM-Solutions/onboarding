import streamlit as st
from openai import OpenAI
import json
from typing import Dict, List, Tuple

def get_system_instructions():
    return {
        "personal_info": """
        You are a medical system assistant collecting personal information.
        
        OBJECTIVE:
        Extract personal information from user input, focusing on three key fields:
        1. name
        2. age
        3. location

        RULES:
        1. Only extract information that is explicitly stated
        2. Format response as JSON: {"name": "", "age": "", "location": ""}
        3. If a field is missing, leave it empty
        4. For age, only accept numeric values
        5. Don't make assumptions about missing information

        EXAMPLES:
        User: "My name is John"
        Response: {"name": "John", "age": "", "location": ""}

        User: "I'm 25 and live in New York"
        Response: {"age": "25", "location": "New York", "name": ""}
        """,

        "medical_info": """
        You are a medical system assistant collecting information about a patient's condition.
        
        OBJECTIVE:
        Extract medical information from user input, focusing on three key fields:
        1. diagnosis
        2. concern
        3. target

        RULES:
        1. Only extract information that is explicitly stated
        2. Format response as JSON: {"diagnosis": "", "concern": "", "target": ""}
        3. If a field is missing, leave it empty
        4. Keep medical terminology as stated by the user
        5. Don't make medical assumptions or suggestions

        EXAMPLES:
        User: "I have diabetes"
        Response: {"diagnosis": "diabetes", "concern": "", "target": ""}

        User: "My blood sugar is high and I want to control it"
        Response: {"diagnosis": "", "concern": "high blood sugar", "target": "control blood sugar"}
        """
    }

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
    if 'conversation_step' not in st.session_state:
        st.session_state.conversation_step = 1
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'conversation_complete' not in st.session_state:
        st.session_state.conversation_complete = False
    if 'attempt_count' not in st.session_state:
        st.session_state.attempt_count = 0
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""

def get_missing_fields(step: int) -> List[str]:
    """Get list of missing fields for current step"""
    if step == 1:
        return [field for field in ['name', 'age', 'location'] 
                if not st.session_state.form_data.get(field)]
    else:
        return [field for field in ['diagnosis', 'concern', 'target'] 
                if not st.session_state.form_data.get(field)]

def check_step_completion(step: int) -> bool:
    """Check if all required fields for the current step are filled"""
    if step == 1:
        return all(st.session_state.form_data.get(field) 
                  for field in ['name', 'age', 'location'])
    else:
        return all(st.session_state.form_data.get(field) 
                  for field in ['diagnosis', 'concern', 'target'])

def get_format_example() -> str:
    """Get appropriate format example based on conversation step"""
    if st.session_state.conversation_step == 1:
        return "Example: 'My name is John Smith, I'm 35 years old, and I live in New York.'"
    else:
        return "Example: 'I've been diagnosed with diabetes, my main concern is high blood sugar, and my target is to maintain normal blood sugar levels.'"

def generate_response(any_info_extracted: bool) -> str:
    """Generate appropriate response based on conversation state"""
    missing = get_missing_fields(st.session_state.conversation_step)
    
    # If no information was extracted
    if not any_info_extracted:
        return f"I couldn't understand your information. Please provide your {', '.join(missing)} like this:\n\n{get_format_example()}"
    
    # If all fields for current step are complete
    if not missing:
        if st.session_state.conversation_step == 1:
            st.session_state.conversation_step = 2
            return "Thank you for your personal information! Now, please tell me about your medical condition:\n\n" + get_format_example()
        else:
            st.session_state.conversation_complete = True
            return "Thank you! I've collected all the necessary information. You can view the summary in the sidebar."
    
    # If some fields are still missing
    if st.session_state.conversation_step == 1:
        return f"Thank you! I still need your {', '.join(missing)}. Please provide them like this:\n\n{get_format_example()}"
    else:
        return f"Thank you! I still need information about your {', '.join(missing)}. Please provide them like this:\n\n{get_format_example()}"

def process_user_input(client: OpenAI, user_input: str) -> str:
    """Process user input and generate next prompt"""
    try:
        # Prevent repeated identical inputs
        if user_input == st.session_state.last_input:
            st.session_state.attempt_count += 1
            if st.session_state.attempt_count >= 3:
                return "I notice you're repeating the same input. Let me help you provide the information in the right format:\n\n" + get_format_example()
            
        # Reset attempt count if input is different
        if user_input != st.session_state.last_input:
            st.session_state.attempt_count = 0
            st.session_state.last_input = user_input

        # Get appropriate instruction set
        instruction_key = "personal_info" if st.session_state.conversation_step == 1 else "medical_info"
        
        # Extract information
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Updated to a valid model name
            messages=[
                {"role": "system", "content": get_system_instructions()[instruction_key]},
                {"role": "user", "content": user_input}
            ]
        )
        
        extracted_info = json.loads(response.choices[0].message.content)
        
        # Update form data with extracted information
        any_info_extracted = False
        for field, value in extracted_info.items():
            if value and not st.session_state.form_data[field]:  # Only update empty fields
                st.session_state.form_data[field] = value
                any_info_extracted = True

        # Return appropriate response
        return generate_response(any_info_extracted)
                
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        return "I'm having trouble processing that. Please provide your information in a clear format:\n\n" + get_format_example()

def display_summary():
    """Display summary of collected information"""
    st.sidebar.markdown("### Information Summary")
    
    # Personal Information
    st.sidebar.markdown("**Personal Information:**")
    st.sidebar.write(f"👤 Name: {st.session_state.form_data.get('name', '-')}")
    st.sidebar.write(f"📅 Age: {st.session_state.form_data.get('age', '-')}")
    st.sidebar.write(f"📍 Location: {st.session_state.form_data.get('location', '-')}")
    
    # Medical Information
    st.sidebar.markdown("**Medical Information:**")
    st.sidebar.write(f"🏥 Diagnosis: {st.session_state.form_data.get('diagnosis', '-')}")
    st.sidebar.write(f"⚕️ Concern: {st.session_state.form_data.get('concern', '-')}")
    st.sidebar.write(f"🎯 Target: {st.session_state.form_data.get('target', '-')}")

def main():
    st.set_page_config(page_title="Medical Information System", layout="wide")
    
    st.title("Medical Information System")
    
    # Initialize OpenAI client
    if 'OPENAI_API_KEY' not in st.secrets:
        st.error("OpenAI API key not found. Please add it to your secrets.")
        return
    
    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    
    # Initialize session state
    initialize_session_state()
    
    # Display summary in sidebar
    display_summary()
    
    # Mode selection
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Enter by Field", use_container_width=True):
            st.session_state.mode = "field"
            # Reset conversation state when switching modes
            st.session_state.conversation_history = []
            st.session_state.conversation_step = 1
            st.session_state.conversation_complete = False
    with col2:
        if st.button("Enter by Conversation", use_container_width=True):
            st.session_state.mode = "conversation"
            # Initialize conversation if empty
            if not st.session_state.conversation_history:
                st.session_state.conversation_history.append(
                    ("system", "Please provide your personal information:\n\n" + get_format_example())
                )
    with col3:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            initialize_session_state()
            st.rerun()
    
    st.divider()
    
    # Form Mode
    if st.session_state.mode == "field":
        st.subheader("Enter Information by Field")
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Form for input
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
        
        # Show progress
        progress = st.progress(0)
        if st.session_state.conversation_complete:
            progress.progress(1.0)
        elif st.session_state.conversation_step == 2:
            progress.progress(0.5)
        
        # Display conversation history
        for speaker, message in st.session_state.conversation_history:
            if speaker == "system":
                st.markdown(f"🤖 **Assistant**: {message}")
            else:
                st.markdown(f"👤 **You**: {message}")
        
        # Input area
        if not st.session_state.conversation_complete:
            user_input = st.text_input(
                "Your response",
                key="conversation_input",
                help=f"Step {st.session_state.conversation_step}: " + 
                     ("Personal Information" if st.session_state.conversation_step == 1 else "Medical Information")
            )
            
            if st.button("Send", type="primary", use_container_width=True) and user_input:
                # Add user input to history
                st.session_state.conversation_history.append(("user", user_input))
                
                # Process input and get next prompt
                next_prompt = process_user_input(client, user_input)
                st.session_state.conversation_history.append(("system", next_prompt))
                
                st.rerun()

if __name__ == "__main__":
    main()
