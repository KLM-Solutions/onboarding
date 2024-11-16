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
    """Initialize all session state variables"""
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

def process_user_input(client: OpenAI, user_input: str) -> str:
    """Process user input and generate next prompt"""
    try:
        # Get appropriate instruction set
        instruction_key = "personal_info" if st.session_state.conversation_step == 1 else "medical_info"
        
        # Extract information
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": get_system_instructions()[instruction_key]},
                {"role": "user", "content": user_input}
            ]
        )
        extracted_info = json.loads(response.choices[0].message.content)
        
        # Update form data with extracted information
        for field, value in extracted_info.items():
            if value:
                st.session_state.form_data[field] = value
        
        # Generate next prompt based on missing fields
        missing = get_missing_fields(st.session_state.conversation_step)
        
        if not missing:
            if st.session_state.conversation_step == 1:
                st.session_state.conversation_step = 2  # Move to next step
                return "Great! Now let's talk about your medical condition. Please share your diagnosis, main concern, and treatment target."
            else:
                st.session_state.conversation_complete = True  # Mark conversation as complete
                return "Thank you! I've collected all the necessary information. You can view the summary in the 'Enter by Field' section."
        
        if st.session_state.conversation_step == 1:
            name = st.session_state.form_data.get('name', '')
            greeting = f"Thank you, {name}! " if name else ""
            return f"{greeting}I still need your {', '.join(missing)}. Please provide them."
        else:
            diagnosis = st.session_state.form_data.get('diagnosis', '')
            if diagnosis:
                return f"I see your diagnosis is {diagnosis}. Could you please also tell me your {', '.join(missing)}?"
            else:
                return f"Please share your {', '.join(missing)}."
                
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        return "I'm having trouble processing that. Could you please try again?"

def display_summary(form_data: Dict[str, str]):
    """Display summary of collected information"""
    if any(form_data.values()):
        # Personal Information
        st.markdown("**Personal Information:**")
        st.write(f"👤 Name: {form_data.get('name', '-')}")
        st.write(f"📅 Age: {form_data.get('age', '-')}")
        st.write(f"📍 Location: {form_data.get('location', '-')}")
        
        # Medical Information
        st.markdown("**Medical Information:**")
        st.write(f"🏥 Diagnosis: {form_data.get('diagnosis', '-')}")
        st.write(f"⚕️ Concern: {form_data.get('concern', '-')}")
        st.write(f"🎯 Target: {form_data.get('target', '-')}")
    else:
        st.info("No information collected yet")

def main():
    st.title("Medical Information System")
    
    # Initialize OpenAI client
    if 'OPENAI_API_KEY' not in st.secrets:
        st.error("OpenAI API key not found. Please add it to your secrets.")
        return
    
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
            st.session_state.conversation_step = 1  # Reset step
            st.session_state.conversation_complete = False  # Reset completion
            st.session_state.conversation_history = [
                ("system", "Tell me about yourself (name, age, location):")
            ]
    with col3:
        if st.button("Reset", use_container_width=True):
            st.session_state.clear()
            initialize_session_state()
            return  # Exit after reset to prevent further processing
    
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
                st.number_input("Age", key="age", 
                              value=int(st.session_state.form_data['age']) if st.session_state.form_data['age'].isdigit() else 0,
                              min_value=0, max_value=150)
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
        
        with col2:
            # Summary section
            st.markdown("### Summary")
            display_summary(st.session_state.form_data)
    
    # Conversation Mode
    elif st.session_state.mode == "conversation":
        st.subheader("Conversation Mode")
        
        # Show current step
        step_text = "Step 1: Personal Information" if st.session_state.conversation_step == 1 else "Step 2: Medical Information"
        st.info(step_text)
        
        # Display conversation history
        for speaker, message in st.session_state.conversation_history:
            if speaker == "system":
                st.markdown(f"🤖 **Assistant**: {message}")
            else:
                st.markdown(f"👤 **You**: {message}")
        
        # Input area with missing fields indicator
        if not st.session_state.conversation_complete:
            missing = get_missing_fields(st.session_state.conversation_step)
            placeholder = f"Type your {', '.join(missing)}..."
            
            # Create a form to handle input submission
            with st.form(key="chat_form"):
                user_input = st.text_input(
                    "Your response",
                    key="conversation_input",
                    placeholder=placeholder
                )
                submit_button = st.form_submit_button("Send")
                
                if submit_button and user_input:
                    # Add user input to history
                    st.session_state.conversation_history.append(("user", user_input))
                    
                    # Process input and get next prompt
                    next_prompt = process_user_input(client, user_input)
                    st.session_state.conversation_history.append(("system", next_prompt))
                    
                    # Clear the input field
                    st.session_state.conversation_input = ""
                    
                    # Only rerun if we're not complete
                    if not st.session_state.conversation_complete:
                        st.rerun()

if __name__ == "__main__":
    main()
