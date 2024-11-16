import streamlit as st
from openai import OpenAI
import json
from typing import Dict, List, Tuple

# System Instructions
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
        User: "I have diabetes and my sugar is high"
        Response: {"diagnosis": "diabetes", "concern": "high sugar", "target": ""}
        """,

        "prompt_generation": """
        You are a medical system assistant guiding a conversation to collect information.
        
        OBJECTIVE:
        Generate appropriate prompts based on missing information and conversation context.

        RULES:
        1. Be conversational but focused
        2. Ask for missing information clearly
        3. Acknowledge provided information
        4. Guide user through steps sequentially
        5. Give examples if user seems confused
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
        fields = ['name', 'age', 'location']
    else:
        fields = ['diagnosis', 'concern', 'target']
    return [field for field in fields if not st.session_state.form_data.get(field)]

def process_user_input(client: OpenAI, user_input: str) -> Dict[str, str]:
    """Process user input using OpenAI API"""
    instructions = get_system_instructions()
    instruction_key = "personal_info" if st.session_state.conversation_step == 1 else "medical_info"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions[instruction_key]},
                {"role": "user", "content": user_input}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        return {}

def generate_next_prompt(client: OpenAI) -> str:
    """Generate the next conversation prompt"""
    missing = get_missing_fields(st.session_state.conversation_step)
    
    if not missing:
        if st.session_state.conversation_step == 1:
            return "Great! Now let's talk about your medical condition. Please share your diagnosis, main concern, and treatment target."
        else:
            return "Thank you! I've collected all the necessary information."
    
    context = {
        "step": st.session_state.conversation_step,
        "missing_fields": missing,
        "collected_data": st.session_state.form_data
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_system_instructions()["prompt_generation"]},
                {"role": "user", "content": f"Generate next prompt for context: {context}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating prompt: {str(e)}")
        return f"Please provide your {', '.join(missing)}."

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
            if not st.session_state.conversation_history:
                st.session_state.conversation_history.append(
                    ("system", "Tell me about yourself (name, age, location):")
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
        
        # Show current step
        current_step = "Step 1: Personal Information" if st.session_state.conversation_step == 1 else "Step 2: Medical Information"
        st.info(current_step)
        
        # Display conversation history
        for speaker, message in st.session_state.conversation_history:
            if speaker == "system":
                st.markdown(f"🤖 **Assistant**: {message}")
            else:
                st.markdown(f"👤 **You**: {message}")
        
        # Input area
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
                # Process user input
                extracted_info = process_user_input(client, user_input)
                
                # Update form data
                for field, value in extracted_info.items():
                    if value:
                        st.session_state.form_data[field] = value
                
                # Update conversation history
                st.session_state.conversation_history.append(("user", user_input))
                
                # Check step completion
                missing_fields = get_missing_fields(st.session_state.conversation_step)
                if not missing_fields:
                    if st.session_state.conversation_step == 1:
                        st.session_state.conversation_step = 2
                        next_prompt = "Great! Now let's talk about your medical condition. Please share your diagnosis, main concern, and treatment target."
                    else:
                        st.session_state.conversation_complete = True
                        next_prompt = "Thank you! I've collected all the necessary information."
                else:
                    next_prompt = generate_next_prompt(client)
                
                st.session_state.conversation_history.append(("system", next_prompt))
                st.rerun()
        
        # Display collected information
        if any(st.session_state.form_data.values()):
            st.divider()
            st.subheader("Collected Information")
            
            # Personal Information
            if any(st.session_state.form_data.get(field) for field in ['name', 'age', 'location']):
                st.write("Personal Information:")
                cols = st.columns(3)
                with cols[0]:
                    st.write("Name:", st.session_state.form_data.get('name', ''))
                with cols[1]:
                    st.write("Age:", st.session_state.form_data.get('age', ''))
                with cols[2]:
                    st.write("Location:", st.session_state.form_data.get('location', ''))
            
            # Medical Information
            if any(st.session_state.form_data.get(field) for field in ['diagnosis', 'concern', 'target']):
                st.write("Medical Information:")
                cols = st.columns(3)
                with cols[0]:
                    st.write("Diagnosis:", st.session_state.form_data.get('diagnosis', ''))
                with cols[1]:
                    st.write("Concern:", st.session_state.form_data.get('concern', ''))
                with cols[2]:
                    st.write("Target:", st.session_state.form_data.get('target', ''))

if __name__ == "__main__":
    main()
