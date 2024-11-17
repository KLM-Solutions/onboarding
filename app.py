import streamlit as st
from openai import OpenAI
import json
from typing import Dict, List, Tuple

def get_system_instructions():
    return {
        "personal_info": """
        You are a medical system assistant collecting personal information. You must strictly validate all information provided.
        
        OBJECTIVE:
        Extract and validate personal information, ensuring data accuracy and real-world validity.
        Required fields:
        1. name (string)
        2. age (numeric)
        3. location (geographic)

        VALIDATION RULES:
        1. Names:
           - Accept alphabetic characters, spaces, and common name punctuation
           - Reject numeric or special characters
        
        2. Age:
           - Must be numeric values only
           - Must be within human life expectancy (0-150)
           - Reject non-numeric or unrealistic ages
        
        3. Location:
           - Must be a verifiable real-world geographic location
           - Format as City, State/Province/Region, Country where possible
           - Reject any non-geographic or fictional locations
           - Reject astronomical or non-Earth locations
        
        Output Format:
        {
            "name": string or empty,
            "age": numeric string or empty,
            "location": validated location string or empty,
            "validation_errors": [array of specific error messages]
        }

     
        "medical_info": """
        You are a medical system assistant collecting medical information. You must strictly validate all medical data.
        
        OBJECTIVE:
        Extract and validate medical information, ensuring clinical accuracy and proper terminology.
        Required fields:
        1. diagnosis (medical condition)
        2. concern (symptoms/issues)
        3. target (treatment goals)

        VALIDATION RULES:
        1. Diagnosis:
           - Must be recognized medical conditions using proper clinical terminology
           - Reject non-medical terms, abbreviations, or colloquial descriptions
           - Verify against known medical conditions database
           - Flag any unusual or non-standard medical terms
        
        2. Concerns:
           - Must relate to medical symptoms or health issues
           - Accept both technical and lay descriptions of symptoms
           - Must be specific and health-related
        
        3. Treatment Targets:
           - Must be realistic medical goals
           - Must align with the stated diagnosis
           - Must be specific and measurable where possible
        
        Output Format:
        {
            "diagnosis": validated medical term or empty,
            "concern": validated concern or empty,
            "target": validated target or empty,
            "validation_errors": [array of specific error messages]
        }

      
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
                return "Great! Now let's talk about your medical condition. Please share your diagnosis, main concern, and treatment target."
            else:
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
        
        with col2:
            # Summary section
            st.markdown("### Summary")
            if any(st.session_state.form_data.values()):
                # Personal Information
                st.markdown("**Personal Information:**")
                st.write(f"👤 Name: {st.session_state.form_data.get('name', '-')}")
                st.write(f"📅 Age: {st.session_state.form_data.get('age', '-')}")
                st.write(f"📍 Location: {st.session_state.form_data.get('location', '-')}")
                
                # Medical Information
                st.markdown("**Medical Information:**")
                st.write(f"🏥 Diagnosis: {st.session_state.form_data.get('diagnosis', '-')}")
                st.write(f"⚕️ Concern: {st.session_state.form_data.get('concern', '-')}")
                st.write(f"🎯 Target: {st.session_state.form_data.get('target', '-')}")
            else:
                st.info("No information collected yet")
    
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
            
            user_input = st.text_input(
                "Your response",
                key="conversation_input",
                placeholder=placeholder
            )
            
            if st.button("Send") and user_input:
                # Add user input to history
                st.session_state.conversation_history.append(("user", user_input))
                
                # Process input and get next prompt
                next_prompt = process_user_input(client, user_input)
                
                # Check step completion and update accordingly
                if check_step_completion(st.session_state.conversation_step):
                    if st.session_state.conversation_step == 1:
                        st.session_state.conversation_step = 2
                        next_prompt = "Great! Now let's talk about your medical condition. Please share your diagnosis, main concern, and treatment target."
                    else:
                        st.session_state.conversation_complete = True
                        next_prompt = "Thank you! I've collected all the necessary information. You can view the summary in the 'Enter by Field' section."
                
                st.session_state.conversation_history.append(("system", next_prompt))
                st.rerun()

if __name__ == "__main__":
    main()
