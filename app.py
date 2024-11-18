import streamlit as st
import requests
import json
import re 
from typing import Dict, Any, Optional, Generator
from openai import OpenAI

class UserProfileManager:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.system_instructions = {
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
            """
        }

    def process_user_input(self, user_input: str, info_type: str) -> Dict[str, str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_instructions[info_type]},
                    {"role": "user", "content": user_input}
                ]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            return {}

class GLP1Bot:
    def __init__(self, pplx_api_key: str):
        self.pplx_api_key = pplx_api_key
        self.pplx_model = "medical-pplx"
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        self.pplx_system_prompt = """
        You are a specialized medical information assistant focused on providing personalized GLP-1 medication information. You must:

        1. CORE RESPONSIBILITIES:
           - ONLY provide information about GLP-1 medications (Ozempic, Wegovy, Mounjaro, etc.)
           - Personalize responses based on the patient profile provided
           - Consider patient's specific diagnosis, concerns, and treatment targets
           - Maintain medical accuracy while being accessible

        2. RESPONSE STRUCTURE:
           a) Patient-Specific Context
              - Acknowledge the patient's specific situation
              - Reference relevant aspects of their medical profile
           
           b) Medical Information
              - Provide GLP-1 specific information relevant to their query
              - Explain how it relates to their condition
              - Include relevant drug interactions or contraindications
           
           c) Safety Considerations
              - Highlight important safety information
              - Note specific precautions based on patient profile
              - Include relevant warnings or contraindications
           
           d) Personalized Recommendations
              - Suggest relevant questions for their healthcare provider
              - Provide lifestyle considerations specific to their profile
           
           e) Sources and Disclaimers
              - Cite medical sources for information provided
              - Include appropriate medical disclaimers

        3. MANDATORY RULES:
           - NEVER provide medical advice outside of GLP-1 medications
           - ALWAYS consider age-specific considerations
           - ALWAYS include relevant contraindications
           - ALWAYS provide source citations
           - MAINTAIN an 11th grade or lower reading level
           - FOCUS on patient-specific concerns

        4. FORMAT REQUIREMENTS:
           - Structure responses clearly with headers
           - Use bullet points for key information
           - Include sources in hyperlink format
           - End with standard medical disclaimer

        5. QUERY HANDLING:
           - For non-GLP-1 queries, respond: "I apologize, but I can only provide information about GLP-1 medications and related topics. Your question appears to be about something else. Please ask a question specifically about GLP-1 medications."
           - For unclear queries, ask for clarification
           - For emergency situations, direct to immediate medical care

        Remember: You are analyzing the query in the context of the patient's specific profile and providing personalized, relevant information about GLP-1 medications only.
        """

    def categorize_query(self, query: str) -> str:
        """Categorize the user query"""
        categories = {
            "dosage": ["dose", "dosage", "how to take", "when to take", "injection", "administration"],
            "side_effects": ["side effect", "adverse", "reaction", "problem", "issues", "symptoms"],
            "benefits": ["benefit", "advantage", "help", "work", "effect", "weight", "glucose"],
            "storage": ["store", "storage", "keep", "refrigerate", "temperature"],
            "lifestyle": ["diet", "exercise", "lifestyle", "food", "alcohol", "eating"],
            "interactions": ["interaction", "drug", "medication", "combine", "mixing"],
            "cost": ["cost", "price", "insurance", "coverage", "afford"]
        }
        
        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"
        
    def generate_personalized_prompt(self, query: str, user_profile: Dict[str, str]) -> str:
        """Generate a personalized prompt based on user profile"""
        return f"""Context for Response Generation:

Patient Profile:
- Name: {user_profile.get('name', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')}
- Location: {user_profile.get('location', 'Unknown')}
- Diagnosis: {user_profile.get('diagnosis', 'Unknown')}
- Medical Concern: {user_profile.get('concern', 'Unknown')}
- Treatment Target: {user_profile.get('target', 'Unknown')}

User Query: {query}

Please provide a personalized response considering:
1. The patient's specific medical condition and concerns
2. Age-specific considerations for GLP-1 medications
3. Any relevant interactions with their current diagnosis
4. How the information relates to their treatment targets
"""

    def stream_pplx_response(self, query: str, user_profile: Dict[str, str]) -> Generator[Dict[str, Any], None, None]:
        try:
            personalized_query = self.generate_personalized_prompt(query, user_profile)
            
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": personalized_query}
                ],
                "temperature": 0.1,
                "max_tokens": 1500,
                "stream": True
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload,
                stream=True
            )
            
            response.raise_for_status()
            accumulated_content = ""
            found_sources = False
            sources_text = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            json_str = line[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            
                            chunk = json.loads(json_str)
                            if chunk['choices'][0]['finish_reason'] is not None:
                                break
                                
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                if "Sources:" in content:
                                    found_sources = True
                                    parts = content.split("Sources:", 1)
                                    if len(parts) > 1:
                                        accumulated_content += parts[0]
                                        sources_text += parts[1]
                                    else:
                                        accumulated_content += parts[0]
                                elif found_sources:
                                    sources_text += content
                                else:
                                    accumulated_content += content
                                
                                yield {
                                    "type": "content",
                                    "data": content,
                                    "accumulated": accumulated_content
                                }
                                
                        except json.JSONDecodeError:
                            continue
            
            formatted_sources = self.format_sources_as_hyperlinks(sources_text.strip()) if sources_text.strip() else "No sources provided"
            
            yield {
                "type": "complete",
                "content": accumulated_content.strip(),
                "sources": formatted_sources
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error communicating with PPLX: {str(e)}"
            }

    def format_sources_as_hyperlinks(self, sources_text: str) -> str:
        """Convert source text into formatted hyperlinks"""
        clean_text = re.sub(r'<[^>]+>', '', sources_text)
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.finditer(url_pattern, clean_text)
        formatted_text = clean_text
        
        for url_match in urls:
            url = url_match.group(0)
            title_match = re.search(rf'([^.!?\n]+)(?=\s*{re.escape(url)})', formatted_text)
            title = title_match.group(1).strip() if title_match else url
            hyperlink = f'[{title}]({url})'
            if title_match:
                formatted_text = formatted_text.replace(f'{title_match.group(0)} {url}', hyperlink)
            else:
                formatted_text = formatted_text.replace(url, hyperlink)
        
        return formatted_text

def set_page_style():
    """Set page style using custom CSS"""
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput>div>div>input {
            background-color: white;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.8rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
        }
        .bot-message {
            background-color: #f5f5f5;
            border-left: 4px solid #43a047;
        }
        .category-tag {
            background-color: #2196f3;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
            display: inline-block;
        }
        .sources-section {
            background-color: #fff3e0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            border-left: 4px solid #ff9800;
        }
        .profile-section {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 4px solid #43a047;
        }
        .step-indicator {
            background-color: #bbdefb;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'profile_complete' not in st.session_state:
        st.session_state.profile_complete = False
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'age': '',
            'location': '',
            'diagnosis': '',
            'concern': '',
            'target': ''
        }
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'personal_info'

def display_profile_summary():
    """Display user profile summary"""
    st.markdown("""
        <div class="profile-section">
            <h4>Your Profile</h4>
            <p><strong>Personal Information:</strong></p>
            <ul>
                <li>Name: {name}</li>
                <li>Age: {age}</li>
                <li>Location: {location}</li>
            </ul>
            <p><strong>Medical Information:</strong></p>
            <ul>
                <li>Diagnosis: {diagnosis}</li>
                <li>Primary Concern: {concern}</li>
                <li>Treatment Target: {target}</li>
            </ul>
        </div>
    """.format(**st.session_state.user_profile), unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Personalized GLP-1 Medical Assistant",
        page_icon="💊",
        layout="wide"
    )
    
    set_page_style()
    
    # Check for required API keys
    if 'OPENAI_API_KEY' not in st.secrets or 'PPLX_API_KEY' not in st.secrets:
        st.error("Required API keys not found. Please configure both OpenAI and PPLX API keys.")
        return
    
    # Initialize clients and session state
    openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    profile_manager = UserProfileManager(openai_client)
    glp1_bot = GLP1Bot(st.secrets['PPLX_API_KEY'])
    
    initialize_session_state()
    
    st.title("💊 Personalized GLP-1 Medication Assistant")
    
    # Profile Collection Phase
    if not st.session_state.profile_complete:
        st.info("Let's collect some information to provide you with personalized guidance.")
        
        if st.session_state.current_step == 'personal_info':
            st.markdown('<div class="step-indicator">Step 1: Personal Information</div>', unsafe_allow_html=True)
            personal_info = st.text_input(
                "Please enter your name, age, and location:",
                help="Example: My name is John Smith, I'm 45 years old and live in New York"
            )
            
            if st.button("Next") and personal_info:
                extracted_info = profile_manager.process_user_input(personal_info, "personal_info")
                st.session_state.user_profile.update(extracted_info)
                if all(st.session_state.user_profile[field] for field in ['name', 'age', 'location']):
                    st.session_state.current_step = 'medical_info'
                    st.rerun()
                else:
                    st.warning("Please provide all required personal information (name, age, and location).")
                
        elif st.session_state.current_step == 'medical_info':
            st.markdown('<div class="step-indicator">Step 2: Medical Information</div>', unsafe_allow_html=True)
            
            # Show collected personal information
            st.markdown("**Collected Personal Information:**")
            display_profile_summary()
            
            medical_info = st.text_input(
                "Please describe your diagnosis, main medical concern, and treatment target:",
                help="Example: I have type 2 diabetes, concerned about blood sugar control, aiming to manage weight and glucose levels"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("← Back"):
                    st.session_state.current_step = 'personal_info'
                    st.rerun()
            with col2:
                if st.button("Complete Profile") and medical_info:
                    extracted_info = profile_manager.process_user_input(medical_info, "medical_info")
                    st.session_state.user_profile.update(extracted_info)
                    if all(st.session_state.user_profile[field] for field in ['diagnosis', 'concern', 'target']):
                        st.session_state.profile_complete = True
                        st.rerun()
                    else:
                        st.warning("Please provide all required medical information (diagnosis, concern, and target).")
    
    # GLP-1 Query Phase
    else:
        # Create two columns for layout
        col1, col2 = st.columns([1, 3])
        
        # Sidebar with profile information and controls
        with col1:
            st.markdown("### Your Profile")
            display_profile_summary()
            
            if st.button("Edit Profile"):
                st.session_state.profile_complete = False
                st.session_state.current_step = 'personal_info'
                st.rerun()
        
        # Main chat interface
        with col2:
            st.markdown("### Ask about GLP-1 Medications")
            st.markdown("""
            <div class="info-box">
            Now that we have your profile information, you can ask specific questions about GLP-1 medications. 
            Your responses will be personalized based on your medical profile.
            </div>
            """, unsafe_allow_html=True)
            
            user_query = st.text_input(
                "What would you like to know about GLP-1 medications?",
                placeholder="e.g., What are the common side effects of Ozempic?"
            )
            
            if st.button("Get Answer") and user_query:
                query_category = glp1_bot.categorize_query(user_query)
                
                # Display user question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>Your Question:</b><br>{user_query}
                </div>
                """, unsafe_allow_html=True)
                
                # Create placeholder for streaming response
                response_placeholder = st.empty()
                sources_placeholder = st.empty()
                
                # Initialize response content
                full_response = ""
                
                # Stream response
                for chunk in glp1_bot.stream_pplx_response(user_query, st.session_state.user_profile):
                    if chunk["type"] == "error":
                        st.error(chunk["message"])
                        break
                        
                    elif chunk["type"] == "content":
                        full_response = chunk["accumulated"]
                        response_placeholder.markdown(f"""
                        <div class="chat-message bot-message">
                            <div class="category-tag">{query_category.upper()}</div>
                            {full_response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif chunk["type"] == "complete":
                        sources_placeholder.markdown(f"""
                        <div class="sources-section">
                            <b>Sources:</b><br>
                            {chunk["sources"]}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "response": full_response,
                    "category": query_category
                })
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### Previous Questions")
                for i, chat in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
                    with st.expander(f"Question {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>Your Question:</b><br>{chat['query']}
                        </div>
                        <div class="chat-message bot-message">
                            <div class="category-tag">{chat['category'].upper()}</div>
                            {chat['response']}
                        </div>
                        """, unsafe_allow_html=True)

def validate_api_keys():
    """Validate the presence and basic format of required API keys"""
    required_keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'PPLX_API_KEY': 'Perplexity'
    }
    
    missing_keys = []
    for key, service in required_keys.items():
        if key not in st.secrets:
            missing_keys.append(service)
        elif not st.secrets[key].strip():
            missing_keys.append(service)
    
    if missing_keys:
        st.error(f"Missing API keys for: {', '.join(missing_keys)}")
        return False
    return True

if __name__ == "__main__":
    try:
        if validate_api_keys():
            main()
        else:
            st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
