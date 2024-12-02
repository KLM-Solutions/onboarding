import streamlit as st
import requests
import json
import re 
from typing import Dict, Any, Optional, Generator
from openai import OpenAI

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

class ProfileAnalyzer:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.analysis_prompt = """
        You are a medical profile analyzer specializing in GLP-1 medication contexts. 
        Review the following patient profile and provide a concise analysis focusing on:

        1. Key Risk Factors:
           - Age-related considerations
           - Diagnosis-specific concerns
           - Potential contraindications

        2. Treatment Context:
           - Relevance of GLP-1 medications to their condition
           - Important monitoring considerations
           - Lifestyle factors to consider

        3. Special Considerations:
           - Key drug interactions to watch for
           - Specific precautions based on medical history
           - Priority health targets

        Format the response as a structured summary that can be used to inform GLP-1 medication discussions.
        Keep the analysis focused and relevant to GLP-1 medications.
        """

    def analyze_profile(self, profile: Dict[str, str]) -> str:
        try:
            prompt = f"""
            Patient Profile:
            - Name: {profile['name']}
            - Age: {profile['age']}
            - Location: {profile['location']}
            - Diagnosis: {profile['diagnosis']}
            - Primary Concern: {profile['concern']}
            - Treatment Target: {profile['target']}

            {self.analysis_prompt}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical profile analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error analyzing profile: {str(e)}")
            return "Error generating profile analysis"

class GLP1Bot:
    def __init__(self, pplx_api_key: str):
        self.pplx_api_key = pplx_api_key
        self.pplx_model = "llama-3.1-sonar-large-128k-online"
        
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        self.pplx_system_prompt = """
        You are a specialized medical information assistant providing highly personalized GLP-1 medication information.
        
        CORE RESPONSIBILITIES:
        1. Provide information EXCLUSIVELY about GLP-1 medications (e.g., Ozempic, Wegovy, Mounjaro)
        2. Tailor responses to the patient's specific profile and medical conditions
        3. Consider age-specific factors and medical history
        4. Highlight relevant interactions with existing conditions
        5. Address patient's specific concerns and treatment targets

        RESPONSE STRUCTURE:
        1. Personal Acknowledgment
           - Reference patient's name and relevant profile details
           - Acknowledge their specific medical situation
        
        2. Targeted Answer
           - Address the specific query
           - Connect information to their medical context
           - Consider their diagnosis and treatment targets
        
        3. Safety and Precautions
           - Highlight relevant warnings based on their profile
           - Note specific contraindications for their condition
           - Address age-specific considerations
        
        4. Personalized Recommendations
           - Suggest relevant monitoring based on their condition
           - Provide lifestyle recommendations aligned with their goals
           - Consider their location for practical advice
        
        5. Next Steps
           - Suggest specific questions for their healthcare provider
           - Recommend relevant monitoring based on their profile
           - Provide actionable takeaways

        6. Medical Disclaimer
           - Include standard medical disclaimer
           - Encourage healthcare provider consultation

        PERSONALIZATION RULES:
        1. For diabetic patients:
           - Focus on blood sugar management
           - Discuss insulin interaction
           - Address hypoglycemia risks
        
        2. For obesity management:
           - Focus on weight loss expectations
           - Discuss lifestyle integration
           - Address dietary considerations
        
        3. For older patients (65+):
           - Emphasize slower titration
           - Focus on side effect management
           - Discuss monitoring requirements
        
        4. For multiple conditions:
           - Address medication interactions
           - Discuss combined management strategies
           - Emphasize coordination of care

        5. For specific concerns:
           - Directly address stated worries
           - Provide relevant monitoring strategies
           - Suggest specific discussion points for healthcare provider

        Always maintain medical accuracy while being accessible and empathetic.
        """

    def generate_personalized_prompt(self, query: str, user_profile: Dict[str, str], profile_analysis: str) -> str:
        # Structure the medical context
        medical_context = {
            'age_group': 'elderly' if int(user_profile.get('age', 0)) >= 65 else 'adult',
            'has_diabetes': any(term in user_profile.get('diagnosis', '').lower() 
                              for term in ['diabetes', 'type 2', 't2dm']),
            'weight_management': any(term in user_profile.get('concern', '').lower() 
                                   for term in ['weight', 'obesity', 'bmi']),
            'blood_sugar': any(term in user_profile.get('target', '').lower() 
                             for term in ['glucose', 'sugar', 'a1c'])
        }
        
        # Generate condition-specific considerations
        specific_considerations = []
        if medical_context['age_group'] == 'elderly':
            specific_considerations.append("- Consider age-related factors for dosing and monitoring")
        if medical_context['has_diabetes']:
            specific_considerations.append("- Address diabetes management and blood sugar monitoring")
        if medical_context['weight_management']:
            specific_considerations.append("- Focus on weight management goals and expectations")
        
        return f"""
        COMPREHENSIVE PATIENT PROFILE
        ---------------------------
        Personal Information:
        - Name: {user_profile.get('name', 'Unknown')}
        - Age: {user_profile.get('age', 'Unknown')} ({medical_context['age_group']})
        - Location: {user_profile.get('location', 'Unknown')}

        Medical Context:
        - Diagnosis: {user_profile.get('diagnosis', 'Unknown')}
        - Primary Concern: {user_profile.get('concern', 'Unknown')}
        - Treatment Target: {user_profile.get('target', 'Unknown')}

        Medical Analysis Summary:
        {profile_analysis}

        Special Considerations:
        {chr(10).join(specific_considerations)}

        Current Query:
        "{query}"

        Please provide a personalized response that:
        1. Addresses {user_profile.get('name', 'the patient')} directly
        2. Considers their {user_profile.get('diagnosis', 'condition')}
        3. Aligns with their goal to {user_profile.get('target', 'improve health')}
        4. Accounts for their specific concern about {user_profile.get('concern', 'health management')}
        5. Includes age-appropriate recommendations for {medical_context['age_group']} patients
        6. Provides location-relevant information where applicable

        Format the response with clear sections for:
        - Personalized greeting and context acknowledgment
        - Direct answer to the query
        - Specific precautions based on their profile
        - Customized recommendations
        - Next steps and monitoring suggestions
        - Medical disclaimer
        """

   def stream_pplx_response(self, query: str, user_profile: Dict[str, str], profile_analysis: str) -> Generator[Dict[str, Any], None, None]:
        try:
            personalized_query = self.generate_personalized_prompt(query, user_profile, profile_analysis)
            
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {
                        "role": "system",
                        "content": self.pplx_system_prompt
                    },
                    {
                        "role": "user",
                        "content": personalized_query
                    }
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_message = f"PPLX API Error: {response.status_code} - {response.text}"
                st.error(error_message)
                yield {
                    "type": "error",
                    "message": error_message
                }
                return

            try:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                
                # Add medical disclaimer if not present
                if "disclaimer" not in content.lower():
                    content += "\n\nDisclaimer: This information is for educational purposes only and should not replace professional medical advice. Always consult your healthcare provider before making any changes to your medication or treatment plan."
                
                # Split content into chunks for streaming simulation
                chunk_size = 50
                chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
                
                accumulated_content = ""
                for chunk in chunks:
                    accumulated_content += chunk
                    yield {
                        "type": "content",
                        "data": chunk,
                        "accumulated": accumulated_content
                    }
                
                yield {
                    "type": "complete",
                    "content": content,
                    "sources": "Information provided by medical literature and FDA guidelines for GLP-1 medications."
                }
                
            except Exception as e:
                error_message = f"Error parsing PPLX response: {str(e)}"
                st.error(error_message)
                yield {
                    "type": "error",
                    "message": error_message
                }
                
        except Exception as e:
            error_message = f"Error communicating with PPLX: {str(e)}"
            st.error(error_message)
            yield {
                "type": "error",
                "message": error_message
            }

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
        .profile-section {
            background-color: #e8f5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            border-left: 4px solid #43a047;
        }
        .analysis-content {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
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
    if 'profile_analysis' not in st.session_state:
        st.session_state.profile_analysis = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'personal_info'

def display_profile_summary(profile_analysis: str):
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
            <p><strong>Medical Analysis:</strong></p>
            <div class="analysis-content">
                {analysis}
            </div>
        </div>
    """.format(
        **st.session_state.user_profile,
        analysis=profile_analysis.replace('\n', '<br>')
    ), unsafe_allow_html=True)

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

def main():
    st.set_page_config(
        page_title="Personalized GLP-1 Medical Assistant",
        page_icon="💊",
        layout="wide"
    )
    
    set_page_style()
    
    if not validate_api_keys():
        return
    
    openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    profile_manager = UserProfileManager(openai_client)
    profile_analyzer = ProfileAnalyzer(openai_client)
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
            display_profile_summary("")
            
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
                        # Generate profile analysis
                        with st.spinner("Analyzing your medical profile..."):
                            st.session_state.profile_analysis = profile_analyzer.analyze_profile(
                                st.session_state.user_profile
                            )
                        st.session_state.profile_complete = True
                        st.success("Profile completed! Analysis generated successfully.")
                        st.rerun()
                    else:
                        st.warning("Please provide all required medical information.")
    
    # GLP-1 Query Phase
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### Your Profile")
            display_profile_summary(st.session_state.profile_analysis)
            
            if st.button("Edit Profile"):
                st.session_state.profile_complete = False
                st.session_state.profile_analysis = None
                st.session_state.current_step = 'personal_info'
                st.rerun()
        
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
                
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b>Your Question:</b><br>{user_query}
                </div>
                """, unsafe_allow_html=True)
                
                response_placeholder = st.empty()
                sources_placeholder = st.empty()
                
                for chunk in glp1_bot.stream_pplx_response(
                    query=user_query,
                    user_profile=st.session_state.user_profile,
                    profile_analysis=st.session_state.profile_analysis
                ):
                    if chunk["type"] == "error":
                        st.error(chunk["message"])
                        break
                        
                    elif chunk["type"] == "content":
                        response_placeholder.markdown(f"""
                        <div class="chat-message bot-message">
                            <div class="category-tag">{query_category.upper()}</div>
                            {chunk["accumulated"]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif chunk["type"] == "complete":
                        # Add response to chat history
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "response": chunk["content"],
                            "category": query_category,
                            "sources": chunk["sources"]
                        })
                        
                        sources_placeholder.markdown(f"""
                        <div class="sources-section">
                            <b>Sources:</b><br>
                            {chunk["sources"]}
                        </div>
                        """, unsafe_allow_html=True)
            
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
                        <div class="sources-section">
                            <b>Sources:</b><br>
                            {chat['sources']}
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")
