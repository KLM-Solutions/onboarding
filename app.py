
import streamlit as st
import requests
import json
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
                ],
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

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

        Return the analysis in JSON format with these exact sections.
        """

    def analyze_profile(self, profile: Dict[str, str]) -> Dict:
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
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

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

        Always provide source which is related to the generated response.
        Provide response in a simple manner that is easy to understand at preferably a 11th grade literacy level with reduced pharmaceutical or medical jargon
        Always Return sources in a hyperlink format
        Always maintain medical accuracy while being accessible and empathetic.

        Format all responses as JSON with appropriate sections matching the above structure.
        """

    def stream_pplx_response(self, query: str, user_profile: Dict[str, str], profile_analysis: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        try:
            prompt = self.generate_personalized_prompt(query, user_profile, profile_analysis)
            
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            
            if response.status_code != 200:
                yield {"status": "error", "message": f"API Error: {response.status_code}"}
                return

            try:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                response_json = json.loads(content)
                
                yield {"status": "success", "data": response_json}
                
            except json.JSONDecodeError as e:
                yield {"status": "error", "message": f"Parsing error: {str(e)}", "raw_content": content}
                
        except Exception as e:
            yield {"status": "error", "message": str(e)}

    def generate_personalized_prompt(self, query: str, user_profile: Dict[str, str], profile_analysis: Dict[str, Any]) -> str:
        return f"""
        Patient Profile:
        Name: {user_profile.get('name')}
        Age: {user_profile.get('age')}
        Location: {user_profile.get('location')}
        Diagnosis: {user_profile.get('diagnosis')}
        Concern: {user_profile.get('concern')}
        Target: {user_profile.get('target')}

        Analysis: {json.dumps(profile_analysis)}

        Query: {query}

        Provide a response in the required JSON format, ensuring all fields are filled appropriately.
        Remember to maintain simple language at an 11th grade reading level.
        Include hyperlinked sources related to the response.
        """

# Initialize session state
if 'profile_complete' not in st.session_state:
    st.session_state.profile_complete = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': '', 'age': '', 'location': '',
        'diagnosis': '', 'concern': '', 'target': ''
    }
if 'profile_analysis' not in st.session_state:
    st.session_state.profile_analysis = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'personal_info'

def main():
    st.title("GLP-1 Medication Assistant")
    
    if 'OPENAI_API_KEY' not in st.secrets or 'PPLX_API_KEY' not in st.secrets:
        st.json({"error": "Missing API keys"})
        return
    
    openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    profile_manager = UserProfileManager(openai_client)
    profile_analyzer = ProfileAnalyzer(openai_client)
    glp1_bot = GLP1Bot(st.secrets['PPLX_API_KEY'])
    
    if not st.session_state.profile_complete:
        if st.session_state.current_step == 'personal_info':
            personal_info = st.text_input("Enter name, age, and location:")
            if st.button("Next") and personal_info:
                info = profile_manager.process_user_input(personal_info, "personal_info")
                st.json(info)
                st.session_state.user_profile.update(info)
                if all(st.session_state.user_profile[f] for f in ['name', 'age', 'location']):
                    st.session_state.current_step = 'medical_info'
                    st.rerun()
        
        else:  # medical_info step
            st.json(st.session_state.user_profile)
            medical_info = st.text_input("Enter diagnosis, concern, and treatment target:")
            if st.button("Complete") and medical_info:
                info = profile_manager.process_user_input(medical_info, "medical_info")
                st.json(info)
                st.session_state.user_profile.update(info)
                if all(st.session_state.user_profile[f] for f in ['diagnosis', 'concern', 'target']):
                    analysis = profile_analyzer.analyze_profile(st.session_state.user_profile)
                    st.json(analysis)
                    st.session_state.profile_analysis = analysis
                    st.session_state.profile_complete = True
                    st.rerun()
    
    else:
        st.json({"profile": st.session_state.user_profile, "analysis": st.session_state.profile_analysis})
        
        query = st.text_input("Ask about GLP-1 medications:")
        if st.button("Submit") and query:
            for response in glp1_bot.stream_pplx_response(
                query=query,
                user_profile=st.session_state.user_profile,
                profile_analysis=st.session_state.profile_analysis
            ):
                st.json(response)
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": response["data"]
                    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.json({"error": str(e)})
