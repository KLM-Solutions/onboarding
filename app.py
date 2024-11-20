
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
            Extract personal information and return ONLY a JSON object with this exact structure:
            {
                "name": "patient name",
                "age": "numeric age",
                "location": "patient location"
            }
            Only include explicitly stated information. Leave fields empty if not mentioned.
            """,
            
            "medical_info": """
            Extract medical information and return ONLY a JSON object with this exact structure:
            {
                "diagnosis": "patient diagnosis",
                "concern": "primary medical concern",
                "target": "treatment target or goal"
            }
            Only include explicitly stated information. Leave fields empty if not mentioned.
            """
        }

    def process_user_input(self, user_input: str, info_type: str) -> Dict[str, str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_instructions[info_type]},
                    {"role": "user", "content": user_input}
                ],
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            return {}

class ProfileAnalyzer:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.analysis_prompt = """
        Analyze the patient profile and return ONLY a JSON object with this exact structure:
        {
            "risk_factors": {
                "age_related": ["list of age-related considerations"],
                "diagnosis_related": ["list of diagnosis-specific concerns"],
                "contraindications": ["list of potential contraindications"]
            },
            "treatment_context": {
                "glp1_relevance": "relevance to condition",
                "monitoring_needs": ["list of monitoring considerations"],
                "lifestyle_factors": ["list of lifestyle factors"]
            },
            "special_considerations": {
                "drug_interactions": ["list of potential drug interactions"],
                "precautions": ["list of specific precautions"],
                "priority_targets": ["list of health targets"]
            }
        }
        """

    def analyze_profile(self, profile: Dict[str, str]) -> Dict[str, Any]:
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
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a medical profile analyzer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing profile: {str(e)}")
            return {}

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
        Return ONLY a JSON object with this exact structure:
        {
            "query_info": {
                "category": "dosage|side_effects|benefits|interactions|lifestyle|general",
                "question": "original question"
            },
            "response": {
                "direct_answer": "clear answer to the query",
                "medical_relevance": "relevance to patient condition",
                "precautions": ["list of relevant warnings"],
                "recommendations": ["list of personalized recommendations"],
                "next_steps": ["list of suggested actions"]
            },
            "metadata": {
                "sources": ["relevant medical guidelines"],
                "confidence_level": "high|medium|low",
                "disclaimer": "medical disclaimer"
            }
        }
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
                yield {
                    "status": "error",
                    "error": {
                        "type": "api_error",
                        "code": response.status_code,
                        "message": "API request failed"
                    }
                }
                return

            try:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                response_json = json.loads(content)
                
                yield {
                    "status": "success",
                    "data": response_json
                }
                
            except json.JSONDecodeError as e:
                yield {
                    "status": "error",
                    "error": {
                        "type": "parsing_error",
                        "message": str(e),
                        "raw_response": content
                    }
                }
                
        except Exception as e:
            yield {
                "status": "error",
                "error": {
                    "type": "general_error",
                    "message": str(e)
                }
            }

    def generate_personalized_prompt(self, query: str, user_profile: Dict[str, str], profile_analysis: Dict[str, Any]) -> str:
        return json.dumps({
            "patient_profile": user_profile,
            "medical_analysis": profile_analysis,
            "query": query
        })

# Initialize session state
if 'profile_complete' not in st.session_state:
    st.session_state.profile_complete = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': '', 'age': '', 'location': '',
        'diagnosis': '', 'concern': '', 'target': ''
    }
if 'profile_analysis' not in st.session_state:
    st.session_state.profile_analysis = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'personal_info'

def main():
    st.title("GLP-1 Medication Assistant")
    
    if 'OPENAI_API_KEY' not in st.secrets or 'PPLX_API_KEY' not in st.secrets:
        st.error(json.dumps({"error": "Missing API keys"}))
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
                st.session_state.user_profile.update(info)
                st.json(info)  # Display JSON response
                if all(st.session_state.user_profile[f] for f in ['name', 'age', 'location']):
                    st.session_state.current_step = 'medical_info'
                    st.rerun()
        
        else:  # medical_info step
            st.json(st.session_state.user_profile)  # Display current profile as JSON
            medical_info = st.text_input("Enter diagnosis, concern, and treatment target:")
            if st.button("Complete") and medical_info:
                info = profile_manager.process_user_input(medical_info, "medical_info")
                st.session_state.user_profile.update(info)
                st.json(info)  # Display JSON response
                if all(st.session_state.user_profile[f] for f in ['diagnosis', 'concern', 'target']):
                    profile_analysis = profile_analyzer.analyze_profile(st.session_state.user_profile)
                    st.session_state.profile_analysis = profile_analysis
                    st.json(profile_analysis)  # Display analysis as JSON
                    st.session_state.profile_complete = True
                    st.rerun()
    
    else:
        st.json(st.session_state.user_profile)  # Display profile as JSON
        st.json(st.session_state.profile_analysis)  # Display analysis as JSON
        
        query = st.text_input("Ask about GLP-1 medications:")
        if st.button("Submit") and query:
            for response in glp1_bot.stream_pplx_response(
                query=query,
                user_profile=st.session_state.user_profile,
                profile_analysis=st.session_state.profile_analysis
            ):
                st.json(response)  # Display all responses as JSON
                if response["status"] == "success":
                    st.session_state.chat_history.append({
                        "timestamp": str(datetime.datetime.now()),
                        "query": query,
                        "response": response["data"]
                    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(json.dumps({"error": str(e)}))
