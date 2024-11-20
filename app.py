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
        # Updated system prompt to enforce JSON response format
        self.pplx_system_prompt = """
        You are a specialized medical information assistant providing GLP-1 medication information.
        
        You MUST format ALL responses as JSON with the following structure:
        {
            "greeting": "Personalized greeting using patient's name",
            "response": {
                "direct_answer": "Clear, concise answer to the query",
                "medical_context": "How this applies to the patient's specific condition",
                "precautions": "Relevant warnings based on patient profile",
                "recommendations": "Personalized recommendations",
                "next_steps": "Suggested actions and monitoring"
            },
            "sources": "Reference to medical guidelines or literature",
            "disclaimer": "Standard medical disclaimer"
        }
        
        Keep all responses factual and tailored to the patient's profile.
        """

    def generate_personalized_prompt(self, query: str, user_profile: Dict[str, str], profile_analysis: str) -> str:
        return f"""
        Patient Profile:
        Name: {user_profile.get('name')}
        Age: {user_profile.get('age')}
        Location: {user_profile.get('location')}
        Diagnosis: {user_profile.get('diagnosis')}
        Concern: {user_profile.get('concern')}
        Target: {user_profile.get('target')}

        Analysis: {profile_analysis}

        Query: {query}

        Provide a response in the required JSON format, ensuring all fields are filled appropriately.
        """

    def stream_pplx_response(self, query: str, user_profile: Dict[str, str], profile_analysis: str) -> Generator[Dict[str, Any], None, None]:
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
                yield {"type": "error", "message": f"API Error: {response.status_code}"}
                return

            try:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
                
                # Parse the response as JSON
                response_json = json.loads(content)
                
                # Ensure disclaimer is present
                if "disclaimer" not in response_json:
                    response_json["disclaimer"] = "This information is for educational purposes only and should not replace professional medical advice. Always consult your healthcare provider before making any changes to your medication or treatment plan."
                
                yield {
                    "type": "complete",
                    "content": response_json,
                    "raw_response": content
                }
                
            except json.JSONDecodeError as e:
                yield {
                    "type": "error", 
                    "message": f"Error parsing JSON response: {str(e)}",
                    "raw_response": content
                }
                
        except Exception as e:
            yield {"type": "error", "message": f"Error: {str(e)}"}

def main():
    st.title("GLP-1 Medication Assistant")
    
    if 'OPENAI_API_KEY' not in st.secrets or 'PPLX_API_KEY' not in st.secrets:
        st.error("Missing API keys")
        return
    
    initialize_session_state()
    
    openai_client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    profile_manager = UserProfileManager(openai_client)
    profile_analyzer = ProfileAnalyzer(openai_client)
    glp1_bot = GLP1Bot(st.secrets['PPLX_API_KEY'])
    
    if not st.session_state.profile_complete:
        # [Previous profile collection code remains the same]
        pass
    else:
        st.write("Profile:", st.session_state.user_profile)
        st.write("Analysis:", st.session_state.profile_analysis)
        
        query = st.text_input("Ask about GLP-1 medications:")
        if st.button("Submit") and query:
            for response in glp1_bot.stream_pplx_response(
                query=query,
                user_profile=st.session_state.user_profile,
                profile_analysis=st.session_state.profile_analysis
            ):
                if response["type"] == "error":
                    st.error(response["message"])
                    if "raw_response" in response:
                        st.write("Raw response:", response["raw_response"])
                elif response["type"] == "complete":
                    json_response = response["content"]
                    
                    # Display formatted JSON response
                    st.json(json_response)
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": json_response,
                        "raw_response": response.get("raw_response", "")
                    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {str(e)}")
