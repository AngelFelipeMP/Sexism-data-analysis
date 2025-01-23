import os
from openai import OpenAI, AzureOpenAI
import logging
from icecream import ic
# from config import (
from config_fill_unaswered_calls import (
    API_PROVIDER, 
    AZURE_ENDPOINT,
    AZURE_API_VERSION
)
# Configure the logger
logging.basicConfig(filename='summarizer_unanswered_calls.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

if API_PROVIDER == 'AZURE':
    client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=os.environ.get('AZURE_OPENAI_KEY'))
else:
    client = OpenAI(api_key = os.environ.get('OPENAI_KEY'))
    print("OpenAI client created")

class OpenAIAPI:
    def __init__(self):
        # self.openai_llms = ["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o-2024-05-13"]
        self.openai_llms = ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06"]
        
    def get_completion(self, llm, temperature=0, max_tokens=10):
        try:
            response = client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": self.llm_role, "content": self.llm_persona},
                    {"role": self.user_role, "content": self.prompt}],
                temperature=temperature,  # this is the degree of randomness of the model's output
                max_tokens=max_tokens,
                top_p=1,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"An error occurred during the OpenAI API call: {e}")
            return None
        
class MistralAPI:
    def __init__(self):
        self.mistral_llms = []
        
    def get_completion(self, llm, temperature, max_tokens):
        pass
        

class llmsAPI(OpenAIAPI, MistralAPI):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        self.user_role = "user"
        self.llm_role = "system"
        self.llm_persona = "You are an expert on understanding sexism in social media texts."
        
    def get_completation(self, llm, temperature, max_tokens):
        if llm in self.openai_llms:
            return OpenAIAPI.get_completion(llm, temperature, max_tokens)
        elif llm in self.mistral_llms:
            return MistralAPI.get_completion(llm, temperature, max_tokens)
        else:
            raise ValueError(f"The llm {llm} is not available.")