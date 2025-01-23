from openai import AzureOpenAI
import os

API_KEY = os.environ.get('AZURE_OPENAI_KEY')
AZURE_ENDPOINT = 'https://angel-m4up853w-eastus2.cognitiveservices.azure.com/'
DEPLOYMENT_ID  = 'gpt-4o-2024-08-06'
AZURE_API_VERSION = '2024-08-01-preview'

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
)

completion = client.chat.completions.create(
    model=DEPLOYMENT_ID,  # e.g. gpt-35-instant
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)

print(completion.to_json())