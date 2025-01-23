import os

REPO_PATH = os.getcwd()
DATA = 'EXIST2023'
DATA_PATH = REPO_PATH + '/data'
PACKAGE_PATH = REPO_PATH + '/' + DATA
LABEL_GOLD_PATH = PACKAGE_PATH + '/evaluation/golds'
ANALYSES_PATH = REPO_PATH + '/analyzes'
PREDICTIONS_PATH = REPO_PATH + '/predictions'
JSON_PREDICTIONS_PATH = REPO_PATH + '/json_predictions'

# S3_BUCKET = '/s3-bucket'
##Deprecated
# PATH_SORCE_PACKAGE = REPO_PATH + S3_BUCKET + '/' + DATA + '_package'
PATH_SORCE_PACKAGE = '/'.join(REPO_PATH.split('/')[:-1]) + '/' + DATA

CATEGORIES_TASK1 = ['YES', 'NO']
CATEGORIES_TASK2 = ['DIRECT', 'REPORTED', 'JUDGEMENTAL']
CATEGORIES_TASK3 = ['IDEOLOGICAL AND INEQUALITY', 
                    'STEREOTYPING AND DOMINANCE', 
                    'OBJECTIFICATION', 
                    'SEXUAL VIOLENCE', 
                    'MISOGYNY AND NON-SEXUAL VIOLENCE']

AGE_GROUPS = ['18-22', '23-45', '46+']
GENDER_GROUPS = ['female', 'male']

LLMS = ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06"]
PROMPTS = ["ZeroShotTask1", "ZeroShotTask2", "ZeroShotTask3"]

API_KEY = 'OPENAI_KEY' #'AZURE_OPENAI_KEY'

API_PROVIDER = 'OPENAI' #'AZURE'

# AZURE_ENDPOINT = 'https://angelsachinopenai.openai.azure.com/'
AZURE_ENDPOINT = 'https://angel-m4up853w-eastus2.cognitiveservices.azure.com/'
AZURE_API_VERSION = '2024-08-01-preview'