import os

REPO_PATH = os.getcwd()
DATA = 'EXIST2023'
DATA_PATH = REPO_PATH + '/data'
PACKAGE_PATH = REPO_PATH + '/' + DATA
LABEL_GOLD_PATH = PACKAGE_PATH + '/evaluation/golds'
ANALYSES_PATH = REPO_PATH + '/analyzes'
PREDICTIONS_PATH = REPO_PATH + '/predictions'

S3_BUCKET = '/s3-bucket'
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