import os

REPO_PATH = os.getcwd()
DATA = 'EXIST2023'
DATA_PATH = REPO_PATH + '/data'
PACKAGE_PATH = REPO_PATH + '/' + DATA
LABEL_GOLD_PATH = PACKAGE_PATH + '/evaluation/golds'
ANALYSES_PATH = REPO_PATH + '/analyzes'

S3_BUCKET = '/s3-bucket'
##Deprecated
# PATH_SORCE_PACKAGE = REPO_PATH + S3_BUCKET + '/' + DATA + '_package'
PATH_SORCE_PACKAGE = '/'.join(REPO_PATH.split('/')[:-1]) + '/' + DATA