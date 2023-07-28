import os

REPO_PATH = os.getcwd()
DATA = 'EXIST2023'
DATA_PATH = REPO_PATH + '/data'
PACKAGE_PATH = REPO_PATH + '/' + DATA
LABEL_GOLD_PATH = PACKAGE_PATH + '/evaluation/golds'

S3_BUCKET = '/s3-bucket'
PATH_SORCE_PACKAGE = REPO_PATH + S3_BUCKET + '/' + DATA + '_package'