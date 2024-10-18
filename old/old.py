import pandas as pd
from config import *
import ast
import json

# Global distribution of the data
df = pd.read_csv(DATA_PATH + '/EXIST2023_training-dev.csv')
# print(df['hard_label_task1'].value_counts())

dict_label_task1 = {'YES': 0, 'NO': 0}
for label_list in df['labels_task1'].tolist():
    for label in ast.literal_eval(label_list):
        dict_label_task1[label] += 1
# print(dict_label_task1)


dict_label_gender_task1 = {'MALE':{'YES': 0, 'NO': 0}, 'FEMALE':{'YES': 0, 'NO': 0}}
for label_list in df['labels_task1'].tolist():
    for label in ast.literal_eval(label_list)[:3]:
        dict_label_gender_task1['FEMALE'][label] += 1
    
    for label in ast.literal_eval(label_list)[3:]:
        dict_label_gender_task1['MALE'][label] += 1   
# print(dict_label_gender_task1)


dict_label_age_task1 = {'18-22':{'YES': 0, 'NO': 0}, '23-45':{'YES': 0, 'NO': 0}, '46+':{'YES': 0, 'NO': 0}}
for label_list in df['labels_task1'].tolist():
    for label in ast.literal_eval(label_list)[:1] + ast.literal_eval(label_list)[3:4]:
        dict_label_age_task1['18-22'][label] += 1
    
    for label in ast.literal_eval(label_list)[1:2] + ast.literal_eval(label_list)[4:5]:
        dict_label_age_task1['23-45'][label] += 1   
        
    for label in ast.literal_eval(label_list)[2:3] + ast.literal_eval(label_list)[5:6]:
        dict_label_age_task1['46+'][label] += 1   
        
# print(dict_label_age_task1)

if __name__ == "__main__":
