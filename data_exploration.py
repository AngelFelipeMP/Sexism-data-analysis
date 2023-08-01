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

##########################################################################################################S

class DataExploration():
    def __init__(self, data_path, file_name, analyses_path):
        self.data_path = data_path
        self.analyses_path = analyses_path
        self.df = pd.read_csv(self.data_path + '/' + file_name)
        
        self.gender_annotetors = self.string_to_list(self.df['gender_annotators'])
        self.age_annotetors = self.string_to_list(self.df['age_annotators'])
        
        self.labels_task1 = self.string_to_list(self.df['labels_task1'])
        self.labels_task2 = self.string_to_list(self.df['labels_task2'])
        self.labels_task3 = self.string_to_list(self.df['labels_task3'])
        
        self.age_groups = self.get_possible_items(self.age_annotetors)
        
        self.possible_labels_task1 = self.get_possible_items(self.labels_task1)
        self.possible_labels_task2 = self.get_possible_items(self.labels_task2)
        self.possible_labels_task3 = self.get_possible_items(self.labels_task3)
        
    def __len__(self):
        return self.df.size
    
    def __getitem__(self, item):
        return self.df.loc[item,['gender_annotators', 'age_annotators', 'labels_task1']]
    
    def string_to_list(self, string_list):
        return [ast.literal_eval(string) for string in string_list.tolist()]
    
    def get_possible_items(self, data_labels):
        possible_labels = set()
        for label_list in data_labels:
            for label in label_list:
                
                if type(label) == list:
                    for sub_label in label:
                        possible_labels.add(sub_label)
                else:
                    possible_labels.add(label)
                    
        return possible_labels
    
    def get_distribution(self, possible_labels, task_labels):
        self.task_distribution = {gender:{age:{label:0 for label in possible_labels} for age in self.age_groups} for gender in ['F', 'M']}
        
        for gender_list, age_list, label_list in zip(self.gender_annotetors, self.age_annotetors, task_labels):
            for gender, age, label in zip(gender_list, age_list, label_list):
                
                if type(label) == list:
                    for sub_label in label:
                        self.task_distribution[gender][age][sub_label] += 1
                else:
                    self.task_distribution[gender][age][label] += 1
                    
        return self.task_distribution
    
    
    def get_overall_distribution(self, possible_labels, task):
        self.overall_task_dict = {'whole_data':{group:{label:0 for label in possible_labels} for group in ['gold_label', 'LWD']} ,
                    'gender':{gender:{label:0 for label in possible_labels} for gender in ['F', 'M']},
                    'age':{age:{label:0 for label in possible_labels} for age in self.age_groups}}
        
        self.overall_task_dict['whole_data']['gold_label'] = dict(self.df['hard_label_' + task].explode().value_counts())
        
        for gender in self.task_distribution.keys():
            for age in self.task_distribution[gender].keys():
                for age_interval, value in self.task_distribution[gender][age].items():
                    
                        self.overall_task_dict['age'][age_interval] += value
                        self.overall_task_dict['gender'][gender] += value
                        self.overall_task_dict['whole_data']['LWD'] += value
                        
        return self.overall_task_dict
    
    
    def tasks_distribution(self):
        tasks_tuple = [('task1', self.labels_task1, self.possible_labels_task1), 
                    ('task2', self.labels_task2, self.possible_labels_task2), 
                    ('task3', self.labels_task3, self.possible_labels_task3)]

        for task, task_labels, possible_labels in tasks_tuple:
            task_dict = self.get_distribution(possible_labels, task_labels)
            task_overall_dict = self.get_overall_distribution(possible_labels, task)
            
            with open(self.analyses_path + '/' + task + '_distribution.json', 'w') as file:
                json.dump(task_dict, file, indent=4)
                
            with open(self.analyses_path + '/' + task + 'overall_distribution.json', 'w') as file:
                json.dump(task_overall_dict, file, indent=4)
            
            print(f'{task} distributions:')
            print('Overall:')
            print(pd.DataFrame.from_dict(task_overall_dict,orient='columns'))
            print('\n')
            print('Fina grained:')
            print(pd.DataFrame.from_dict(task_dict,orient='columns'))
            print('\n')
            

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataExploration(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    data_exploration.tasks_distribution()
    
    
    ######### STOP HERE !!!! DEBUG !! ##########