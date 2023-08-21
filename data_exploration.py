import pandas as pd
from config import *
import ast
import json

class DataExploration():
    def __init__(self, data_path, file_name, analyses_path):
        self.data_path = data_path
        self.analyses_path = analyses_path
        self.df = pd.read_csv(self.data_path + '/' + file_name)
        self.resources()
        
    def __len__(self):
        return self.df.size
    
    def __getitem__(self, item):
        return self.df.loc[item,['gender_annotators', 'age_annotators', 'labels_task1']]
        
    def resources(self):  
        self.gender_annotetors = self.string_to_list(self.df['gender_annotators'])
        self.age_annotetors = self.string_to_list(self.df['age_annotators'])
        
        self.labels_task1 = self.string_to_list(self.df['labels_task1'])
        self.labels_task2 = self.string_to_list(self.df['labels_task2'])
        self.labels_task3 = self.string_to_list(self.df['labels_task3'])
        
        self.age_groups = self.get_possible_items(self.age_annotetors)
        
        self.possible_labels_task1 = self.get_possible_items(self.labels_task1)
        self.possible_labels_task2 = self.get_possible_items(self.labels_task2)
        self.possible_labels_task3 = self.get_possible_items(self.labels_task3)
        
    def string_to_list(self, string_list):
        return [ast.literal_eval(string) for string in string_list.tolist()]
    
    def get_possible_items(self, data_tems):
        possible_tems = set()
        for tem_list in data_tems:
            for tem in tem_list:
                
                if type(tem) == list:
                    for sub_tem in tem:
                        possible_tems.add(sub_tem)
                else:
                    possible_tems.add(tem)
                    
        return possible_tems
    
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
        
        if task == 'task1' or task == 'task2':
            self.overall_task_dict['whole_data']['gold_label'] = self.df['hard_label_' + task ].value_counts().to_dict()
        else:
            self.overall_task_dict['whole_data']['gold_label'] = self.df['hard_label_' + task ].apply(ast.literal_eval).explode().value_counts().to_dict()
        
        for gender in self.task_distribution.keys():
            for age_interval in self.task_distribution[gender].keys():
                for category, value in self.task_distribution[gender][age_interval].items():
                    
                    self.overall_task_dict['age'][age_interval][category] += value
                    self.overall_task_dict['gender'][gender][category] += value
                    self.overall_task_dict['whole_data']['LWD'][category] += value
                        
        return self.overall_task_dict
    
    
    def tasks_distribution(self):
        tasks_tuple = [('task1', self.labels_task1, self.possible_labels_task1), 
                    ('task2', self.labels_task2, self.possible_labels_task2), 
                    ('task3', self.labels_task3, self.possible_labels_task3)]

        for task, task_labels, possible_labels in tasks_tuple:
            self.get_distribution(possible_labels, task_labels)
            self.get_overall_distribution(possible_labels, task)
            
            with open(self.analyses_path + '/' + task + '_distribution.json', 'w') as file:
                json.dump(self.task_distribution, file, indent=4)
            
            with open(self.analyses_path + '/' + task + '_overall_distribution.json', 'w') as file:
                json.dump(self.overall_task_dict, file, indent=4)
            
            print(f'############## {task} distributions ##############')
            print('#### Overall:')
            print(json.dumps(self.task_distribution, indent=4))
            print('\n')
            print('#### Fina grained:')
            print(json.dumps(self.overall_task_dict, indent=4))
            print('\n')
            

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataExploration(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    data_exploration.tasks_distribution()
    
    
##TODO: REVIEW CODE [X]
##TODO: REVIEW VARIABLES NAMES AND FUNC NAME [X]
##TODO: MAKE SURA THE CODE MAKE SENCE AND IT HAS THE MINIMAL NUMBER OF LINES POSSIBLE [X]
##TODO: CREATE TABLES FOR THE DISTRIBUTION 
##TODO: CHECK THE TABLES BY SUM VALIUS
##TODO: CREATE A TABLE FOR THE OVERALL DISTRIBUTION
##TODO: THINK ABOUT TO TRAIN MODELS FOR EACH SUB-GROUP BECAUIS THEIR FUNCTIBS ARE DIFFERENT
##TODO: CRATE GRAFICS FOR THE DISTRIBUTION