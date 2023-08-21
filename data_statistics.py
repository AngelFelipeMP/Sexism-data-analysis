import pandas as pd
from config import *
import ast
import json
from scipy import stats

class DataStatistics():
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
    
    
    def t_test(self):
        
        print(self.gender_annotetors[:5])
        print('/n')
        print(self.labels_task1[:5])
        # return stats.ttest_ind(task1, task2) 
            

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataStatistics(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    data_exploration.t_test()