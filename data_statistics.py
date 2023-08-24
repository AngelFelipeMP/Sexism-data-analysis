import pandas as pd
from config import *
import ast
import json
from scipy.stats import f_oneway, ttest_ind, tukey_hsd
import numpy as np
import itertools

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
    
    def mean_distribution_task1(self):
        anotations = {}
        anotations['gender'] = {g:[] for g in ['F', 'M']}
        anotations['age'] = {a:[] for a in self.age_groups}
        anotations['gender_age'] = {g + '_' + a:[] for g in ['F', 'M'] for a in self.age_groups}
        
        num_annotators = {'gender':3, 'age':2, 'gender_age':1}
        for set_gender, set_age, set_annotation in zip(self.gender_annotetors, self.age_annotetors, self.labels_task1):
            for gender, age, annotation in zip(set_gender, set_age, set_annotation):
                anotations['gender'][gender].append(1 if annotation=='YES' else 0)
                anotations['age'][age].append(1 if annotation=='YES' else 0)
                anotations['gender_age'][gender + '_' + age].append(1 if annotation=='YES' else 0)
                    
            for set_annotation in anotations.keys():
                for key in anotations[set_annotation].keys():
                    anotations[set_annotation][key] = anotations[set_annotation][key][:-num_annotators[set_annotation]] + [sum(anotations[set_annotation][key][-num_annotators[set_annotation]:]) / len(anotations[set_annotation][key][-num_annotators[set_annotation]:])]
                    
        return anotations
    
    
    def anova_test(self):
        print('###### ANOVA ######')
        dist_task1 = self.mean_distribution_task1()
        # dist_task1 = [v for key in dist_task1.keys() for v in dist_task1[key].values()]
        
        for group, labels in dist_task1.items():
            print('Group: ', group)
            dist_group = [l for l in labels.values()]
            print(f_oneway(*dist_group))


    def tukey_hsd_test(self):
        print('###### Tukey HSD ######')
        dist_task1 = self.mean_distribution_task1()
        # dist_task1 = [(k,v) for key in dist_task1.keys() for k,v in dist_task1[key].items()]
        
        for group, labels in dist_task1.items():
            print('Group: ', group)
            
            dist_group = list(labels.items())
            dist_group.sort(key=lambda x: x[0])
            dist_group_names = [k for k,_ in dist_group]
            dist_group_values = [v for _,v in dist_group]
            
            # for i in range(len(dist_group)):
            #     print(dist_group_names[i], ': ', i)
            # print(tukey_hsd(*dist_group_values))
            
            for i in range(len(dist_group)):
                print(dist_group_names[i], ': ', i)
            rest = tukey_hsd(*dist_group_values)
            print('pvalues: ', rest.pvalue)

    def t_test(self):
        print('###### T-test ######')
        scores = {'M':[], 'F':[]}
        num_annotators = 3
        for set_gender, set_annotation in zip(self.gender_annotetors, self.labels_task1):
            for gender, annotation in zip(set_gender, set_annotation):
                scores[gender].append(1 if annotation=='YES' else 0)

            for g in scores.keys():
                scores[g] = scores[g][:-num_annotators] + [sum(scores[g][-num_annotators:]) / len(scores[g][-num_annotators:])]


        t, p = ttest_ind(scores['M'], scores['F'])
        print('Group: Gender')
        print('p: ', p)

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataStatistics(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    # data_exploration.mean_distribution()
    data_exploration.t_test()
    print('\n')
    data_exploration.anova_test()
    print('\n')
    data_exploration.tukey_hsd_test()
    
    #DOTO Check the sequence of teste (folow my annotation)
    #1) caculate anova and tukey for age groups [X]
    #2) caculate between mixed groups [X]
    #3) fix code to retrive all the necessary data [ ]
    #4) function to plot the matrix/table with the colors[ ]
    
    
    
    