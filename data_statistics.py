import pandas as pd
import glob
from config import *
from scipy.stats import f_oneway, ttest_ind, tukey_hsd
import numpy as np
from data_exploration import DataExploration
import matplotlib.pyplot as plt
import ast
from icecream import ic
import json
import krippendorff
from itertools import combinations_with_replacement
from statistics import mean
import random
from tqdm import tqdm

class DataStatistics(DataExploration):
    def __init__(self, data_path, file_name, analyses_path):
        super().__init__(data_path, file_name, analyses_path)
        self.num_annotators = {'gender':3, 'age':2, 'gender_age':1}
        self.create_columns_by_class()
        self.df['labels_task1'] = self.df['labels_task1'].apply(lambda x: ast.literal_eval(x))
        
    def create_columns_by_class(self):
        for (prefix, categoty) in [('labels_task2', self.possible_labels_task2), ('labels_task3', self.possible_labels_task3)]:
            for label in categoty:
                if 'task2' in prefix:
                    self.df[prefix + '_' + label] = self.df[prefix].apply(lambda x: ['YES' if label == l else 'NO' for l in ast.literal_eval(x)])
                else:
                    self.df[prefix + '_' + label] = self.df[prefix].apply(lambda x: ['YES' if label in l else 'NO' for l in ast.literal_eval(x)])
    
    def average_last_n_values(self, dict_scores, n):
        for key in dict_scores.keys():
            dict_scores[key] = dict_scores[key][:-n] + [sum(dict_scores[key][-n:]) / len(dict_scores[key][-n:])]  
        return dict_scores
    
    def mean_distributions(self, labels_task_n):
        anotations = {}
        anotations['gender'] = {g:[] for g in ['F', 'M']} 
        anotations['age'] = {a:[] for a in self.age_groups}
        anotations['gender_age'] = {g + '_' + a:[] for g in ['F', 'M'] for a in self.age_groups}
    
        for set_gender, set_age, set_annotation in zip(self.gender_annotetors, self.age_annotetors, labels_task_n):
            for gender, age, annotation in zip(set_gender, set_age, set_annotation):
                anotations['gender'][gender].append(1 if annotation=='YES' else 0)
                anotations['age'][age].append(1 if annotation=='YES' else 0)
                anotations['gender_age'][gender + '_' + age].append(1 if annotation=='YES' else 0)
                    
            for set_annotation in anotations.keys():
                anotations[set_annotation] = self.average_last_n_values(anotations[set_annotation], self.num_annotators[set_annotation])
                
        return anotations
    
    def mean_value(self, dist_task_n, task, categoty):
        print('******* Mean *******')
        for group, labels in dist_task_n.items():
            print('Group: ', group)
            for label, values in labels.items():
                print(label, np.mean(values))
                
                dist_task_n[group][label] = np.mean(values)
            
            dist_task_n[group] = {k: dist_task_n[group][k] for k in sorted(dist_task_n[group].keys())}   
            
            print('\n')
            df = pd.DataFrame.from_dict({k: [v] for k, v in dist_task_n[group].items()})
            df.to_csv(self.analyses_path + '/means_'+ task + '_' + categoty + '_' + group +'.csv', index=True)
    
    
    def anova_test(self, dist_task_n):
        print('###### ANOVA ######')
    
        for group, labels in dist_task_n.items():
            print('Group: ', group)
            rest = f_oneway(*[l for l in labels.values()])
            print('p: ',rest.pvalue)

    def tukey_hsd_test(self, dist_task_n, task, categoty):
        print('###### Tukey HSD ######')
        
        for group, labels in dist_task_n.items():
            print('Group: ', group)
            
            dist_group = list(labels.items())
            dist_group.sort(key=lambda x: x[0])
            dist_group_names = [k for k,_ in dist_group]
            dist_group_values = [v for _,v in dist_group]
            
            rest = tukey_hsd(*dist_group_values)
            df = self.array_df(rest.pvalue, dist_group_names, dist_group_names)
            df.to_csv(self.analyses_path + '/tukey_'+ task + '_' + categoty + '_' + group +'.csv', index=True)
            print(df)
            
            
    def Krippendorff_alpha(self, dist_task_n, task, categoty):
        print('###### Krippendorff alpha ######')
        
        for group, labels in dist_task_n.items():
            print('Group: ', group)
            
            dist_group = list(labels.items())
            dist_group.sort(key=lambda x: x[0])
            row_columns = [k for k,_ in dist_group]
            alpha_matrix = pd.DataFrame(index=row_columns, columns=row_columns)
            p_value_matrix = pd.DataFrame(index=row_columns, columns=row_columns)
            level_of_measurement = 'nominal' if len(row_columns) == (6 + len(LLMS)) else 'ratio'
            
            for (col1, labels1), (col2, labels2) in tqdm(combinations_with_replacement(dist_group, 2), desc="Krippendorff alpha", ncols=100):
                
                if labels1 == labels2:
                    alpha_value = 1
                else:
                    #DEBUG:
                    print('@@@@@ DEBUG @@@@')
                    print(col1)
                    print(type(labels1))
                    print(len(labels1))
                    print(col2)
                    print(type(labels2))
                    print(len(labels2))
                    
                    alpha_value = krippendorff.alpha([labels1, labels2], level_of_measurement=level_of_measurement)
                
                alpha_matrix.at[col1, col2] = alpha_value
                alpha_matrix.at[col2, col1] = alpha_value  # Symmetric matrix
            
                # calculate statistic significance for alpha matrix
                p_value = self.statistic_significance_Krippendorff_alpha(level_of_measurement, labels1, labels2, alpha_value, n_permutations=10)
                
                p_value_matrix.at[col1, col2] = p_value
                p_value_matrix.at[col2, col1] = p_value  # Symmetric matrix
            
            
            # save alpha matrix to csv file
            alpha_matrix.to_csv(self.analyses_path + '/Krippendorff-alpha_'+ task + '_' + categoty + '_' + group +'.csv', index=True)
            p_value_matrix.to_csv(self.analyses_path + '/P_vakues_Krippendorff-alpha_'+ task + '_' + categoty + '_' + group +'.csv', index=True)
            print(alpha_matrix)
            

    def statistic_significance_Krippendorff_alpha(self, level_of_measurement, annotations1, annotations2, alpha_value, n_permutations, seed=42):
        np.random.seed(seed)
        
        permuted_alphas = []
        
        for _ in range(n_permutations):
            perm_annotations1 = np.random.permutation(annotations1).tolist()
            perm_annotations2 = np.random.permutation(annotations2).tolist()
            
            if perm_annotations1 == perm_annotations2:
                permuted_alpha = 1
            else:
                # Compute Krippendorff's alpha for the permuted dataset
                permuted_alpha = krippendorff.alpha([perm_annotations1, perm_annotations2],level_of_measurement=level_of_measurement)
                
            permuted_alphas.append(float(permuted_alpha))
        
        # Calculate p-value: proportion of permuted alphas >= original alpha
        p_value = sum(1 for pa in permuted_alphas if pa >= alpha_value) / n_permutations
        
        return p_value


    def t_test(self, labels_task_n):
        print('###### T-test ######')
        if len(labels_task_n['gender'].values()) <= 2:
            _, p = ttest_ind(*list(labels_task_n['gender'].values()))
            print('Group: gender')
            print('p: ', p)
        
    def array_df(self, array, columns, index):
        return pd.DataFrame(array, columns=columns, index=index)
    
        
    def statistic_significances(self):
        for n, categories in zip(range(1, 4), [{''}, self.possible_labels_task2, self.possible_labels_task3]):
            join = '_' if n>1 else ''
            
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@ TASK'+ str(n) + ' @@@@@@@@@@@@@@@@@@')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            
            print('\n')
            
            #DEBUG
            # self.df.to_csv(REPO_PATH + '/data_visualization_Task' + str(n) + '_' +  '.csv', index=True)
            
            for label in categories:
                anotations = self.mean_distributions(self.df['labels_task'+ str(n) + join + label].tolist())
                label_column = 'YES' if label == '' else 'IDEOLOGICAL-AND-INEQUALITY' if label == 'IDEOLOGICAL-INEQUALITY' else 'STEREOTYPING-AND-DOMINANCE' if label == 'STEREOTYPING-DOMINANCE' else 'MISOGYNY-AND-NON-SEXUAL-VIOLENCE' if label == 'MISOGYNY-NON-SEXUAL-VIOLENCE' else label
                anotations = include_llms_preds(anotations,label_column, n)
                
                #DEBUG
                # save_dict_to_json(data=anotations, file_path=REPO_PATH + '/data_visualization_Task' + str(n) + '_' + '.json')
                
                print('#####################')
                print('###### Class: ' + label_column)
                print('#####################')
                print('\n')
                self.t_test(anotations)
                print('\n') 
                self.anova_test(anotations)
                print('\n') 
                self.tukey_hsd_test(anotations, 'task'+ str(n), label)
                print('\n')
                self.Krippendorff_alpha(anotations, 'task'+ str(n), label)
                print('\n')
                self.mean_value(anotations, 'task'+ str(n), label)
                
                
def save_dict_to_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
        
def include_llms_preds(annotations, category, task_number):
    
    llms_preds = sorted([file for file in os.listdir(JSON_PREDICTIONS_PATH) if (('_' + '-'.join(category.split(' ')) + '_') in file) and ('Task' + str(task_number) in file)])
    #DEBUG:
    print('LLMs predictions: ', llms_preds)
    
    for preds_file in llms_preds:
        with open(os.path.join(JSON_PREDICTIONS_PATH, preds_file), 'r') as f:
            llm_preds_json = json.load(f)
    
        # Iterate through the first layer keys
        for key in annotations.keys():
                # Add the LLM predictions to the annotations dict
                for llm, preds in llm_preds_json[key].items():
                    annotations[key][llm] = preds

    return annotations

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataStatistics(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    data_exploration.statistic_significances()
    exit()