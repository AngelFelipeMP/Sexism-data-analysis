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
    
    # def mean_distributions(self, labels_task_n):
    #     anotations = {}
    #     anotations['gender'] = {g:[] for g in ['F', 'M']}
    #     anotations['age'] = {a:[] for a in self.age_groups}
    #     anotations['gender_age'] = {g + '_' + a:[] for g in ['F', 'M'] for a in self.age_groups}
    
    #     for set_gender, set_age, set_annotation in zip(self.gender_annotetors, self.age_annotetors, labels_task_n):
    #         for gender, age, annotation in zip(set_gender, set_age, set_annotation):
    #             anotations['gender'][gender].append(1 if annotation=='YES' else 0)
    #             anotations['age'][age].append(1 if annotation=='YES' else 0)
    #             anotations['gender_age'][gender + '_' + age].append(1 if annotation=='YES' else 0)
                    
    #         for set_annotation in anotations.keys():
    #             anotations[set_annotation] = self.average_last_n_values(anotations[set_annotation], self.num_annotators[set_annotation])
                
    #     return anotations            
    
    def mean_value(self, dist_task_n, task, categoty):
        print('******* Mean *******')
        for group, labels in dist_task_n.items():
            print('Group: ', group)
            for label, values in labels.items():
                print(label, np.mean(values))
                
                dist_task_n[group][label] = np.mean(values)
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
            
    # self.Krippendorff_alpha(anotations, 'task'+ str(n), label)
    def Krippendorff_alpha(self, dist_task_n, task, categoty):
        print('###### Krippendorff alpha ######')
        
        for group, labels in dist_task_n.items():
            print('Group: ', group)
            
            dist_group = list(labels.items())
            dist_group.sort(key=lambda x: x[0])
            row_columns = [k for k,_ in dist_group]
            alpha_matrix = pd.DataFrame(index=row_columns, columns=row_columns)
            level_of_measurement = 'nominal' if len(row_columns) == 6 else 'ratio'
            
            for (col1, labels1), (col2, labels2) in combinations_with_replacement(dist_group, 2):
            
                alpha_value = krippendorff.alpha([labels1, labels2], level_of_measurement=level_of_measurement)
                
                alpha_matrix.at[col1, col2] = alpha_value
                alpha_matrix.at[col2, col1] = alpha_value  # Symmetric matrix
            
            alpha_matrix.to_csv(self.analyses_path + '/Krippendorff-alpha_'+ task + '_' + categoty + '_' + group +'.csv', index=True)
            print(alpha_matrix)

    def t_test(self, labels_task_n):
        print('###### T-test ######')
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
                #COMMENT: Including LLMs predictions-> working on it
                # annotations = self.include_llms_preds(anotations)
                
                #DEBUG
                # save_dict_to_json(data=anotations, file_path=REPO_PATH + '/data_visualization_Task' + str(n) + '_' + '.json')
                
                print('#####################')
                print('###### Class: ' + label)
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
        
        

def dataframe_to_dict(df):
    




class LlmsPredsProcessing:
    def __init__(self, tsv_files_path, task):
        self.tsv_files_path = tsv_files_path
        self.task = task
    
    def _get_llms_preds_paths_(self):
        return glob.glob(os.path.join(self.tsv_files_path, '*{self.task}_processed.tsv'))
    
    def _load_llms_preds_(self, preds_paths):
        return {('-').join(path.split('_')[0].split('-')[:2]): pd.read_csv(path, sep='\t', index_col='id_EXIST') for path in preds_paths}
    
    def _add_non_sexist_index_(self, dict_dataframes, llms_preds_path):
        for key in dict_dataframes.keys():
            path = [('Task1.').join(p.split('Task*.')) for p in llms_preds_path if p.starswith(key + '.tsv')][0]
            df_task1 = pd.read_csv(path, sep='\t', index_col='id_EXIST')
            dict_dataframes[key] = dict_dataframes[key].reindex(df_task1.index, fill_value=0)
            dict_dataframes[key]['lang'] =  df_task1['lang']
            
        return dict_dataframes
    
    def _dataframe_to_dict_(self, dict_dataframes):
        #### STOP HERE !!!!!
        
    def main(self):
        llms_preds_files = self._get_llms_preds_paths_()
        llms_preds_dataframes = self._load_llms_preds_(llms_preds_files)
        
        if self.task != 'task1':
            # Add non-sexist index to the dataframes
            llms_preds_dataframes = self._add_non_sexist_index_(llms_preds_dataframes, llms_preds_files)
            
        return 
        

Llms_preds = LlmsPredsProcessing(PREDICTIONS_PATH)

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataStatistics(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    data_exploration.statistic_significances()
    exit()