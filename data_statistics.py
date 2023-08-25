import pandas as pd
from config import *
from scipy.stats import f_oneway, ttest_ind, tukey_hsd
import numpy as np
from data_exploration import DataExploration
import matplotlib.pyplot as plt

class DataStatistics(DataExploration):
    def __init__(self, data_path, file_name, analyses_path):
        super().__init__(data_path, file_name, analyses_path)
        self.num_annotators = {'gender':3, 'age':2, 'gender_age':1}
        
    def average_last_n_values(self, dict_scores, n):
        for key in dict_scores.keys():
            dict_scores[key] = dict_scores[key][:-n] + [sum(dict_scores[key][-n:]) / len(dict_scores[key][-n:])]  
        return dict_scores
    
    def mean_distribution_task1(self):
        anotations = {}
        anotations['gender'] = {g:[] for g in ['F', 'M']}
        anotations['age'] = {a:[] for a in self.age_groups}
        anotations['gender_age'] = {g + '_' + a:[] for g in ['F', 'M'] for a in self.age_groups}
    
        for set_gender, set_age, set_annotation in zip(self.gender_annotetors, self.age_annotetors, self.labels_task1):
            for gender, age, annotation in zip(set_gender, set_age, set_annotation):
                anotations['gender'][gender].append(1 if annotation=='YES' else 0)
                anotations['age'][age].append(1 if annotation=='YES' else 0)
                anotations['gender_age'][gender + '_' + age].append(1 if annotation=='YES' else 0)
                    
            for set_annotation in anotations.keys():
                anotations[set_annotation] = self.average_last_n_values(anotations[set_annotation], self.num_annotators[set_annotation])      
        return anotations
    
    def anova_test(self):
        print('###### ANOVA ######')
        dist_task1 = self.mean_distribution_task1()
        
        for group, labels in dist_task1.items():
            print('Group: ', group)
            rest = f_oneway(*[l for l in labels.values()])
            print('p: ',rest.pvalue)

    def tukey_hsd_test(self):
        print('###### Tukey HSD ######')
        dist_task1 = self.mean_distribution_task1()
        
        for group, labels in dist_task1.items():
            print('Group: ', group)
            
            dist_group = list(labels.items())
            dist_group.sort(key=lambda x: x[0])
            dist_group_names = [k for k,_ in dist_group]
            dist_group_values = [v for _,v in dist_group]
            
            rest = tukey_hsd(*dist_group_values)
            df = self.array_df(rest.pvalue, dist_group_names, dist_group_names)
            df.to_csv(self.analyses_path + '/tukey_'+ group +'.csv', index=True)
            print(df)
            print('\n')

    def t_test(self):
        print('###### T-test ######')
        scores = {'F':[], 'M':[]}
        for set_gender, set_annotation in zip(self.gender_annotetors, self.labels_task1):
            for gender, annotation in zip(set_gender, set_annotation):
                scores[gender].append(1 if annotation=='YES' else 0)
            scores = self.average_last_n_values(scores, self.num_annotators['gender'])

        _, p = ttest_ind(*list(scores.values()))
        print('Group: gender')
        print('p: ', p)
        
    def array_df(self, array, columns, index):
        return pd.DataFrame(array, columns=columns, index=index)
    
box_plot('gender', [v for _,v in list_scores], [k for k,_ in list_scores], self.analyses_path)
def box_plot(group, scores, classes, path):
    fig, ax = plt.subplots(1, 1)
    ax.boxplot(scores)
    ax.set_xticklabels(classes) 
    ax.set_ylabel("mean")
    ax.set_title("Boxplot of " + group) 
    plt.savefig(path +'/boxplot_'+ group +'.png')

if __name__ == "__main__":
    # Global distribution of the data
    data_exploration = DataStatistics(DATA_PATH, 'EXIST2023_training-dev.csv', ANALYSES_PATH)
    data_exploration.t_test()
    print('\n')
    data_exploration.anova_test()
    print('\n')
    data_exploration.tukey_hsd_test()