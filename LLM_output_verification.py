import pandas as pd
import os
import ast
import re
import json
from difflib import get_close_matches
from config import PREDICTIONS_PATH, CATEGORIES_TASK1, CATEGORIES_TASK2, CATEGORIES_TASK3, JSON_PREDICTIONS_PATH, AGE_GROUPS, GENDER_GROUPS

def output_report():
    preds_list =  sorted(os.listdir(PREDICTIONS_PATH))

    for preds_file in preds_list:
        print(f'PRED FILE: {preds_file}')
        file_path = os.path.join(PREDICTIONS_PATH, preds_file)
        df = pd.read_csv(file_path, sep='\t', index_col='id_EXIST')
        
        llm = preds_file.split('_')[0]
        llm = llm+'_processed' if '_processed' in preds_file else llm
            
        if 'Task3' in preds_file: 
            list_outputs = [ast.literal_eval(output) for output in df[llm].to_list()]
            flattened_list = [item for sublist in list_outputs for item in sublist]
            print(list(set(flattened_list)))
        else:
            print(df[llm].unique())
            
        print('\n')

def process_categories(official_categories, llm_output):
    official_categories_plus_empty = official_categories + ['']
    filtered_categories = re.findall(r'\b(?:[A-Z]+(?:-[A-Z]+)*\s?)+\b', llm_output)
    processed_categories = [
            cat if cat in official_categories_plus_empty else get_close_matches(cat, official_categories_plus_empty, n=1)[0]
            for cat in filtered_categories
            ]
    
    if len(official_categories_plus_empty) == len(CATEGORIES_TASK3+['']):
        return processed_categories
    else:
        return processed_categories[0] if processed_categories else ''

def add_UNKNOWN_categories(llm_output):
    if isinstance(llm_output, list):
        return ['UNKNOWN'] if not llm_output else llm_output
    elif isinstance(llm_output, str):
        return llm_output if llm_output != '' else 'UNKNOWN'
    else:
        return 'UNKNOWN'
    
def add_category_no_sexist(pred, task):
    if isinstance(pred, list):
        return pred
    else:
        if pd.isna(pred):
            if 'Task2' in task:
                return '-'
            elif 'Task3' in task:
                return ['-']
        else:
            return pred
    
    
def process_predictions():
    preds_list = [file for file in os.listdir(PREDICTIONS_PATH) if '_processed' not in file]
    preds_list = sorted(preds_list, key=lambda x: (0 if 'Task1' in x else 1, x))
    
    # preds_list = ['gpt-4o-2024-08-06_ZeroShotTask1.tsv']

    for preds_file in preds_list:
        print(f'PROCESSING FILE: {preds_file}')
        file_path = os.path.join(PREDICTIONS_PATH, preds_file)
        df = pd.read_csv(file_path, sep='\t', index_col='id_EXIST')
        
        #DEBUG
        print(len(df))
        
        llm = preds_file.split('_')[0]
        
        df[llm+'_processed'] = df[llm].apply(lambda x: '' if isinstance(x, float) else x)
        categories_task = CATEGORIES_TASK1 if 'Task1' in preds_file else CATEGORIES_TASK2 if 'Task2' in preds_file else CATEGORIES_TASK3
        df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: process_categories(categories_task, x))
        df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: add_UNKNOWN_categories(x))
        
        if 'Task2' in preds_file or 'Task3' in preds_file:
            task1_file = preds_file.split('.tsv')[0][:-1] + '1' + '_processed.tsv'
            #DEBUG
            print(task1_file)
            
            df_task1 = pd.read_csv(os.path.join(PREDICTIONS_PATH, task1_file), sep='\t', index_col='id_EXIST')
            
            #DEBUG
            print(len(df_task1))
            print(df_task1[llm+'_processed'].unique())
            print(len(df_task1.loc[df_task1[llm+'_processed']=='NO',['lang']]))
            print(len(df_task1.loc[df_task1[llm+'_processed']=='YES',['lang']]))
            
            df_task1 = df_task1.loc[df_task1[llm+'_processed']=='NO',['lang']]
            
            #DEBUG
            print(len(df))
            
            df = pd.concat([df,df_task1])
            
            #DEBUG
            print(len(df))

            df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: add_category_no_sexist(x, preds_file))
            df = df.sort_index()
        
        df = create_categories_columns(df, llm, categories_task)
        
        save_path = os.path.join(PREDICTIONS_PATH, preds_file.split('.tsv')[0] + '_processed.tsv')
        df.to_csv(save_path, sep='\t')
        
#TODO: add ['UNKNOWN'] + ['-'] categories task 2 and Task 3
# def create_categories_columns(df, llm_file):
#     for llm in df.columns:
#         if 'processed' in llm:
#             if 'Task3' in llm_file:
#                 categories = [cat for cat_list in df[llm].to_list() for cat in cat_list]
#                 categories = list(set(categories))
#             else:
#                 categories = df[llm].unique()
            
#     for category in categories:
#         df[category] = df[llm].apply(lambda x: 1 if category in x else 0)
        
#     return df

def create_categories_columns(df, llm, standard_categories):
    categories = standard_categories + ['UNKNOWN'] + ['-']
    
    for category in categories:
        df[category] = df[llm+'_processed'].apply(lambda x: 1 if category in x else 0)
        
    return df


def predictions_to_json():
    preds_list = [file for file in os.listdir(PREDICTIONS_PATH) if '_processed' in file]
    
    for preds_file in preds_list:
        file_path = os.path.join(PREDICTIONS_PATH, preds_file)
        df = pd.read_csv(file_path, sep='\t', index_col='id_EXIST')
        
        #COMMENT: Adpatading for acommodation LLM with demographics
        # llm = preds_file.split('_')[0]
        llm = preds_file.split('ZeroShotTask')[0]
        
        categories_task = CATEGORIES_TASK1 if 'Task1' in preds_file else CATEGORIES_TASK2 if 'Task2' in preds_file else CATEGORIES_TASK3 
        #TODO: chech if the codes need categori: ['-']
        categories_task = categories_task + ['UNKNOWN'] + ['-']
        
        for cat in categories_task:
            dict_cats = {"gender":{},"age":{},"gender_age":{}}
            
            if any(age in llm for age in AGE_GROUPS) and any(gender in llm for gender in GENDER_GROUPS):
                dict_cats["gender_age"][llm] = df[cat].to_list()
            elif any(age in llm for age in AGE_GROUPS):
                dict_cats["age"][llm] = df[cat].to_list()
            elif any(gender in llm for gender in GENDER_GROUPS):
                dict_cats["gender"][llm] = df[cat].to_list()
            else:
                for key in dict_cats.keys():
                    dict_cats[key][llm] = df[cat].to_list()
                
                
                
                
                #DEBUG !!!!
                    
            with open(os.path.join(JSON_PREDICTIONS_PATH, preds_file.split('.tsv')[0] + '_' + '-'.join(cat.split(' ')) + '_.json'), 'w') as f:
                json.dump(dict_cats, f, indent=4)
            

if __name__ == "__main__":
    output_report()
    process_predictions()
    output_report()
    predictions_to_json()