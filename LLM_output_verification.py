import pandas as pd
import os
import ast
import re
from difflib import get_close_matches
from config import PREDICTIONS_PATH, CATEGORIES_TASK1, CATEGORIES_TASK2, CATEGORIES_TASK3

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
        
        llm = preds_file.split('_')[0]
        
        df[llm+'_processed'] = df[llm].apply(lambda x: '' if isinstance(x, float) else x)
        categories_task = CATEGORIES_TASK1 if 'Task1' in preds_file else CATEGORIES_TASK2 if 'Task2' in preds_file else CATEGORIES_TASK3
        df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: process_categories(categories_task, x))
        df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: add_UNKNOWN_categories(x))
        
        if 'Task2' in preds_file or 'Task3' in preds_file:
            task1_file = preds_file.split('.tsv')[0][:-1] + '1' + '_processed.tsv'
            df_task1 = pd.read_csv(os.path.join(PREDICTIONS_PATH, task1_file), sep='\t', index_col='id_EXIST')
            df_task1 = df_task1.loc[df_task1[llm+'_processed']=='NO',['lang']]
            
            df = pd.concat([df,df_task1])

            df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: add_category_no_sexist(x, preds_file))
            df = df.sort_index()
        
        save_path = os.path.join(PREDICTIONS_PATH, preds_file.split('.tsv')[0] + '_processed.tsv')
        df.to_csv(save_path, sep='\t')

if __name__ == "__main__":
    output_report()
    process_predictions()
    output_report()