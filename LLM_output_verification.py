import pandas as pd
import os
import ast
import re
from difflib import get_close_matches
from config import PREDICTIONS_PATH, CATEGORIES_TASK1, CATEGORIES_TASK2, CATEGORIES_TASK3

# path = 'gpt-3.5-turbo-0125_ZeroShotTask{number}.tsv'

def output_report():
    preds_list = os.listdir(PREDICTIONS_PATH)

    for preds in preds_list:
        print(f'PRED FILE: {preds}')
        file_path = os.path.join(PREDICTIONS_PATH, preds)
        df = pd.read_csv(file_path, sep='\t')
        
        llm = preds.split('_')[0]
        
        if 'Task3' in preds: 
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
    

def process_predictions():
    preds_list = [file for file in os.listdir(PREDICTIONS_PATH) if '_processed' not in file]
    
    # preds_list = ['gpt-4o-2024-08-06_ZeroShotTask1.tsv']

    for preds in preds_list:
        file_path = os.path.join(PREDICTIONS_PATH, preds)
        df = pd.read_csv(file_path, sep='\t', index_col='id_EXIST')
        
        llm = preds.split('_')[0]
        
        df[llm+'_processed'] = df[llm].apply(lambda x: '' if isinstance(x, float) else x)
        categories_task = CATEGORIES_TASK1 if 'Task1' in preds else CATEGORIES_TASK2 if 'Task2' in preds else CATEGORIES_TASK3
        df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: process_categories(categories_task, x))
        df[llm+'_processed'] = df[llm+'_processed'].apply(lambda x: add_UNKNOWN_categories(x))
        
        if 'Task2' in preds or 'Task3' in preds:
            task1_file = preds.split('.tsv')[0][:-1] + '1' + '_processed.tsv'
            df_task1 = pd.read_csv(os.path.join(PREDICTIONS_PATH, task1_file), sep='\t', index_col='id_EXIST')
            df_task1 = df_task1.loc[df_task1[llm+'_processed']=='NO',['lang']]
            
            
            print(len(df))
            print(len(df_task1))
            df = pd.concat([df,df_task1])
            print(len(df))
            
            ## STOPED HERE !!!!
            df = df.fillna('-') if 'Task2' in preds else df.fillna(['-'])
            
            print(df_task1.head())
            print(df.tail())
            exit()
        
        save_path = os.path.join(PREDICTIONS_PATH, preds.split('.tsv')[0] + '_processed.tsv')
        # print(f'SAVED: {save_path}')
        # exit()
        df.to_csv(save_path, sep='\t', index=False)

if __name__ == "__main__":
    process_predictions()
    # output_report()