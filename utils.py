import os
import shutil
import pandas as pd
import json

def grabe_exist_package(source_package_path, package_path):
    #create a data folder
    if os.path.exists(package_path):
        shutil.rmtree(package_path)
    shutil.copytree(source_package_path, package_path)


def merge_data_labels(package_path, label_gold_path, data_path, dataset):
    #create a data folder
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    
    # partition dev/traninig
    for partition in ['dev', 'training']:
        path_partition = package_path + '/' + partition + '/' + dataset + '_' + partition + '.json'
        df_partition = pd.read_json(path_partition, orient='index')
        
        # Task 1/2/3
        # for task in ['task1', 'task2', 'task3']:
        #     path_label = label_gold_path + '/' + dataset + '_' + partition + '_' + task + '_gold_soft.json'
        #     df_label = pd.read_json(path_label, orient='index')
        #     df_label.rename(columns={"soft_label": 'soft_label' + '_' + task }, inplace=True)
            
        #     df_partition = pd.concat([df_partition, df_label], axis=1)
        
        for task in ['task1', 'task2', 'task3']:
            for type_label in ['soft', 'hard']:
                path_label = label_gold_path + '/' + dataset + '_' + partition + '_' + task + '_gold_' + type_label + '.json'
                df_label = pd.read_json(path_label, orient='index')
                df_label.rename(columns={type_label + "_label": type_label + '_label' + '_' + task }, inplace=True)
            
                df_partition = pd.concat([df_partition, df_label], axis=1)
                df_partition[type_label + '_label' + '_' + task ].fillna('TIE', inplace=True)
            
        path_csv = data_path + '/' + dataset + '_' + partition + '.csv'
        df_partition.to_csv(path_csv, index=False)


def merge_training_dev(data_path, dataset):
    partition_list =[]
    partition_strings_list = [] 
    # partition dev/traninig
    for partition in ['training', 'dev']:
        path_partition = data_path + '/' + dataset + '_' + partition + '.csv'
        df_partition = pd.read_csv(path_partition)
        partition_list.append(df_partition)
        partition_strings_list.append(partition)
            
    df_partition = pd.concat(partition_list)
            
    path_csv = data_path + '/' + dataset + '_' + '-'.join(partition_strings_list) + '.csv'
    df_partition.to_csv(path_csv, index=False)


def merge_gold_soft_label(label_gold_path, dataset):
    for task in ['task1', 'task2', 'task3']:
        json_list = []
        for partition in ['training','dev']:
            
            path_label = label_gold_path + '/' + dataset + '_' + partition + '_' + task + '_gold_soft.json'
            with open(path_label, 'r') as file:
                data = json.load(file)
            json_list.append(data)

        path_merge = label_gold_path + '/' + dataset + '_' + 'training-dev' + '_' + task + '_gold_soft.json'
        json_list[0].update(json_list[1])
        with open(path_merge, 'w') as merged_file:
            json.dump(json_list[0], merged_file, indent=2)