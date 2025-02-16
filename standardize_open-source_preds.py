from config import *
import pandas as pd
import json


class StandardizeOpenSourcePreds:
    def __init__(self):
        self.open_source_preds_path = REPO_PATH + '/preds_open_source'
        self.prompt_type = 'ZeroShotTask1_o1_Preview'
        self.save_path = REPO_PATH + '/predictions_test'
        # self.save_path = PREDICTIONS_PATH
        self.data_file = DATA_PATH + '/'+ 'EXIST2023_training-dev.csv'
        
    def _list_files(self):
        return os.listdir(self.open_source_preds_path)
    
    def _filter_for_tsv(self, files):
        return sorted([file for file in files if file.endswith('.tsv')])
    
    def _read_map(self, files):
        json_file = [file for file in files if file.endswith('.json')][0]
        with open(self.open_source_preds_path + '/' + json_file, 'r') as f:
            map_data = json.load(f)
        return map_data
    
    def save_predictions(self, df, model, gender, age):
        df.to_csv(self.save_path + '/' + model + '_' + gender + '_' + age + '_' + self.prompt_type + '.tsv', sep='\t')
        
    def load_data(self):
        columns = ['id_EXIST','lang']
        return pd.read_csv(self.data_file,
                                usecols=columns,
                                index_col='id_EXIST')
    
    def _include_lang(self, df, llm):
        load_data = self.load_data()
        df['lang'] = load_data['lang']
        return df[['lang', llm]]
    
    
    def _marge_train_dev(self, tsv_files, maping):
        for i in range(0,len(tsv_files)):
            file_a = tsv_files[i]
            llm=file_a.split('-')[0]
            df = pd.read_csv(self.open_source_preds_path + '/' + file_a, names=['id_EXIST', llm], index_col='id_EXIST', sep='\t')
                        
            for j in range(i+1, len(tsv_files)):
                file_b = tsv_files[j]
                
                if file_a.split('.')[0] == file_b.split('.')[0]:
                    df_b = pd.read_csv(self.open_source_preds_path + '/' + file_b, names=['id_EXIST', llm], index_col='id_EXIST', sep='\t')
                    
                    #DEBUG
                    print('file_a:', file_a)
                    print(len(df))
                    print('file_b:', file_b)
                    print(len(df_b))
                    
                    
                    df = pd.concat([df, df_b])
                    df = df.sort_index()
                    
            #COMMENT: Once I have train/dev preds move the line below to the left
                    df = self._include_lang(df, llm)
                    demographics = maping[file_a.split('-')[-1].split('.')[0]]
                    self.save_predictions(df, llm, demographics['gender'], demographics['age'])
                    
                    break

            
    
    def main(self):
        files = self._list_files()
        tsv_files = self._filter_for_tsv(files)
        map_file = self._read_map(files)
        self._marge_train_dev(tsv_files, map_file)
        
        

if __name__ == '__main__':
    StandardizeOpenSourcePreds().main()
    