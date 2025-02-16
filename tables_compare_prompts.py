from config import *
import pandas as pd 


files_pred_list =  sorted(os.listdir(PREDICTIONS_PATH))


# for llm in LLMS:
for llm in ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06", "o1"]:
    # for ["ZeroShotTask1_VersionZero", "ZeroShotTask1_AdpatAlister", "ZeroShotTask1_Johanne", "ZeroShotTask1_o1_Preview"] in PROMPTS:
    for demographic in [ "__", "female", "male", "18-22", "23-45", "46+"]:
        
        dataframes = []
        for file in files_pred_list:
            if file.startswith(llm) and '_' + demographic in file:
                file_path = os.path.join(PREDICTIONS_PATH, file)
                print(file_path)
                df = pd.read_csv(file_path, usecols=['id_EXIST', llm], index_col='id_EXIST', sep='\t')
                
                prompt = file.split('_')[-1].split('.')[0]
                df.rename(columns={llm:prompt}, inplace=True)
                
                dataframes.append(df)
                
        if dataframes:
            df = pd.concat(dataframes, axis=1)
            
            df.to_csv(ANALYSES_PATH + '/' + 'prompt_analizes' + '_' + llm + '_' + demographic + '.tsv', sep='\t')