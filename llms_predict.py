import pandas as pd
from llmsapi import llmsAPI
import logging
from icecream import ic
from config import *
from tqdm import tqdm

# Configure the logger
logging.basicConfig(filename='summarization.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class PROMPT:
    def __init__(self):
        pass
    
    def ZeroShotTask1(self, tweet):
        return '''The system's primary task is determining whether a tweet contains sexist expressions or 
                    behaviours (i.e., the tweet is sexist itself, describes a sexist situation, or criticises sexist 
                    behaviour). The system must retrieve YES if the TWEET below contains sexist
                    expressions or behaviours and retrieve NO otherwise. The system must not retrieve any text 
                    apart from the two possible categories, YES and NO.
        
        TWEET: {tweet}'''.format(tweet=tweet)
        
    def ZeroShotTask2(self, tweet):
        return '''The system's primary task is categorising a tweet containing sexist expressions or behaviours
                    according to the author's intention. There are three possible categories: DIRECT, REPORTED
                    and JUDGEMENTAL.
                    
                Categories definition:
                    DIRECT: the intention was to write a message that is sexist by itself or incites to be sexist.
                    REPORTED: the intention was to report and share a sexist situation suffered by a woman or women in the first or third person.
                    JUDGEMENTAL: the intention was to judge since the tweet describes sexist situations or behaviours with the aim of condemning them.
                    
                The system must classify the TWEET below among one of the three categories. The system must not retrieve any text apart from the category, including explanations.
                    
        
        TWEET: {tweet}'''.format(tweet=tweet)
        
    def ZeroShotTask3(self, tweet):
        return '''The system's primary task is to label a tweet containing sexist expressions or behaviours 
                    according to the sexism types presented in the tweet. It is a multilabel task, so a tweet can be 
                    assigned to one or more categories. 

                There are five possible sexism-type categories: 
                    IDEOLOGICAL AND INEQUALITY
                    STEREOTYPING AND DOMINANCE
                    OBJECTIFICATION
                    SEXUAL VIOLENCE
                    MISOGYNY AND NON-SEXUAL VIOLENCE

                Sexism types definition:
                    IDEOLOGICAL AND INEQUALITY: The text discredits the feminist movement, rejects inequality between men and women or presents men as victims of gender-based oppression.
                    STEREOTYPING AND DOMINANCE: The text expresses false ideas about women that suggest they are more suitable to fulfill certain roles (mother, wife, family caregiver, faithful, tender, loving, submissive, etc.), or inappropriate for certain tasks (driving, hardwork, etc), or claims that men are somehow superior to women.
                    OBJECTIFICATION: The text presents women as objects apart from their dignity and personal aspects or assumes or describes certain physical qualities that women must have in order to fulfil traditional gender roles (compliance with beauty standards, hypersexualisation of female attributes, women’s bodies at the disposal of men, etc.).
                    SEXUAL VIOLENCE: Sexual suggestions, requests for sexual favours or harassment of a sexual nature (rape or sexual assault) are made.
                    MISOGYNY AND NON-SEXUAL VIOLENCE: The text expresses hatred and violence towards women.


                The system must label the TWEET below with one or more sexism-type categories and put them inside parentheses as ['category', 'category', …]. The system must not retrieve any text apart from the categories, including explanations.

        
        TWEET: {tweet}'''.format(tweet=tweet)
        
    
    def get_prompt(self, prompt_type, tweet):
        if prompt_type == "ZeroShotTask1":
            return self.ZeroShotTask1(tweet)
        elif prompt_type == "ZeroShotTask2":
            return self.ZeroShotTask2(tweet)
        elif prompt_type == "ZeroShotTask3":
            return self.ZeroShotTask3(tweet)
        else:
            raise ValueError(f"The prompt type {prompt_type} is not available.")
        

class DATA:
    def __init__(self):
        self.data_file = DATA_PATH + '/'+ 'EXIST2023_training-dev.csv'

    def load_data(self):
        columns = ['id_EXIST','lang','tweet']
        self.df_data = pd.read_csv(self.data_file,
                                usecols=columns,
                                index_col='id_EXIST')
        
        #DEBUG:
        # self.df_data = self.df_data.head(10)
        
        if any(task in self.prompt_type for task in ['Task2', 'Task3']) :
            self.df_data = self._task_filter_data_()
        
    def _task_filter_data_(self):
        df_task1 = pd.read_csv(PREDICTIONS_PATH + '/' + self.model + '_' + 'ZeroShotTask1' + '.tsv', sep='\t', index_col='id_EXIST')
        # return self.df_data.loc[df_task1.loc[df_task1[self.model]=='YES'].index.tolist()]
            # Get the indices where the model prediction is 'YES'
        yes_indices = df_task1.loc[df_task1[self.model] == 'YES'].index
        
        # Find the intersection of indices between self.df_data and yes_indices
        common_indices = self.df_data.index.intersection(yes_indices)
        
        # Return the filtered DataFrame
        return self.df_data.loc[common_indices]
    

class ChatLLM(DATA, PROMPT):
    def __init__(self, model, prompt_type, max_tokens=50):
        super().__init__()
        self.prompt_type = prompt_type
        self.model = model
        self.max_tokens = max_tokens

    def get_predictions(self):
        predictions = []
        
        for i in tqdm(range(len(self.df_data)), desc="Predicting Sexism", leave=False, position=1, ncols=100):  # Add tqdm to the loop
            tweet = self.df_data.iloc[i]['tweet']
            prompt = super().get_prompt(self.prompt_type, tweet)
            
            #DEBUG:
            # tqdm.write('\n')
            # tqdm.write(prompt)
            # exit()
            
            llms_api = llmsAPI(prompt)
            output = llms_api.get_completion(llm=self.model, max_tokens=self.max_tokens)
            predictions.append(output)

        self.df_data[self.model] = predictions
        
        
    def save_predictions(self):
        self.df_data.loc[:,['lang', self.model]].to_csv(PREDICTIONS_PATH + '/' + self.model + '_' + self.prompt_type + '.tsv', sep='\t')
    
    def main(self):
        super().load_data()
        self.get_predictions()
        self.save_predictions()
        


if __name__ == '__main__':
    # for llm in tqdm(["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o-2024-05-13"], desc="LLMs", position=0,ncols=100):
    #     for prompt in tqdm(["ZeroShotTask1", "ZeroShotTask2", "ZeroShotTask3"], desc="Prompts", position=1, ncols=100):
    
    #DEBUG:
    for llm in tqdm(["gpt-3.5-turbo-0125"], desc="LLMs", position=0,ncols=100):
        for prompt in tqdm(["ZeroShotTask3"], desc="Prompts", position=1, ncols=100):
        
            LlmPreds = ChatLLM(
                                model=llm, 
                                prompt_type=prompt,
                                max_tokens=50)
            LlmPreds.main()
        