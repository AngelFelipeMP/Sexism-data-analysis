import pandas as pd
from llmsapi import llmsAPI
import logging
from icecream import ic
from config import *
from tqdm import tqdm
import time

# Configure the logger
logging.basicConfig(filename='summarization.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class PROMPT:
    def __init__(self):
        pass
    
    def ZeroShotTask1(self, tweet, demographic_info=''):
        return '''The system's primary task is determining whether a tweet contains sexist expressions or 
                    behaviours (i.e., the tweet is sexist itself, describes a sexist situation, or criticises sexist 
                    behaviour){demographic_info}. The system must retrieve YES if the TWEET below contains sexist
                    expressions or behaviours and retrieve NO otherwise. The system must not retrieve any text 
                    apart from the two possible categories, YES and NO.
        
        TWEET: {tweet}'''.format(tweet=tweet, demographic_info=demographic_info)
        
    def ZeroShotTask2(self, tweet, demographic_info=''):
        return '''The system's primary task is categorising a tweet containing sexist expressions or behaviours
                    according to the author's intention{demographic_info}. There are three possible categories: DIRECT, REPORTED
                    and JUDGEMENTAL.
                    
                Categories definition:
                    DIRECT: the intention was to write a message that is sexist by itself or incites to be sexist.
                    REPORTED: the intention was to report and share a sexist situation suffered by a woman or women in the first or third person.
                    JUDGEMENTAL: the intention was to judge since the tweet describes sexist situations or behaviours with the aim of condemning them.
                    
                The system must classify the TWEET below among one of the three categories. The system must not retrieve any text apart from the category, including explanations.
                    
        
        TWEET: {tweet}'''.format(tweet=tweet, demographic_info=demographic_info)
        
    def ZeroShotTask3(self, tweet, demographic_info=''):
        return '''The system's primary task is to label a tweet containing sexist expressions or behaviours 
                    according to the sexism types presented in the tweet{demographic_info}. It is a multilabel task, so a tweet can be 
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


                The system must label the TWEET below with ONE or MORE sexism-type categories and put them inside parentheses as ['category', 'category', … ]. The system must not retrieve any text apart from the categories, including explanations.


        TWEET: {tweet}'''.format(tweet=tweet, demographic_info=demographic_info)
        
        
        
    def ZeroShotTask1_VersionZero(self, tweet, gender, age):
        if gender and age:
            demographic_info = f''' from the perspective of a {gender} aged {age}'''
        elif gender:
            demographic_info = f''' from a {gender}'s perspective'''
        elif age:
            demographic_info = f''' from the perspective of someone aged {age} years old'''
        else:
            demographic_info = ''
        
        line1 = '''The system's primary task is determining whether a tweet contains sexist expressions or behaviours (i.e., a tweet is sexist itself, describes a sexist situation, or criticises sexist behaviour){demographic_info}.''' 
        line2 = '''The system must retrieve YES if the TWEET below contains sexist expressions or behaviours and retrieve NO otherwise. The system must not retrieve text apart from the two possible categories, YES and NO.'''
        line3 = '''TWEET: {tweet}'''
        
        return f"{line1} {line2} \n\n {line3}".format(tweet=tweet, demographic_info=demographic_info)
    
    def ZeroShotTask1_AdpatAlister(self, tweet, gender, age):
        if gender and age:
            demographic_info_1 = f''' from the perspective of a {gender} person aged {gender} reading the tweet'''
            demographic_info_2 = f''' a male person aged 18-22 reading'''
            demographic_info_3 = f''' would feel that it'''
        elif gender:
            demographic_info_1 = f''' from the perspective of a {gender} person reading the tweet'''
            demographic_info_2 = f''' a {gender} person reading'''
            demographic_info_3 = f''' would feel that it'''
            
        elif age:
            demographic_info_1 = f''' from the perspective of a person aged {age} reading the tweet'''
            demographic_info_2 = f''' a person aged {age} reading'''
            demographic_info_3 = f''' would feel that it'''
        else:
            demographic_info_1 = ''
            demographic_info_2 = ''
            demographic_info_3 = ''

        line1 = ''''Your task is to identify whether a tweet is sexist itself, describes a sexist situation, or criticises sexist behaviour{demographic_info_1}.'''
        line2 = '''You should reply YES if{demographic_info_2} the TWEET below{demographic_info_3} contains sexist expressions or behaviours and reply NO otherwise.'''
        line3 = '''You must not retrieve text apart from the two possible categories, YES and NO.'''
        line4 = '''TWEET: {tweet}'''
        return f"{line1} {line2} {line3} \n\n {line4}".format(tweet=tweet, demographic_info_1=demographic_info_1, demographic_info_2=demographic_info_2, demographic_info_3=demographic_info_3)
    
    def ZeroShotTask1_Johanne(self, tweet, gender, age):
        if gender and age:
            demographic_info_1 = f''' a {gender}'''
            demographic_info_2 = f''', aged {age}'''
        elif gender:
            demographic_info_1 = f''' a {gender}'''
            demographic_info_2 = ''
        elif age:
            demographic_info_1 = f''' an'''
            demographic_info_2 = f''', aged {age}'''
        else:
            demographic_info_1 = f''' an'''
            demographic_info_2 = ''

        line1 = '''You are{demographic_info_1} expert in identifying sexism in social media content{demographic_info_2}.'''
        line2 = '''You need to evaluate if a tweet contains sexist expressions or behaviours. These could include:'''
        line3 = '''- Directly sexist tweets'''
        line4 = '''- Tweets that describe a sexist situation'''
        line5 = '''- Tweets that criticize sexist behaviour'''
        line6 = '''Classify each tweet into one of two categories:'''
        line7 = '''-YES: The tweet contains sexist expressions, describes sexism, or references sexist behaviours.'''
        line8 = '''-NO: The tweet does not contain any sexist expressions or behaviours.'''
        line9 = '''Give your classification with only YES or NO for each tweet.'''
        line10 = '''You must not retrieve text apart from the two possible categories, YES and NO.'''
        line11 = '''TWEET: {tweet}'''
        
        return f"{line1} \n\n {line2} \n {line3} \n {line4} \n {line5} \n\n {line6} \n {line7} \n {line8} \n\n {line9} \n {line10} \n\n {line11}".format(tweet=tweet, demographic_info_1=demographic_info_1, demographic_info_2=demographic_info_2)
    
    def ZeroShotTask1_o1_Preview(self, tweet, gender, age):
        if gender and age:
            demographic_info = f''' Your demographic information is as follows: \n - Sex: {gender} \n - Age Group: {age}'''
        elif gender:
            demographic_info = f''' Your demographic information is as follows: \n - Sex: {gender}'''
        elif age:
            demographic_info = f''' Your demographic information is as follows: \n - Age Group: {age}'''
        else:
            demographic_info = ''
            
            
        line1 = '''You are an expert linguist specializing in detecting sexism in social media texts.{demographic_info}'''
        line2 = '''Task: Analyze the tweet and determine whether it contains sexist expressions or behaviors. This includes tweets that:'''
        line3 = ''' - Are sexist themselves (contain sexist language or promote sexist ideas).'''
        line4 = ''' - Describe a sexist situation (narrate or depict scenarios where sexism occurs).'''
        line5 = ''' - Criticize sexist behavior (call out or oppose sexism).'''
        line6 = '''Instructions:'''
        line7 = ''' 1. Read the Tweet Carefully: Examine the content of the tweet thoroughly.'''
        line8 = ''' 2. Make a Determination: Decide if the tweet fits any of the above categories related to sexism.'''
        line9 = ''' 3. Classification:'''
        line10 = '''    * Assign "YES" if the tweet contains sexist content as defined.'''
        line11 = '''    * Assign "NO" if the tweet does not contain sexist content.'''
        line12 = '''4. Output Format: Provide the assigned category in plain text.'''
        line13 = '''5. Constraint: You must not retrieve text apart from the two possible categories, YES and NO.'''
        line14 = '''TWEET: {tweet}'''

        return f"{line1} \n\n {line2} \n {line3} \n {line4} \n {line5} \n\n {line6} \n {line7} \n {line8} \n {line9} \n {line10} \n {line11} \n {line12} \n {line13} \n\n {line14}".format(tweet=tweet, demographic_info=demographic_info)
        
        
    
    def get_prompt(self, prompt_type, tweet, gender, age):
        if prompt_type == "ZeroShotTask1_VersionZero":
            return self.ZeroShotTask1_VersionZero(tweet, gender, age)
        elif prompt_type == "ZeroShotTask1_AdpatAlister":
            return self.ZeroShotTask1_AdpatAlister(tweet, gender, age)
        elif prompt_type == "ZeroShotTask1_Johanne":
            return self.ZeroShotTask1_Johanne(tweet, gender, age)
        elif prompt_type == "ZeroShotTask1_o1_Preview":
            return self.ZeroShotTask1_o1_Preview(tweet, gender, age)
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
        # self.df_data = self.df_data.sample(n=20, random_state=1)
        # self.df_data = self.df_data.head(10)
        # self.df_data = self.df_data.loc[[100025,100107,100111,100115,100123]]
        
        if any(task in self.prompt_type for task in ['Task2', 'Task3']) :
            self.df_data = self._task_filter_data_()
        
    def _task_filter_data_(self):
        #COMMENT Adpated to include gender and age features
        # v1 # df_task1 = pd.read_csv(PREDICTIONS_PATH + '/' + self.model + '_' + 'ZeroShotTask1' + '.tsv', sep='\t', index_col='id_EXIST')
        # v2 # task1_file = [file for file in os.listdir(PREDICTIONS_PATH) if all(features in file for features in [self.model, self.gender, self.age, 'ZeroShotTask1', 'processed'])][0]
        # v2 # df_task1 = pd.read_csv(PREDICTIONS_PATH + '/' + task1_file, sep='\t', index_col='id_EXIST')
        
        # v3
        df_task1 = pd.read_csv(PREDICTIONS_PATH + '/' + self.model + '_' + self.gender + '_' + self.age + '_ZeroShotTask1_processed.tsv', sep='\t', index_col='id_EXIST')
        
        # return self.df_data.loc[df_task1.loc[df_task1[self.model]=='YES'].index.tolist()]
            # Get the indices where the model prediction is 'YES'
        yes_indices = df_task1.loc[df_task1[self.model] == 'YES'].index
        
        # Find the intersection of indices between self.df_data and yes_indices
        common_indices = self.df_data.index.intersection(yes_indices)
        
        # Return the filtered DataFrame
        return self.df_data.loc[common_indices]
    

class ChatLLM(DATA, PROMPT):
    def __init__(self, model, prompt_type, gender, age, demographics_text='', max_tokens=50):
        super().__init__()
        self.prompt_type = prompt_type
        self.model = model
        self.max_tokens = max_tokens
        self.demographics = demographics_text
        self.gender = gender
        self.age = age
        

    def get_predictions(self):
        predictions = []
        
        for i in tqdm(range(len(self.df_data)), desc="Predicting Sexism", leave=False, position=4, ncols=100):  # Add tqdm to the loop
            tweet = self.df_data.iloc[i]['tweet']
            prompt = super().get_prompt(self.prompt_type, tweet, self.gender, self.age)
            
            #DEBUG:
            # tqdm.write('\n')
            # tqdm.write(prompt)
            # exit() 
            
            llms_api = llmsAPI(prompt)
            
            while True:
                try:
                    output = llms_api.get_completion(llm=self.model, max_tokens=self.max_tokens)
                    break
                except Exception as e:
                    if "429" in str(e):
                        logging.warning("Rate limit exceeded. Retrying after delay...")
                        time.sleep(10)  # Wait for 10 seconds before retrying
                    elif "400" in str(e) and "content_filter" in str(e):
                        logging.error(f"Content filter triggered. Skipping tweet: {tweet}")
                        output = "FILTERED"
                        break
                    else:
                        raise e
            
            
            # #DEBUG:
            # print('\n')
            # print('######### PROMPT #########')
            # print(prompt)
            # print('\n')
            # print('######### OUTPUT #########')
            # print(output)
            # print('\n')
            # exit()
            
            predictions.append(output)

        self.df_data[self.model] = predictions
        
        
    def save_predictions(self):
        self.df_data.loc[:,['lang', self.model]].to_csv(PREDICTIONS_PATH + '/' + self.model + '_' + self.gender + '_' + self.age + '_' + self.prompt_type + '.tsv', sep='\t')
    
    def main(self):
        super().load_data()
        self.get_predictions()
        self.save_predictions()
        


if __name__ == '__main__':
    for llm in tqdm(LLMS, desc="LLMs", position=3,ncols=100):
        for prompt in tqdm(["ZeroShotTask1_o1_Preview"], desc="Prompts", position=2, ncols=100):
            
            # Demographics 
            for gender in tqdm([ "", "female", "male"], desc="Genders", position=1, ncols=100):
                for age in tqdm(["18-22", "23-45", "46+", ""], desc="Ages", position=0, ncols=100):
            # for gender in tqdm(["female", "male"], desc="Genders", position=1, ncols=100):
            #     for age in tqdm(["18-22", "23-45", "46+"], desc="Ages", position=0, ncols=100):
                
                    # if gender == "" and age == "":
                    #     break
                    # if gender != "" and age != "":
                    #     continue

                    LlmPreds = ChatLLM(
                                model=llm, 
                                prompt_type=prompt,
                                demographics_text='',
                                gender=gender,
                                age=age,
                                max_tokens=50)
                    LlmPreds.main()


    #DEBUG:
    # for llm in tqdm(["gpt-4-turbo-2024-04-09"], desc="LLMs", position=0,ncols=100):
    #     for prompt in tqdm(["ZeroShotTask3"], desc="Prompts", position=1, ncols=100):
        
    #         LlmPreds = ChatLLM(
    #                             model=llm, 
    #                             prompt_type=prompt,
    #                             max_tokens=50)
    #         LlmPreds.main()
        