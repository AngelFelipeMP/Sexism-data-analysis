import torch
import pandas as pd
from transformers import pipeline

def zero_shot_paraphrase(llm_model, question, num_paraphrases=10):
    paraphraser = pipeline("text2text-generation", model=llm_model, tokenizer=llm_model)
    paraphrases = paraphraser(question, num_return_sequences=num_paraphrases)

    paraphrase_list = []
    for item in paraphrases:
        paraphrase_list.append(item['generated_text'].strip())

    return paraphrase_list

def main():
    # Replace 'MODEL_NAME' with the specific LLM model you want to use,
    # e.g., 'facebook/bart-large-mnli', 't5-small', 't5-base', etc.
    model_name = 'MODEL_NAME'
    llm_model = torch.hub.load('huggingface/transformers', model_name)

    input_file = 'input.csv'
    output_file = 'output.csv'

    # Load the CSV file with questions (assuming the column name is 'Questions')
    df = pd.read_csv(input_file)

    paraphrased_questions = []
    for question in df['Questions']:
        paraphrases = zero_shot_paraphrase(llm_model, question)
        paraphrased_questions.append(paraphrases)

    # Create a new DataFrame to store the paraphrases
    df_output = pd.DataFrame(paraphrased_questions)
    df_output.to_csv(output_file, index=False, header=False)

if __name__ == "__main__":
    main()



import torch
import pandas as pd
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModel, AutoConfig

def paraphrase_question(question, tokenizer, model, num_paraphrases=10):
    # Modify the prompt to guide the paraphrasing process (if applicable for the selected model)
    prompt = f"paraphrase: {question} </s>"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    input_ids = inputs.input_ids

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.repeat(num_paraphrases, 1),
            max_length=50,  # Adjust the maximum length of paraphrases as needed
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            num_return_sequences=num_paraphrases
        )

    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrases

def main(input_csv, output_csv):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    df = pd.read_csv(input_csv)

    paraphrased_data = []
    for _, row in df.iterrows():
        question = row[0]
        paraphrases = paraphrase_question(question, tokenizer, model, num_paraphrases=10)
        paraphrased_data.append(paraphrases)

    paraphrased_df = pd.DataFrame(paraphrased_data)
    paraphrased_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv_file = "input_questions.csv"  # Replace with the name of your input CSV file
    output_csv_file = "paraphrased_questions.csv"  # Replace with the desired output CSV file name
    main(input_csv_file, output_csv_file)
