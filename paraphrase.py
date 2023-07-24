import pandas as pd
from transformers import pipeline, LlamaModel, LlamaTokenizer, LlamaConfig

def zero_shot_paraphrase(llm_model, question, llm_tokenizer, num_paraphrases=10):
    paraphraser = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer)
    prompt = f"paraphrase: {question} </s>"
    paraphrases = paraphraser(prompt, num_return_sequences=num_paraphrases)

    paraphrase_list = [item['generated_text'].strip() for item in paraphrases]
    return paraphrase_list

def main():
    # Replace 'MODEL_NAME' with the specific LLM model you want to use,
    # e.g., 'facebook/bart-large-mnli', 't5-small', 't5-base', etc.
    configuration = LlamaConfig()
    model_name = '/home/ubuntu/LLaMA-QA-Rephrase/ssd-volume/llama-7B'
    llm_model = LlamaModel.from_pretrained(model_name)
    llm_tokenizer = LlamaTokenizer.from_pretrained(model_name)

    input_file = '/home/ubuntu/LLaMA-QA-Rephrase/ssd-volume/FAQ.csv'
    output_file = '/home/ubuntu/LLaMA-QA-Rephrase/ebs-volume/paraphrase.csv'

    # Load the CSV file with questions (assuming the column name is 'Questions')
    df = pd.read_csv(input_file)

    paraphrased_questions = []
    for question in df['Questions']:
        paraphrases = zero_shot_paraphrase(llm_model, question, llm_tokenizer)
        paraphrased_questions.append(paraphrases)

    # Create a new DataFrame to store the paraphrases
    df_output = pd.DataFrame(paraphrased_questions)
    df_output.to_csv(output_file, index=False, header=False)

if __name__ == "__main__":
    main()

# python /home/ubuntu/LLaMA-QA-Rephrase/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py     --input_dir /home/ubuntu/LLaMA-QA-Rephrase/ssd-volume/llama-7B --model_size 7B --output_dir /home/ubuntu/LLaMA-QA-Rephrase/ssd-volume


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py