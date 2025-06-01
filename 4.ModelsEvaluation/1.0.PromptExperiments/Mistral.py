import pandas as pd
import json
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm

input_csv_file = '3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv'
prompts_file_path = "4.ModelsEvaluation/1.0.PromptExperimnts/prompts.json"

output_csv_template = "4.ModelsEvaluation/1.1.PromptExperimentsResults/Mistral/FullGS445_Mistral-7B_Prompt{}.csv"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
text_generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=AutoTokenizer.from_pretrained(model_id),
    torch_dtype=torch.float16,
    device_map="auto"
)

def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        prompts_text = file.read()
    prompts = json.loads(prompts_text)
    return prompts

def read_csv(input_csv, delimiter=";"):
    return pd.read_csv(input_csv, delimiter=delimiter, dtype=str)

def write_csv(dataframe, output_csv):
    dataframe.to_csv(output_csv, index=False)
    print(f"CSV file saved: {output_csv}")

def generate_response(instruction, text):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]

    prompt = text_generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = text_generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"]
    result_text = generated_text[len(prompt):] if len(generated_text) > len(prompt) else ""
    return result_text.strip()

def evaluate_paper(title, abstract, prompt_text):
    formatted_prompt = prompt_text.format(title=title, summary=abstract)
    try:
        response = generate_response("You are a helpful assistant.", formatted_prompt)
        return response.strip()
    except Exception as e:
        print(f"Error during model call: {e}")
        return None

def extract_label_from_response(response):
    label_marker = "Label:"
    start_index = response.find(label_marker)

    if start_index != -1:
        start_index += len(label_marker)
        label = response[start_index:].strip().split()[0]
        return label.lower() if label in ["yes", "no"] else "not extracted"

    return "not extracted"

def process_papers_for_all_prompts():
    prompts = load_prompts(prompts_file_path)
    papers_df = read_csv(input_csv_file)

    for prompt_num in range(1, 4):
        prompt_key = f"prompt{prompt_num}"
        if prompt_key not in prompts:
            print(f"Warning: {prompt_key} not found in the prompt file. Skipping...")
            continue

        prompt_text = prompts[prompt_key]
        output_csv_file = output_csv_template.format(prompt_num)

        papers_df[f"Full_Model_Response_Prompt{prompt_num}"] = ""
        papers_df[f"Mentions LLM Limitations_Prompt{prompt_num}"] = ""

        for index, row in tqdm(papers_df.iterrows(), total=papers_df.shape[0], desc=f"Processing Prompt {prompt_num}"):
            title = row.get("Title", "No Title")
            abstract = row.get("Abstract", "No Abstract")

            evaluation_result = evaluate_paper(title, abstract, prompt_text)

            if evaluation_result is None:
                print(f"Skipping paper at index {index} due to API call failure.")
                continue

            label = extract_label_from_response(evaluation_result)

            papers_df.at[index, f"Full_Model_Response_Prompt{prompt_num}"] = evaluation_result
            papers_df.at[index, f"Mentions LLM Limitations_Prompt{prompt_num}"] = label

        write_csv(papers_df, output_csv_file)

if __name__ == "__main__":
    process_papers_for_all_prompts()
