import csv
import torch
import json
import time
import re
import pandas as pd
from transformers import pipeline

input_csv_path = '3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv'
prompts_file_path = "4.ModelsEvaluation/1.0.PromptExperimnts/prompts.json"
output_csv_template = "4.ModelsEvaluation/1.1.PromptExperimentsResults/Llama/FullGS458_Llama-3.1-70B_Prompt{}.csv"

def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        prompts = json.load(file)
    return prompts

# Initialize the Llama model pipeline
model_id = "meta-llama/Llama-3.1-70B-Instruct"
text_generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    token=""  # Use your token
)

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

    eos_token_id = text_generator.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID is undefined, please check the tokenizer initialization.")

    outputs = text_generator(
        prompt,
        max_new_tokens=100,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"]
    result_text = generated_text[len(prompt):] if len(generated_text) > len(prompt) else ""
    print(f"Model Output:\n{result_text.strip()}\n")
    return result_text.strip()

def evaluate_paper(title, summary, prompt_text):
    instruction = "You are a helpful assistant."
    formatted_prompt = prompt_text.format(title=title, summary=summary)
    try:
        response = generate_response(instruction, formatted_prompt)
        return response
    except Exception as e:
        print(f"Error during model call: {e}")
        return None

def extract_fields_from_response(response):
    patterns = {
        "talks_about_llms": r"Does it talk about LLMs:\s*(Yes|No)",
        "rate": r"Rate Limitations of LLMs:\s*(\d+)",
        "evidence": r"Evidence:\s*(.*)"
    }
    fields = {"talks_about_llms": "not extracted", "rate": "not extracted", "evidence": "not extracted"}

    if response:
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                fields[key] = match.group(1).strip()
    return fields

def process_papers_for_all_prompts():
    prompts = load_prompts(prompts_file_path)
    papers_df = pd.read_csv(input_csv_path, delimiter=";", dtype=str)
    for prompt_num in range(1, 4):
        prompt_key = f"prompt{prompt_num}"
        if prompt_key not in prompts:
            print(f"Warning: {prompt_key} not found in the prompt file. Skipping...")
            continue

        prompt_text = prompts[prompt_key]
        output_csv_file = output_csv_template.format(prompt_num)

        papers_df[f'Rate_Llama-3.1-70b_Prompt{prompt_num}'] = ""
        papers_df[f'Talks_about_LLMs_Llama-3.1-70b_Prompt{prompt_num}'] = ""
        papers_df[f'Evidence_Llama-3.1-70b_Prompt{prompt_num}'] = ""
        papers_df[f'Full_response_rate_Llama-3.1-70b_Prompt{prompt_num}'] = ""

        for index, row in papers_df.iterrows():
            title = row.get("Title", "No Title")
            summary = row.get("Abstract", "No Summary")

            response = evaluate_paper(title, summary, prompt_text)
            extracted_fields = extract_fields_from_response(response)
            papers_df.at[index, f'Rate_Llama-3.1-70b_Prompt{prompt_num}'] = extracted_fields["rate"]
            papers_df.at[index, f'Talks_about_LLMs_Llama-3.1-70b_Prompt{prompt_num}'] = extracted_fields["talks_about_llms"]
            papers_df.at[index, f'Evidence_Llama-3.1-70b_Prompt{prompt_num}'] = extracted_fields["evidence"]
            papers_df.at[index, f'Full_response_rate_Llama-3.1-70b_Prompt{prompt_num}'] = response

            print(f"Processed: {title} - Rate: {extracted_fields['rate']} - Talks about LLMs: {extracted_fields['talks_about_llms']}")
            time.sleep(1)

        papers_df.to_csv(output_csv_file, index=False, sep=";")
        print(f"CSV file saved: {output_csv_file}")

if __name__ == "__main__":
    process_papers_for_all_prompts()

