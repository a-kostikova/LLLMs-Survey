import json
import time
import pandas as pd
from openai import OpenAI

# OpenAI API Key
openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

input_csv_path = "3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv"
prompts_file_path = "4.ModelsEvaluation/1.0.PromptExperimnts/prompts.json"
output_csv_template = "4.ModelsEvaluation/1.1.PromptExperimentsResults/GPT/FullGS445_GPT4_Prompt{}.csv"

# Load prompts from JSON
def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def evaluate_paper_gpt4(title, summary, prompt_text):
    formatted_prompt = prompt_text.format(title=title, summary=summary)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=520,
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error"

def extract_fields_from_response(response):
    fields = {"talks_about_llms": "not extracted", "rate": "not extracted", "evidence": "not extracted"}

    if response.startswith("Does it talk about LLMs:"):
        lines = response.split("\n")
        for line in lines:
            if line.startswith("Does it talk about LLMs:"):
                fields["talks_about_llms"] = line.split(":")[1].strip()
            elif line.startswith("Rate Limitations of LLMs:"):
                fields["rate"] = line.split(":")[1].strip()
            elif line.startswith("Evidence:"):
                fields["evidence"] = line.split(":")[1].strip()

    return fields

def process_papers_for_all_prompts():

    prompts = load_prompts(prompts_file_path)
    papers_df = pd.read_csv(input_csv_path, delimiter=";", dtype=str)

    for prompt_num in range(1, 4):
        prompt_key = f"prompt{prompt_num}"
        if prompt_key not in prompts:
            print(f"Warning: {prompt_key} not found in prompts.json. Skipping...")
            continue

        prompt_text = prompts[prompt_key]
        output_csv_file = output_csv_template.format(prompt_num)

        papers_df[f'Rate_GPT4_Prompt{prompt_num}'] = ""
        papers_df[f'Talks_about_LLMs_GPT4_Prompt{prompt_num}'] = ""
        papers_df[f'Evidence_GPT4_Prompt{prompt_num}'] = ""
        papers_df[f'Full_response_GPT4_Prompt{prompt_num}'] = ""

        for index, row in papers_df.iterrows():
            title = row.get("Title", "No Title")
            summary = row.get("Abstract", "No Summary")

            response = evaluate_paper_gpt4(title, summary, prompt_text)
            extracted_fields = extract_fields_from_response(response)

            papers_df.at[index, f'Rate_GPT4_Prompt{prompt_num}'] = extracted_fields["rate"]
            papers_df.at[index, f'Talks_about_LLMs_GPT4_Prompt{prompt_num}'] = extracted_fields["talks_about_llms"]
            papers_df.at[index, f'Evidence_GPT4_Prompt{prompt_num}'] = extracted_fields["evidence"]
            papers_df.at[index, f'Full_response_GPT4_Prompt{prompt_num}'] = response

            print(f"Processed: {title} - Rate: {extracted_fields['rate']} - Talks about LLMs: {extracted_fields['talks_about_llms']}")
            time.sleep(1)

        papers_df.to_csv(output_csv_file, index=False, sep=";")
        print(f"CSV file saved: {output_csv_file}")

if __name__ == "__main__":
    process_papers_for_all_prompts()
