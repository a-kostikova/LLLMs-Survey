import json
import time
import os
import re
from openai import OpenAI

client = OpenAI(
    api_key=""
)

input_file = 'ACL_3-4-5-PapersKeyphrases.json'
output_file = 'arxiv_3-4-5-PapersKeyphrases.json'


def generate_keyphrases(evidence_text):
    """
    Generates keyphrases for the given evidence text using GPT-4.
    """
    prompt = f"""
    I am trying to cluster evidence from research papers that discuss the limitations of large language models (LLMs).
    For each evidence text, provide a comprehensive set of keyphrases that describe the limitations mentioned in the text.
    These keyphrases should emphasize problems or challenges of LLMs. Generate the set of keyphrases as a JSON-formatted list.

    Evidence: "LLMs struggle to handle domain-specific tasks due to limited training on diverse datasets."
    Keyphrases: ["domain-specific limitations", "training data diversity", "task-specific challenges"]

    Evidence: "They exhibit significant biases due to uneven representation in the training data."
    Keyphrases: ["bias in training", "uneven data representation", "bias issues"]

    Evidence: "Scaling LLMs leads to increasing compute and energy requirements, making them less sustainable."
    Keyphrases: ["compute scaling issues", "energy inefficiency", "sustainability challenges"]

    Evidence: "{evidence_text}"
    Keyphrases:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5
        )
        keyphrase_response = response.choices[0].message.content.strip()
        print("GPT Response:", keyphrase_response)
        return keyphrase_response
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Error"

def extract_fields_from_response(response):
    fields = {"keywords": "not extracted"}
    try:
        # Strip Markdown block if present
        response_clean = re.sub(r"^```json\n|\n```$", "", response.strip())
        parsed_response = json.loads(response_clean)
        if isinstance(parsed_response, list):
            fields["keywords"] = parsed_response
    except json.JSONDecodeError:
        print("Error parsing response as JSON:", response)
    return fields

if os.path.exists(output_file):
    print(f"Resuming from last saved progress: {output_file}")
    with open(output_file, 'r') as f:
        data = json.load(f)
else:
    print(f"Starting fresh from input file: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

start_index = next((i for i, item in enumerate(data)
                    if 'Keyphrases' not in item or item.get('Keyphrases') == "Error"), len(data))

for i in range(start_index, len(data)):
    item = data[i]
    evidence_text = item.get('Evidence_Llama-3.1-70b', '')

    if evidence_text and (item.get('Keyphrases') == "Error" or 'Keyphrases' not in item):
        keyphrases = generate_keyphrases(evidence_text)
        item['Keyphrases'] = keyphrases
        extracted_fields = extract_fields_from_response(keyphrases)
        item.update(extracted_fields)
        print(f"Processed {i + 1}/{len(data)}: {evidence_text[:50]}... -> Keyphrases: {keyphrases}")

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

        time.sleep(1)

print(f"Keyphrases extraction completed. Output saved to {output_file}")
