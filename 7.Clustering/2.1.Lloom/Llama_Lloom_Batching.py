import json
import pandas as pd
import torch
import sys
from transformers import pipeline
import math
import os

batch_id = int(sys.argv[1])
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Paths
batch_file = f"batches/batch_{batch_id}.json"
output_json = os.path.join(results_dir, f"ARXIV_concept_scoring_results_llama_batch_{batch_id}.json")

with open(batch_file, "r") as f:
    data = json.load(f)

if os.path.exists(output_json):
    with open(output_json, "r") as f:
        processed_data = json.load(f)
    processed_titles = set(item.get("title") for item in processed_data if "concept_scores" in item)
    print(f"Resuming from {len(processed_titles)} already processed examples.")
else:
    processed_data = []
    processed_titles = set()

model_id = "meta-llama/Llama-3.1-70B-Instruct"
text_generator = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
    token=""
)

def generate_response(instruction, text):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]

    prompt = text_generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    eos_token_id = text_generator.tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("EOS token ID is undefined, please check the tokenizer initialization.")

    outputs = text_generator(
        prompt, max_new_tokens=500, eos_token_id=eos_token_id, do_sample=True, temperature=0.6, top_p=0.9
    )

    generated_text = outputs[0]["generated_text"]
    return generated_text[len(prompt):].strip() if len(generated_text) > len(prompt) else ""

concepts_path = "7.Clustering/2.1.Lloom/Llama_Lloom_ACL/acl_cluster_prompts.json"
with open(concepts_path, "r", encoding="utf-8") as f:
    concepts = json.load(f)

group_sizes = [5, 5, 5]

def split_into_variable_groups(concepts, group_sizes):
    groups = []
    start_idx = 0
    for size in group_sizes:
        groups.append(concepts[start_idx:start_idx + size])
        start_idx += size
    return groups

concept_groups = split_into_variable_groups(concepts, group_sizes)

for idx, group in enumerate(concept_groups):
    print(f"Group {idx+1}: {[c['name'] for c in group]}")

score_highlight_prompt_template = """CONTEXT: 
    I have the following text example:
    "{example_text}"

TASK:
    Evaluate the following **patterns** strictly:
{concept_blocks}

    - ONLY assign a pattern if the example **clearly** and **primarily** matches it.
    - If the example **contains multiple ideas**, choose the **most dominant** limitation.
    - Do **NOT** assign a pattern if the connection is **vague, indirect, or secondary**.
    - If a pattern applies **partially**, but is not the main focus, choose: **C: Neither agree nor disagree**.

    For **each pattern**, provide:
    - A 1-sentence **RATIONALE** explaining why the example **directly** aligns with the pattern. If unsure, lean toward exclusion.
    - An **ANSWER** (choose ONE letter only):  
        - A: Strongly agree (The limitation is the **main focus** of the example)  
        - B: Agree (The limitation is **heavily present**, but might share space with others)  
        - C: Neither agree nor disagree (The limitation is **weakly present** or **only indirectly implied**)  
        - D: Disagree (The limitation is **mostly irrelevant** to this example)  
        - E: Strongly disagree (The limitation does **not apply at all**)  
    - A **QUOTE** from the example that illustrates this pattern.

    Respond in JSON format, using the concept name as the key:
    {{
        "Concept Name 1": {{
            "rationale": "<rationale>",
            "answer": "<answer>",
            "quote": "<quote>"
        }},
        "Concept Name 2": {{
            "rationale": "<rationale>",
            "answer": "<answer>",
            "quote": "<quote>"
        }},
        ...
    }}
"""

for idx, example in enumerate(data):
    title = example.get("title", "")

    if title in processed_titles:
        continue

    example_text = example.get("summary", "")
    if not example_text.strip():
        continue

    example["concept_scores"] = {}

    for group in concept_groups:
        concept_blocks = "\n".join([
            f'    - "{concept["name"]}": "{concept["prompt"]}"' for concept in group
        ])

        prompt = score_highlight_prompt_template.format(
            example_text=example_text, concept_blocks=concept_blocks
        )

        response_text = generate_response("You are a helpful assistant for text evaluation.", prompt)

        try:
            response_json = json.loads(response_text)

            for concept in group:
                concept_name = concept["name"]

                if concept_name in response_json:
                    example["concept_scores"][concept_name] = {
                        "rationale": response_json[concept_name].get("rationale", "No rationale provided."),
                        "answer": response_json[concept_name].get("answer", "N/A"),
                        "quote": response_json[concept_name].get("quote", "No quote provided.")
                    }
                else:
                    example["concept_scores"][concept_name] = {
                        "rationale": "Missing response.",
                        "answer": "N/A",
                        "quote": "N/A"
                    }

        except json.JSONDecodeError:
            for concept in group:
                example["concept_scores"][concept["name"]] = {
                    "rationale": "Error parsing response.",
                    "answer": "N/A",
                    "quote": "N/A"
                }

    processed_data.append(example)

    if len(processed_data) % 10 == 0 or idx == len(data) - 1:
        with open(output_json, "w") as f:
            json.dump(processed_data, f, indent=4)

        print(f"Incremental save completed at index {idx}")

print(f"\nFinal results saved to {output_json}")
