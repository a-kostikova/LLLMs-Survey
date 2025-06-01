import json
import re
import os

def load_keywords(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [key for key, value in data.items() if value['included']]

def compile_keyword_pattern(keywords):
    pattern = '|'.join([re.escape(keyword).replace(r'\ ', r'[\s-]*') for keyword in keywords])
    return re.compile(r'\b(?:' + pattern + r')\b', re.IGNORECASE)

def contains_keyword(paper, keyword_pattern):
    title = paper.get('title', '').replace('\n', ' ').replace('\r', ' ').strip()
    summary = paper.get('summary', '').replace('\n', ' ').replace('\r', ' ').strip()
    text = title + " " + summary
    matches = keyword_pattern.findall(text)
    return len(set(matches)) >= 1

def save_filtered_papers(papers, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)

def process_json_files(input_dir, keyword_pattern, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename.replace('.json', '_filtered.json'))
            with open(input_file_path, encoding='utf-8') as f:
                papers = json.load(f)
            filtered_papers = [paper for paper in papers if contains_keyword(paper, keyword_pattern)]
            save_filtered_papers(filtered_papers, output_file_path)
            print(f"Processed {filename} → {output_file_path} ({len(filtered_papers)} papers)")

def main():
    keyword_json = '2.1.ExtractingKeywords/keywords_metadata.json'
    keywords = load_keywords(keyword_json)
    keyword_pattern = compile_keyword_pattern(keywords)

    inputs_outputs = [
        ('1.Data_crawling/acl_data/ACL_Data', '2.Filtering_dataset/acl_filtered_data'),
        ('1.Data_crawling/arXiv_data', '2.Filtering_dataset/arxiv_filtered_data')
    ]

    for input_dir, output_dir in inputs_outputs:
        process_json_files(input_dir, keyword_pattern, output_dir)

if __name__ == "__main__":
    main()
