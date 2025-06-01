import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def extract_abstract_from_xml(xml_path):
    """Extracts the abstract from the given XML file."""
    try:
        with open(xml_path, 'r', encoding='utf-8') as xml_file:
            soup = BeautifulSoup(xml_file, 'xml')
        abstract_text = soup.find('abstract').get_text(strip=True)
        return abstract_text
    except Exception as e:
        print(f"Error reading {xml_path}: {e}")
        return "Abstract not found."

def update_json_with_abstracts(json_path, xml_folder):
    """Updates the JSON file with abstracts extracted from XML files."""
    with open(json_path, 'r', encoding='utf-8') as json_file:
        papers = json.load(json_file)

    for paper in papers:
        pdf_url = paper['pdf_link']
        pdf_filename = urlparse(pdf_url).path.split('/')[-1]
        xml_filename = pdf_filename.replace('.pdf', '.xml')
        xml_path = os.path.join(xml_folder, xml_filename)

        if os.path.exists(xml_path):
            abstract = extract_abstract_from_xml(xml_path)
            paper['abstract'] = abstract
        else:
            print(f"XML file not found for {pdf_filename}")

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(papers, json_file, indent=4, ensure_ascii=False)
    print("Updated JSON file with abstracts.")

# Example usage
json_file_path = '1.Data_crawling/acl_data/Scripts/AACL2023_retreiever/aacl2023.json'
xml_directory_path = '1.Data_crawling/acl_data/Scripts/AACL2023_retreiever/aacl2023_xmls'
update_json_with_abstracts(json_file_path, xml_directory_path)