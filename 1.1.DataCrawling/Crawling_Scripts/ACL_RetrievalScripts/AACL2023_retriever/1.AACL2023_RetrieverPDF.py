import json
from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
from urllib.parse import urljoin
import os

# Corrected URL for IJCNLP-AACL 2023
conferences = {
    "aacl2023": "https://aclanthology.org/events/ijcnlp-2023/"
}

# Only allow these ID prefixes
ALLOWED_TRACKS = ['ijcnlp-main', 'ijcnlp-short', 'findings-ijcnlp']

def download_pdf(pdf_url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    response = requests.get(pdf_url)
    filename = pdf_url.split('/')[-1]
    file_path = os.path.join(folder, filename)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} to {folder}")
    else:
        print(f"Failed to download {filename}")

def fetch_papers(conference, year):
    conference_key = f"{conference.lower()}{year}"
    if conference_key not in conferences:
        print(f"Conference {conference} {year} is not in the list.")
        return []

    conference_url = conferences[conference_key]
    html_doc = requests.get(conference_url).text
    soup = BeautifulSoup(html_doc, 'html.parser')

    # Match only desired tracks
    pattern = rf"/{year}\.(?:{'|'.join(ALLOWED_TRACKS)})\.\d+/$"
    main_papers = soup.find_all('a', href=re.compile(pattern))

    print(f"Found {len(main_papers)} papers for {conference} {year}.")

    final = []
    base_url = 'https://aclanthology.org/'

    for paper in tqdm(main_papers, desc=f"{conference}{year}"):
        link = urljoin(base_url, paper['href'])
        print(f"Fetching paper: {link}")
        try:
            paper_html = requests.get(link).text
            tmp_soup = BeautifulSoup(paper_html, 'html.parser')

            title_elem = tmp_soup.find('h2', id='title')
            authors_elem = tmp_soup.find('p', class_='lead')
            year_elem = tmp_soup.find('dt', string='Year:')
            published_elem = year_elem.find_next_sibling('dd') if year_elem else None
            pdf_link_elem = tmp_soup.find('meta', {'name': 'citation_pdf_url'})

            title = title_elem.get_text().strip() if title_elem else 'No title available'
            authors = [author.get_text().strip() for author in authors_elem.find_all('a')] if authors_elem else ['No authors listed']
            published = published_elem.get_text().strip() if published_elem else 'No publication year available'
            pdf_link = pdf_link_elem['content'] if pdf_link_elem else 'No PDF link available'

            paper_info = {
                "title": title,
                "authors": authors,
                "published": published,
                "pdf_link": pdf_link
            }
            final.append(paper_info)

        except Exception as e:
            print(f'Error fetching paper: {link}')
            print(e)

    print(f"Collected {len(final)} papers for {conference} {year}.")
    return final

def save_papers_to_json_and_download_pdfs(conference, year):
    papers = fetch_papers(conference, year)
    json_path = f'1.Data_crawling/acl_data/Data/{conference.lower()}{year}.json'
    pdf_folder = f'1.Data_crawling/acl_data/Scripts/AACL2023_retreiever/{conference.lower()}{year}_pdfs'  # Folder to save PDFs

    with open(json_path, 'w', encoding='utf-8') as jsonf:
        json.dump(papers, jsonf, ensure_ascii=False, indent=4)

    for paper in papers:
        if paper["pdf_link"] != 'No PDF link available':
            download_pdf(paper["pdf_link"], pdf_folder)

    print(f"Data successfully saved to {json_path} and PDFs downloaded to {pdf_folder}.")

# Example usage
save_papers_to_json_and_download_pdfs('AACL', '2023')
