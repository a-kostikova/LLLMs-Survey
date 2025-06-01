import json
from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
from urllib.parse import urljoin

conferences = {
    "acl2022": "https://aclanthology.org/events/acl-2022/",
    "naacl2022": "https://aclanthology.org/events/naacl-2022/",
    "emnlp2022": "https://aclanthology.org/events/emnlp-2022/",
    "aacl2022": "https://aclanthology.org/events/aacl-2022/",
    "eacl2023": "https://aclanthology.org/events/eacl-2023/",
    "acl2023": "https://aclanthology.org/events/acl-2023/",
    "emnlp2023": "https://aclanthology.org/events/emnlp-2023/",
    "naacl2024": "https://aclanthology.org/events/naacl-2024/",
    "acl2024": "https://aclanthology.org/events/acl-2024/",
    "eacl2024": "https://aclanthology.org/events/eacl-2024/",
    "emnlp2024": "https://aclanthology.org/events/emnlp-2024/",
    # "aacl2023": "https://aclanthology.org/events/ijcnlp-2023/" # Processed separately. See 1.Data_crawling/acl_data/Scripts/AACL2023_retreiever
}

def fetch_papers(conference, year):
    conference_key = f"{conference.lower()}{year}"
    if conference_key not in conferences:
        print(f"Conference {conference} {year} is not in the list.")
        return []

    conference_url = conferences[conference_key]
    html_doc = requests.get(conference_url).text
    soup = BeautifulSoup(html_doc, 'html.parser')

    if "openreview.net" in conference_url:
        main_papers = soup.find_all('a', href=re.compile(r'/forum\?id=[^&]+$'))
    else:
        excluded_tracks = ['srw', 'demo', 'demos', 'industry', 'tutorial']
        pattern = rf"/{year}\.(findings-)?{conference.lower()}-([a-z]+)\.\d+/$"
        matches = soup.find_all('a', href=re.compile(pattern))

        # Filter out excluded tracks
        main_papers = [
            tag for tag in matches
            if re.match(pattern, tag['href']) and
               re.match(pattern, tag['href']).group(2) not in excluded_tracks
        ]

    print(f"Found {len(main_papers)} papers for {conference} {year}.")

    final = []
    base_url = 'https://aclanthology.org/'

    for paper in tqdm(main_papers, desc=f"{conference}{year}"):
        link = urljoin(base_url, paper['href'])
        try:
            paper_html = requests.get(link).text
            tmp_soup = BeautifulSoup(paper_html, 'html.parser')

            title_elem = tmp_soup.find('h2', id='title')
            abstract_container = tmp_soup.find('div', class_="acl-abstract")
            abstract_elem = abstract_container.find('span') if abstract_container else None
            authors_elem = tmp_soup.find('p', class_='lead')
            year_elem = tmp_soup.find('dt', text='Year:')
            published_elem = year_elem.find_next_sibling('dd') if year_elem else None
            pdf_link_elem = tmp_soup.find('meta', {'name': 'citation_pdf_url'})

            title = title_elem.get_text().strip() if title_elem else 'No title available'
            abstract = abstract_elem.get_text().strip() if abstract_elem else 'No abstract available'
            authors = [author.get_text().strip() for author in authors_elem.find_all('a')] if authors_elem else ['No authors listed']
            published = published_elem.get_text().strip() if published_elem else 'No publication year available'
            pdf_link = pdf_link_elem['content'] if pdf_link_elem else 'No PDF link available'

            paper_info = {
                "title": title,
                "authors": authors,
                "published": published,
                "summary": abstract,
                "pdf_link": pdf_link,
                "source": f"{conference.lower()}{year}"
            }
            final.append(paper_info)

        except Exception as e:
            print(f'Error fetching paper: {link}')
            print(e)

    print(f"Collected {len(final)} papers for {conference} {year}.")

    return final

def save_papers_to_json(conference, year):
    papers = fetch_papers(conference, year)
    json_path = f'{conference.lower()}{year}.json'
    with open(json_path, 'w', encoding='utf-8') as jsonf:
        json.dump(papers, jsonf, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {json_path}.")

# Example usage
for conference, year in [#('ACL', 2022), ('ACL', 2023), ('ACL', 2024),
                        #('NAACL', 2022), ('NAACL', 2024),
                       #('EMNLP', 2022), ('EMNLP', 2023), ('EMNLP', 2024),
                        #('AACL', 2022),
                        #('EACL', 2023), ('EACL', 2024)
                         ]:
    save_papers_to_json(conference, year)
