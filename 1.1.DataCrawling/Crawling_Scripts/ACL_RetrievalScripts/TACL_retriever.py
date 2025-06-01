import json
from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm
from urllib.parse import urljoin

conference = 'tacl'
year = 2022 # 2022, 2023, 2024

conference_url = f"https://aclanthology.org/events/{conference}-{year}/"

html_doc = requests.get(conference_url).text
soup = BeautifulSoup(html_doc, 'html.parser')

tacl_pattern = rf"/{year}\.{conference}-\d+\.\d+/$"
main_papers = soup.find_all('a', href=re.compile(tacl_pattern))

print(f"Found {len(main_papers)} papers.")

final = []

base_url = 'https://aclanthology.org/'

for paper in tqdm(main_papers, desc=f"{conference}{year}"):
    link = urljoin(base_url, paper['href'])

    print(f"Fetching paper: {link}")
    try:
        paper_html = requests.get(link).text
        tmp_soup = BeautifulSoup(paper_html, 'html.parser')

        title_elem = tmp_soup.find('h2', id='title')
        abstract_elem = tmp_soup.find('div', class_="card-body acl-abstract").find('span')
        authors_elem = tmp_soup.find('p', class_='lead')
        published_elem = tmp_soup.find('dt', text='Year:').find_next_sibling('dd')

        #print(f"Title element: {title_elem}")
        #print(f"Abstract element: {abstract_elem}")
        #print(f"Authors element: {authors_elem}")
        #print(f"Published element: {published_elem}")

        if title_elem and abstract_elem and authors_elem and published_elem:
            title = title_elem.get_text().strip()
            abstract = abstract_elem.get_text().strip()
            authors = [author.get_text().strip() for author in authors_elem.find_all('a')]
            published = published_elem.get_text().strip()
            pdf_link = tmp_soup.find('meta', {'name': 'citation_pdf_url'})['content']

            paper_info = {
                "title": title,
                "authors": authors,
                "published": published,
                "summary": abstract,
                "pdf_link": pdf_link,
                "source": "tacl2024"
            }
            final.append(paper_info)
        else:
            print(f"Missing information for paper: {link}")
    except Exception as e:
        print(f'Error fetching paper: {link}')
        print(e)
print(f"Collected {len(final)} papers.")

json_path = f'{conference}{year}.json'
with open(json_path, 'w', encoding='utf-8') as jsonf:
    json.dump(final, jsonf, ensure_ascii=False, indent=4)