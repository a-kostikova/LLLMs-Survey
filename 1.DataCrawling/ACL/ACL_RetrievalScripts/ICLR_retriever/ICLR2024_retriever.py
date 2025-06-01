import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm

def fetch_all_iclr2024_posters(start=17365, end=19626):  # poster IDs on the official ICLR 2024 virtual site
    base_url = "https://iclr.cc/virtual/2024/poster/"
    all_papers = []

    for poster_id in tqdm(range(start, end), desc="Fetching ICLR 2024 posters"):
        url = f"{base_url}{poster_id}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            script_tag = soup.find('script', type='application/ld+json')
            if not script_tag:
                continue

            data = json.loads(script_tag.string)
            title = data.get("name", "No title found")
            authors = [author.get("name", "Unknown") for author in data.get("author", [])]

            # Extract abstract
            abstract_tag = soup.find('meta', {'name': 'citation_abstract'})
            if abstract_tag and abstract_tag.get('content'):
                abstract = abstract_tag['content'].strip()
            else:
                div_tag = soup.find('div', id='abstractExample')
                abstract = div_tag.get_text(strip=True) if div_tag else "No abstract found"

            paper = {
                "title": title,
                "authors": authors,
                "summary": abstract,
                "pdf_link": url,
                "source": "iclr2024"
            }

            all_papers.append(paper)

        except Exception as e:
            print(f"Error at {url}: {e}")
            continue

    return all_papers

def save_papers_to_json(papers, filename="iclr2024.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(papers)} entries to {filename}")

posters = fetch_all_iclr2024_posters()
print(f"\nTotal ICLR 2024 posters collected: {len(posters)}")
save_papers_to_json(posters)
