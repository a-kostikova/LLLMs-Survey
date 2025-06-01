import openreview
import json

def fetch_submissions(client, venue_id, relevant_venues):
    """
    Fetch all accepted submissions from OpenReview that match relevant venue tags.
    """
    submissions = client.get_all_notes(invitation=f"{venue_id}/-/Blind_Submission")
    papers = []
    year = venue_id.split('/')[1]

    for submission in submissions:
        venue = submission.content.get('venue', '').lower()

        if any(rv.lower() in venue for rv in relevant_venues):
            title = submission.content.get('title', 'No Title Available')
            authors = submission.content.get('authors', [])
            abstract = submission.content.get('abstract', 'No Abstract Available')
            pdf_link = f"https://openreview.net{submission.content.get('pdf', 'No PDF Available')}"
            forum_url = f"https://openreview.net/forum?id={submission.id}"

            paper_info = {
                "title": title,
                "authors": authors,
                "published": venue,
                "summary": abstract,
                "pdf_link": pdf_link,
                "forum_url": forum_url,
                "source": f"iclr{year}"
            }
            papers.append(paper_info)

    return papers


def save_combined_json(papers, year):
    json_path = f"iclr{year}.json"
    with open(json_path, 'w', encoding='utf-8') as jsonf:
        json.dump(papers, jsonf, ensure_ascii=False, indent=4)
    print(f"Saved {len(papers)} papers to {json_path}")


def main():
    client = openreview.Client(baseurl='https://api.openreview.net', username='', password='')

    config = {
        '2022': ["ICLR 2022 Oral", "ICLR 2022 Spotlight", "ICLR 2022 Poster"],
        '2023': ["ICLR 2023 notable top 5%", "ICLR 2023 notable top 25%", "ICLR 2023 poster"]
    }

    for year, venues in config.items():
        venue_id = f"ICLR.cc/{year}/Conference"
        all_papers = fetch_submissions(client, venue_id, venues)
        save_combined_json(all_papers, year)

main()