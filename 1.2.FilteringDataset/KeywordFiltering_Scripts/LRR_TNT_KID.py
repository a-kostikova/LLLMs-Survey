import sys
import numpy as np
import json
import os

def read_keywords_data(filename):
    term_frequency = {}
    document_frequency = {}
    number_of_documents = 0

    with open(filename, 'r') as file:
        papers = json.load(file)

    for paper in papers:
        keywords = paper['keywords']  # Extracting keywords from the TNT-KID 'keywords' field
        current_keywords = set(keywords)

        for keyword in current_keywords:
            term_frequency[keyword] = term_frequency.get(keyword, 0) + 1
            document_frequency[keyword] = document_frequency.get(keyword, 0) + 1

        number_of_documents += 1

    return term_frequency, document_frequency, number_of_documents

def calculate_llr(accepted_terms, rejected_terms, accepted_doc_freq, rejected_doc_freq, total_accepted, total_rejected):
    llr_scores = {}
    for term in set(accepted_terms).union(set(rejected_terms)):
        if accepted_doc_freq.get(term, 0) < 7:
            continue

        accepted_count = accepted_terms.get(term, 0.5)
        rejected_count = rejected_terms.get(term, 0.5)

        total_terms = total_accepted + total_rejected
        expected_accepted = total_accepted * (accepted_count + rejected_count) / total_terms
        expected_rejected = total_rejected * (accepted_count + rejected_count) / total_terms

        llr_value = 2 * (accepted_count * np.log((accepted_count + 0.5) / expected_accepted) +
                         rejected_count * np.log((rejected_count + 0.5) / expected_rejected))
        if accepted_count / total_accepted > rejected_count / total_rejected:
            sign = 1
        else:
            sign = -1
        llr_scores[term] = (sign * llr_value, accepted_count, rejected_count)
    return llr_scores

def main(input_files, output_filename, top_k):
    accepted_terms, accepted_doc_freq, accepted_count = read_keywords_data(input_files[0])
    rejected_terms, rejected_doc_freq, rejected_count = read_keywords_data(input_files[1])

    total_accepted = sum(accepted_terms.values())
    total_rejected = sum(rejected_terms.values())

    llr_scores = calculate_llr(accepted_terms, rejected_terms, accepted_doc_freq, rejected_doc_freq, total_accepted, total_rejected)

    sorted_terms = sorted(llr_scores.items(), key=lambda x: x[1][0], reverse=True)

    # Write results to a file
    with open(output_filename, 'w') as f:
        for term, value in sorted_terms[:top_k]:
            f.write(f'{term}: {value}\n')

    print(f"Results have been saved in {output_filename}")

if __name__ == "__main__":
    input_files = [sys.argv[1], sys.argv[2]]
    output_directory = os.path.dirname(input_files[0])
    output_filename = os.path.join(output_directory, 'combined_keywords.txt')
    top_k = 10000
    main(input_files, output_filename, top_k)