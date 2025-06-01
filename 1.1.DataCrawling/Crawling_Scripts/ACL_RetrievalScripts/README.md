# ACL Data Retrieval Scripts

This folder contains scripts for collecting paper metadata (title, authors, abstract, PDF link, etc.) from ACL conferences. Most data comes from [ACL Anthology](https://aclanthology.org/), but some exceptions (like ICLR and AACL 2023) require alternative methods.

---

## Structure Overview

```text
📁 ACL_RetrievalScripts/
├── ACL_retriever.py           ← General script for ACL/EMNLP/NAACL/EACL
├── TACL_retriever.py          ← Custom script for TACL journals
├── 📁 AACL2023_retriever/     ← Multi-step pipeline using GROBID (PDF-based abstracts)
├── 📁 ICLR_retriever/          ← ICLR-specific logic (OpenReview + Virtual site scraping)
```
---

## General ACL Conferences (`ACL_retriever.py`)

Supports:  
- ACL, NAACL, EMNLP, EACL, AACL (2022–2024)

Scrapes metadata directly from ACL Anthology.

## TACL Journal (`TACL_retriever.py`)
TACL papers have a different structure in ACL Anthology.

## AACL 2023 (`AACL2023_retriever/`)
AACL 2023 does not include abstracts on ACL Anthology webpages.
This 3-step pipeline downloads PDFs and uses GROBID to extract abstracts.

## ICLR Conferences (`ICLR_retriever/`)

ICLR papers are hosted on [OpenReview](https://openreview.net), but:

- For **ICLR 2022** and **2023**, we use the OpenReview API.
- For **ICLR 2024**, metadata is **not fully exposed via the API**, so we scrape the [ICLR virtual site](https://iclr.cc/virtual/2024/) instead.
