# Scrape AACL 2023 + Abstract Extractor (GROBID)

Since some conferences (e.g. **AACL 2023**) do **not include abstracts on the website**, only in the PDF. This workflow retrieves papers from ACL Anthology (e.g. AACL 2023), downloads their PDFs, extracts abstracts using GROBID, and builds a complete JSON.

---

## Pipeline Steps

### 1. Download metadata and PDFs
Run `1.AACL2023_RetrieverPDF.py`
→ Saves `aacl2023.json` and `aacl2023_pdfs/`

### Start GROBID server
If you have Docker:

```bash
docker run --rm -t --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

Check: http://localhost:8070/api/isalive → should return "true"

### Convert PDFs to XML
Run `2.ParsePDFintoXML.py`
→ Saves XMLs to aacl2023_xmls/

### Add abstracts to JSON
Run 3.AddAbstractToJSONelements.py
→ Updates aacl2023.json with "abstract" field from XMLs