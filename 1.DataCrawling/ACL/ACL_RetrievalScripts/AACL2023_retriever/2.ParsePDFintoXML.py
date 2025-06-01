import os
import requests

# 📁 Set to match your PDF download output directory
pdf_folder = "github/ForUpload/1.Data_crawling/acl_data/Scripts/AACL2023_retreiever/aacl2023_pdfs"

# 🌐 GROBID service endpoint
grobid_url = "http://localhost:8070/api/processFulltextDocument"

# 📁 Output directory for XMLs (same base as PDFs for convenience)
xml_folder = "github/ForUpload/1.Data_crawling/acl_data/Scripts/AACL2023_retreiever/aacl2023_xmls"
os.makedirs(xml_folder, exist_ok=True)

# 📄 Process each PDF using GROBID
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        with open(pdf_path, 'rb') as pdf:
            files = {'input': pdf}
            response = requests.post(grobid_url, files=files)

        # 💾 Save GROBID's XML response
        xml_output_path = os.path.join(xml_folder, f"{pdf_file[:-4]}.xml")
        with open(xml_output_path, 'wb') as xml_file:
            xml_file.write(response.content)

        print(f"Processed {pdf_file} into {xml_output_path}")
