import fitz  # PyMuPDF
import os
from tqdm import tqdm
from pathlib import Path

# Configuration
PDF_DIR = "data/raw_pdfs"
TEXT_DIR = "data/extracted_text"
ERROR_LOG = "data/extraction_errors.txt"

# Create output directory
os.makedirs(TEXT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF with error handling"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            
            # Skip empty pages
            if page_text.strip():
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += page_text
        
        doc.close()
        return full_text, None
    
    except Exception as e:
        return None, str(e)

# Get list of PDF files (case-insensitive)
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]

print(f"📁 Found {len(pdf_files)} PDF files")

# Track errors
errors = []
successful = 0

# Process each PDF
for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    
    # Extract text
    full_text, error = extract_text_from_pdf(pdf_path)
    
    if error:
        errors.append(f"{pdf_file}: {error}")
        continue
    
    # Check if extraction was successful
    if not full_text or len(full_text.strip()) < 100:
        errors.append(f"{pdf_file}: Empty or too short (possibly scanned PDF)")
        continue
    
    # Save as .txt (handle both .PDF and .pdf extensions)
    txt_filename = Path(pdf_file).stem + ".txt"
    txt_path = os.path.join(TEXT_DIR, txt_filename)
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    successful += 1

# Save error log
if errors:
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    print(f"⚠️  {len(errors)} files had errors (see {ERROR_LOG})")

print(f"✅ Successfully extracted {successful}/{len(pdf_files)} PDFs")
print(f"📄 Text files saved to: {TEXT_DIR}/")