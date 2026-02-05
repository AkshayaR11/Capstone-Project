import re

def clean_legal_text(text: str) -> str:
    """
    Cleans raw Indian legal judgment text for NLP processing.
    Optimized for Supreme Court judgments.
    """
    
    if not text or len(text.strip()) < 50:
        return ""
    
    # 1. Remove Indian Kanoon watermarks/URLs
    text = re.sub(r'Indian Kanoon.*?http://indiankanoon\.org/doc/\d+/', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+', '', text)
    
    # 2. Remove page markers
    text = re.sub(r'---\s*Page\s+\d+\s*---', '\n', text)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
    
    # 3. Remove citation headers (but keep citations in judgment body)
    text = re.sub(r'^(Equivalent citations?|CITATION).*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # 4. Remove metadata headers (only at the start)
    lines = text.split('\n')
    cleaned_lines = []
    in_header = True
    
    for line in lines:
        # Skip header metadata
        if in_header and any(keyword in line.lower() for keyword in [
            'author:', 'bench:', 'petitioner:', 'respondent:', 
            'date of judgment:', 'equivalent citations', 'act:'
        ]):
            continue
        
        # Once we hit substantial text, stop skipping
        if len(line.strip()) > 100:
            in_header = False
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # 5. Clean up excessive whitespace
    text = re.sub(r'\s{2,}', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines → double newline
    
    # 6. Remove lines with only numbers/special characters
    text = re.sub(r'^\s*[\d\.\-\*]+\s*$', '', text, flags=re.MULTILINE)
    
    # 7. Fix encoding issues
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\t', ' ')
    
    # 8. Remove excessive punctuation/symbols
    text = re.sub(r'[=\*_]{3,}', '', text)
    
    return text.strip()


def extract_case_metadata(text: str) -> dict:
    """
    Extract basic metadata from judgment text.
    Returns dict with case_number, date, parties, court.
    """
    metadata = {
        'case_number': None,
        'date': None,
        'petitioner': None,
        'respondent': None,
        'court': None
    }
    
    lines = text.split('\n')[:50]  # Only check first 50 lines
    
    for line in lines:
        # Extract date
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})', line)
        if date_match and not metadata['date']:
            metadata['date'] = date_match.group(1)
        
        # Extract petitioner
        if 'PETITIONER:' in line.upper():
            metadata['petitioner'] = line.split(':', 1)[1].strip()
        
        # Extract respondent
        if 'RESPONDENT:' in line.upper():
            metadata['respondent'] = line.split(':', 1)[1].strip()
        
        # Detect court
        if 'SUPREME COURT' in line.upper():
            metadata['court'] = 'Supreme Court of India'
    
    return metadata