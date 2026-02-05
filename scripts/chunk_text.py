import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from clean_text import clean_legal_text, extract_case_metadata

# Configuration
INPUT_DIR = "data/extracted_text"
OUTPUT_DIR = "data/chunks"
SUMMARY_FILE = "data/processing_summary.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimized splitter for legal text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,          # ~400-500 words (better for embeddings)
    chunk_overlap=200,        # Good overlap for context
    separators=[
        "\n\n\n",             # Section breaks
        "\n\n",               # Paragraph breaks
        "\n",                 # Line breaks
        ". ",                 # Sentences
        " "                   # Words (fallback)
    ],
    length_function=len,
)

MIN_CHUNK_LENGTH = 150  # Characters (not words) - faster

def is_valid_chunk(chunk: str) -> bool:
    """Check if chunk contains meaningful legal content"""
    # Too short
    if len(chunk) < MIN_CHUNK_LENGTH:
        return False
    
    # Count words
    word_count = len(chunk.split())
    if word_count < 50:  # Minimum 50 words
        return False
    
    # Reject if mostly numbers/special chars
    alphanumeric = sum(c.isalnum() for c in chunk)
    if alphanumeric / len(chunk) < 0.5:
        return False
    
    return True


def remove_metadata_section(text: str) -> str:
    """
    Remove the header section (before 'JUDGMENT' or 'HEADNOTE')
    but keep the actual judgment text.
    """
    # Find where the actual judgment starts
    judgment_markers = ['JUDGMENT:', 'HEADNOTE:', 'The Judgment of the Court']
    
    for marker in judgment_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) == 2:
                return marker + parts[1]  # Keep marker + judgment text
    
    # If no marker found, return as is (better than losing content)
    return text


# Statistics
stats = {
    'total_files': 0,
    'successful': 0,
    'failed': [],
    'total_chunks': 0,
    'avg_chunks_per_case': 0
}

txt_files = list(Path(INPUT_DIR).glob("*.txt"))
stats['total_files'] = len(txt_files)

print(f"📄 Processing {len(txt_files)} text files...")

all_chunks_count = []

for txt_file in tqdm(txt_files, desc="Chunking texts"):
    case_id = txt_file.stem
    
    try:
        # Read raw text
        with open(txt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # Extract metadata first
        metadata = extract_case_metadata(raw_text)
        
        # Clean text
        cleaned_text = clean_legal_text(raw_text)
        
        # Remove header metadata section
        cleaned_text = remove_metadata_section(cleaned_text)
        
        # Check if we have enough content
        if len(cleaned_text) < 500:
            stats['failed'].append(f"{case_id}: Too short after cleaning")
            continue
        
        # Split into chunks
        chunks = text_splitter.split_text(cleaned_text)
        
        # Filter valid chunks
        valid_chunks = [c for c in chunks if is_valid_chunk(c)]
        
        if not valid_chunks:
            stats['failed'].append(f"{case_id}: No valid chunks generated")
            continue
        
        # Prepare output data
        output_data = {
            "case_id": case_id,
            "metadata": metadata,
            "num_chunks": len(valid_chunks),
            "total_chars": sum(len(c) for c in valid_chunks),
            "avg_chunk_size": sum(len(c) for c in valid_chunks) // len(valid_chunks),
            "chunks": [
                {
                    "chunk_id": idx,
                    "char_count": len(chunk),
                    "word_count": len(chunk.split()),
                    "text": chunk.strip()
                }
                for idx, chunk in enumerate(valid_chunks)
            ]
        }
        
        # Save to JSON
        output_path = Path(OUTPUT_DIR) / f"{case_id}_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        stats['successful'] += 1
        all_chunks_count.append(len(valid_chunks))
    
    except Exception as e:
        stats['failed'].append(f"{case_id}: {str(e)}")

# Calculate statistics
stats['total_chunks'] = sum(all_chunks_count)
if all_chunks_count:
    stats['avg_chunks_per_case'] = sum(all_chunks_count) / len(all_chunks_count)

# Save summary
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

# Print results
print("\n" + "="*50)
print("📊 PROCESSING SUMMARY")
print("="*50)
print(f"✅ Successfully processed: {stats['successful']}/{stats['total_files']}")
print(f"📦 Total chunks created: {stats['total_chunks']}")
print(f"📈 Average chunks per case: {stats['avg_chunks_per_case']:.1f}")

if stats['failed']:
    print(f"\n⚠️  {len(stats['failed'])} files failed:")
    for error in stats['failed'][:5]:  # Show first 5 errors
        print(f"   • {error}")
    if len(stats['failed']) > 5:
        print(f"   ... and {len(stats['failed']) - 5} more")

print(f"\n💾 Chunks saved to: {OUTPUT_DIR}/")
print(f"📋 Full report saved to: {SUMMARY_FILE}")