"""
Generate embeddings for chunks using LegalBERT (INCREMENTAL & OPTIMIZED VERSION)
Only processes NEW cases if embeddings already exist, skipping previously calculated ones.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Configuration
CHUNKS_DIR = "data/chunks"
OUTPUT_DIR = "data/embeddings"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.npy"
INDEX_FILE = "data/embeddings/chunk_index.pkl"
BATCH_SIZE = 32  # Optimized CPU batch size

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utilize available CPU threads for parallel matrix multiplication
if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(os.cpu_count())

# Load existing embeddings and metadata if available
existing_embeddings = []
existing_metadata = []
processed_case_ids = set()

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
    print("📦 Found existing embeddings file. Loading cached data...")
    try:
        existing_embeddings = np.load(EMBEDDINGS_FILE)
        with open(INDEX_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
        
        processed_case_ids = set(m["case_id"] for m in existing_metadata)
        print(f"✅ Loaded {len(existing_embeddings)} existing chunk embeddings across {len(processed_case_ids)} cases.")
    except Exception as e:
        print(f"⚠️ Error loading cached embeddings ({e}). Starting fresh.")
        existing_embeddings = []
        existing_metadata = []
        processed_case_ids = set()

# Collect chunk files
chunk_files = list(Path(CHUNKS_DIR).glob("*_chunks.json"))
print(f"📄 Found {len(chunk_files)} total case chunk files.")

# Filter out already processed cases
new_chunk_files = [f for f in chunk_files if f.stem.replace("_chunks", "") not in processed_case_ids]

if not new_chunk_files:
    print("\n🎉 All case chunks are already embedded! No new processing needed.")
    exit(0)

print(f"✨ Identified {len(new_chunk_files)} NEW case files to process.")

# Load LegalBERT model
print("\n📥 Loading LegalBERT model...")
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Collect text and metadata for new chunks only
new_chunks = []
new_metadata = []

for chunk_file in tqdm(new_chunk_files, desc="Loading new chunks"):
    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    case_id = data['case_id']
    for chunk in data['chunks']:
        new_chunks.append(chunk['text'].lower())
        new_metadata.append({
            'case_id': case_id,
            'chunk_id': chunk['chunk_id'],
            'word_count': chunk['word_count'],
            'date': data['metadata'].get('date'),
            'court': data['metadata'].get('court')
        })

print(f"\n📊 Total new chunks to process: {len(new_chunks)} (Batch Size: {BATCH_SIZE})")

# Generate embeddings for new chunks
new_embeddings_list = []
for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding batches"):
    batch = new_chunks[i:i + BATCH_SIZE]
    encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
    new_embeddings_list.append(batch_embeddings.numpy())

# Merge new embeddings with existing embeddings
if len(new_embeddings_list) > 0:
    new_matrix = np.vstack(new_embeddings_list)
    if len(existing_embeddings) > 0:
        final_embeddings = np.vstack([existing_embeddings, new_matrix])
        final_metadata = existing_metadata + new_metadata
    else:
        final_embeddings = new_matrix
        final_metadata = new_metadata
else:
    final_embeddings = existing_embeddings
    final_metadata = existing_metadata

# Save back to disk
print(f"\n💾 Saving updated embeddings dataset ({len(final_embeddings)} total chunks) to {EMBEDDINGS_FILE}...")
np.save(EMBEDDINGS_FILE, final_embeddings)

print(f"💾 Saving updated chunk index to {INDEX_FILE}...")
with open(INDEX_FILE, 'wb') as f:
    pickle.dump(final_metadata, f)

summary = {
    'total_cases': len(chunk_files),
    'total_chunks': len(final_metadata),
    'embedding_dim': int(final_embeddings.shape[1]),
    'model_name': model_name,
    'file_size_mb': os.path.getsize(EMBEDDINGS_FILE) / 1024 / 1024,
    'sample_metadata': final_metadata[:5]
}

with open(os.path.join(OUTPUT_DIR, 'embedding_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("🎉 EMBEDDINGS UPDATED SUCCESSFULLY!")
print("="*60)