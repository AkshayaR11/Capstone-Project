"""
Generate embeddings for all chunks using LegalBERT
This enables semantic search for precedent retrieval
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle

# Configuration
CHUNKS_DIR = "data/chunks"
OUTPUT_DIR = "data/embeddings"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.npy"
INDEX_FILE = "data/embeddings/chunk_index.pkl"
BATCH_SIZE = 8  # Smaller batch for BERT model

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load LegalBERT model (fine-tuned on legal text)
print("📥 Loading LegalBERT model (this takes ~1-2 minutes)...")
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print(f"✅ Model loaded: {model_name} (768-dimensional embeddings)\n")

# Mean pooling function for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Collect all chunks
all_chunks = []
chunk_metadata = []

chunk_files = list(Path(CHUNKS_DIR).glob("*_chunks.json"))
print(f"📄 Found {len(chunk_files)} case files\n")

for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    case_id = data['case_id']
    
    for chunk in data['chunks']:
        all_chunks.append(chunk['text'].lower())  # LegalBERT is uncased
        chunk_metadata.append({
            'case_id': case_id,
            'chunk_id': chunk['chunk_id'],
            'word_count': chunk['word_count'],
            'date': data['metadata'].get('date'),
            'court': data['metadata'].get('court')
        })

print(f"\n📊 Total chunks to process: {len(all_chunks)}")
print(f"💾 Estimated embedding size: {len(all_chunks) * 768 * 4 / 1024 / 1024:.2f} MB\n")

# Generate embeddings in batches (to avoid memory issues)
print("🧠 Generating embeddings...")
all_embeddings = []

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Processing batches"):
    batch = all_chunks[i:i + BATCH_SIZE]
    encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)  # Normalize for cosine similarity
    all_embeddings.append(batch_embeddings.numpy())

# Combine all batches
embeddings_matrix = np.vstack(all_embeddings)

print(f"\n✅ Generated {embeddings_matrix.shape[0]} embeddings")
print(f"📐 Embedding dimensions: {embeddings_matrix.shape[1]}")

# Save embeddings
print(f"\n💾 Saving embeddings to {EMBEDDINGS_FILE}...")
np.save(EMBEDDINGS_FILE, embeddings_matrix)

# Save metadata index
print(f"💾 Saving chunk index to {INDEX_FILE}...")
with open(INDEX_FILE, 'wb') as f:
    pickle.dump(chunk_metadata, f)

# Save a human-readable summary
summary = {
    'total_cases': len(chunk_files),
    'total_chunks': len(all_chunks),
    'embedding_dim': int(embeddings_matrix.shape[1]),
    'model_name': model_name,
    'file_size_mb': os.path.getsize(EMBEDDINGS_FILE) / 1024 / 1024,
    'sample_metadata': chunk_metadata[:5]
}

with open(os.path.join(OUTPUT_DIR, 'embedding_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("🎉 EMBEDDINGS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"📁 Files created:")
print(f"   • {EMBEDDINGS_FILE}")
print(f"   • {INDEX_FILE}")
print(f"   • {OUTPUT_DIR}/embedding_summary.json")
print(f"\n💡 Next: Build semantic search system!")
print("="*60)