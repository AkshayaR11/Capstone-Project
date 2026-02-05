"""
Generate embeddings for all chunks using Sentence Transformers
This enables semantic search for precedent retrieval
"""

import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pickle

# Configuration
CHUNKS_DIR = "data/chunks"
OUTPUT_DIR = "data/embeddings"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.npy"
INDEX_FILE = "data/embeddings/chunk_index.pkl"
BATCH_SIZE = 16  # Process 16 chunks at a time

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load a legal-optimized model (small enough for laptop)
print("📥 Loading embedding model (this takes ~30 seconds)...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Alternative: 'BAAI/bge-small-en-v1.5' (better quality, slightly slower)

print(f"✅ Model loaded: {model.get_sentence_embedding_dimension()}-dimensional embeddings\n")

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
        all_chunks.append(chunk['text'])
        chunk_metadata.append({
            'case_id': case_id,
            'chunk_id': chunk['chunk_id'],
            'word_count': chunk['word_count'],
            'date': data['metadata'].get('date'),
            'court': data['metadata'].get('court')
        })

print(f"\n📊 Total chunks to process: {len(all_chunks)}")
print(f"💾 Estimated embedding size: {len(all_chunks) * 384 * 4 / 1024 / 1024:.2f} MB\n")

# Generate embeddings in batches (to avoid memory issues)
print("🧠 Generating embeddings...")
all_embeddings = []

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Processing batches"):
    batch = all_chunks[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

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
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
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