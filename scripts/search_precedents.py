"""
Simple semantic search for legal precedents
"""

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model and data
print("Loading model and embeddings...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = np.load('data/embeddings/all_embeddings.npy')

with open('data/embeddings/chunk_index.pkl', 'rb') as f:
    chunk_index = pickle.load(f)

print(f"✅ Loaded {len(embeddings)} embeddings\n")


def search_cases(query, top_k=10, deduplicate=True):
    """
    Search for similar cases
    
    Args:
        query: Search text
        top_k: Number of results
        deduplicate: Show only one result per case
    
    Returns:
        List of results with case_id, similarity, date, court
    """
    # Get query embedding
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top results
    if deduplicate:
        # Get more candidates to account for duplicates
        top_indices = similarities.argsort()[-(top_k * 5):][::-1]
        
        results = []
        seen_cases = set()
        
        for idx in top_indices:
            metadata = chunk_index[idx]
            case_id = metadata['case_id']
            
            # Skip duplicates
            if case_id in seen_cases:
                continue
            
            seen_cases.add(case_id)
            results.append({
                'case_id': case_id,
                'chunk_id': metadata['chunk_id'],
                'similarity': float(similarities[idx]),
                'date': metadata.get('date', 'Unknown'),
                'court': metadata.get('court', 'Unknown')
            })
            
            if len(results) >= top_k:
                break
    else:
        # Return all chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            metadata = chunk_index[idx]
            results.append({
                'case_id': metadata['case_id'],
                'chunk_id': metadata['chunk_id'],
                'similarity': float(similarities[idx]),
                'date': metadata.get('date', 'Unknown'),
                'court': metadata.get('court', 'Unknown')
            })
    
    return results


# Simple command-line interface
if __name__ == "__main__":
    print("="*60)
    print("🔍 LEGAL PRECEDENT SEARCH")
    print("="*60)
    
    while True:
        query = input("\nEnter search query (or 'quit'): ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        results = search_cases(query, top_k=5)
        
        print(f"\n📝 Results for: '{query}'")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['case_id']}")
            print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Date: {result['date']}")
            print(f"   Court: {result['court']}")