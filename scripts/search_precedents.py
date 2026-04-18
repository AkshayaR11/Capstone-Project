"""
Semantic search for legal precedents using LegalBERT and pgvector (PostgreSQL)
"""

import psycopg2
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

print("Loading LegalBERT model...")
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Model loaded")

# Mean pooling function for sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def search_cases(query, top_k=10, deduplicate=True):
    """
    Search for similar cases using PostgreSQL pgvector
    """
    # Get query embedding
    encoded_input = tokenizer([query.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    query_emb = F.normalize(query_emb, p=2, dim=1).numpy()[0]
    
    results = []
    
    try:
        conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        cur = conn.cursor()
        
        if deduplicate:
            # Query deduplicates by case_id using DISTINCT ON, ordering by shortest vector distance
            sql = """
                SELECT DISTINCT ON (c.case_id) 
                    c.case_id, 
                    cc.chunk_id, 
                    cc.embedding <=> %s::vector AS distance, 
                    c.case_date,
                    c.max_severity_score
                FROM case_chunks cc
                JOIN cases c ON cc.case_id = c.case_id
                ORDER BY c.case_id, distance ASC
            """
            # To get overall top K, we wrap it in a subquery
            sql = f"""
                SELECT * FROM ({sql}) AS unique_cases
                ORDER BY distance ASC
                LIMIT %s;
            """
            cur.execute(sql, (query_emb.tolist(), top_k))
        else:
            # Standard query returning all chunks
            sql = """
                SELECT c.case_id, cc.chunk_id, cc.embedding <=> %s::vector AS distance, c.case_date, c.max_severity_score
                FROM case_chunks cc
                JOIN cases c ON cc.case_id = c.case_id
                ORDER BY distance ASC
                LIMIT %s;
            """
            cur.execute(sql, (query_emb.tolist(), top_k))
            
        records = cur.fetchall()
        cur.close()
        conn.close()
        
        for r in records:
            sim_score = 1 - float(r[2])  # Convert Cosine Distance back to Similarity Score
            results.append({
                'case_id': r[0],
                'chunk_id': r[1],
                'similarity': sim_score,
                'date': r[3] if r[3] else 'Unknown',
                'severity': r[4]
            })
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        
    return results


# Simple command-line interface
if __name__ == "__main__":
    print("="*60)
    print("🔍 LEGAL PRECEDENT SEARCH (PGVECTOR)")
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
            print(f"   Date: {result['date']} | Severity: {result['severity']}")