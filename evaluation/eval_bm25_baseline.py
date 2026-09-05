# save as eval_bm25_baseline.py
import psycopg2
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import numpy as np
import sys
sys.path.append(".")
from config import DATABASE_URL

# ── Load all case texts from chunk files ──────────────────
print("Loading case texts...")
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("SELECT case_id FROM cases")
all_case_ids = [r[0] for r in cur.fetchall()]
conn.close()

corpus_ids = []
corpus_texts = []

for case_id in all_case_ids:
    try:
        with open(f"data/chunks/{case_id}_chunks.json", "r") as f:
            data = json.load(f)
        text = " ".join(c["text"] for c in data["chunks"])
        corpus_ids.append(case_id)
        corpus_texts.append(text)
    except:
        continue

print(f"Loaded {len(corpus_texts)} cases")

# ── BM25 Index ────────────────────────────────────────────
print("Building BM25 index...")
tokenized = [t.lower().split() for t in corpus_texts]
bm25 = BM25Okapi(tokenized)

# ── LegalBERT ─────────────────────────────────────────────
print("Loading LegalBERT...")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer([text.lower()], padding=True, truncation=True,
                      max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**inputs)
    emb = output.last_hidden_state.mean(dim=1)
    return F.normalize(emb, p=2, dim=1).numpy()[0]

def search_legalbert(query, top_k=10):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    emb = get_embedding(query)
    cur.execute("""
        SELECT * FROM (
            SELECT DISTINCT ON (c.case_id)
                c.case_id,
                cc.embedding <=> %s::vector AS distance
            FROM case_chunks cc
            JOIN cases c ON cc.case_id = c.case_id
            ORDER BY c.case_id, distance ASC
        ) AS matches
        ORDER BY distance ASC
        LIMIT %s;
    """, (emb.tolist(), top_k))
    results = [r[0] for r in cur.fetchall()]
    conn.close()
    return results

def search_bm25(query, top_k=10):
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [corpus_ids[i] for i in top_indices]

# ── Test queries ──────────────────────────────────────────
# These are deliberately chosen to match case_ids in your DB
# so results are interpretable without manual labels
queries = [
    "murder conviction Section 302 appeal",
    "rape accused Section 376 bail",
    "dowry death cruelty husband Section 304B",
    "habeas corpus illegal detention custody",
    "property dispute civil injunction court",
    "cheating fraud Section 420 criminal",
    "attempt to murder Section 307",
    "corruption bribery public servant",
    "divorce matrimonial dispute Hindu Marriage Act",
    "environmental pollution PIL public interest"
]

print("\n" + "="*70)
print("BM25 vs LegalBERT COMPARISON")
print("="*70)

overlap_scores = []

for query in queries:
    bm25_results = search_bm25(query, top_k=10)
    bert_results = search_legalbert(query, top_k=10)

    # Overlap@10: how many results appear in both top 10
    overlap = len(set(bm25_results[:10]) & set(bert_results[:10]))
    overlap_scores.append(overlap)

    print(f"\nQuery: '{query}'")
    print(f"  BM25 Top 3:      {[c.replace('_', ' ')[:40] for c in bm25_results[:3]]}")
    print(f"  LegalBERT Top 3: {[c.replace('_', ' ')[:40] for c in bert_results[:3]]}")
    print(f"  Overlap@10: {overlap}/10")

print(f"\n{'='*70}")
print(f"Mean Overlap@10: {np.mean(overlap_scores):.1f}/10")
print(f"{'='*70}")

# ── Qualitative difference table ──────────────────────────
print("\n=== QUALITATIVE COMPARISON (for paper Table) ===")
print(f"{'Query':<35} {'BM25 Top1':<35} {'BERT Top1':<35}")
print("-"*105)
for query in queries[:5]:
    b = search_bm25(query, 1)[0].replace('_',' ')[:33]
    l = search_legalbert(query, 1)[0].replace('_',' ')[:33]
    print(f"{query[:33]:<35} {b:<35} {l:<35}")