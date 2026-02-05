"""
Streamlit Web Dashboard for Legal Precedent Search System
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Legal Precedent Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .case-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .similarity-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .similarity-good {
        color: #ffc107;
        font-weight: bold;
    }
    .similarity-fair {
        color: #fd7e14;
        font-weight: bold;
    }
    .similarity-poor {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load resources with caching
@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_data
def load_embeddings():
    """Load embeddings and chunk index"""
    embeddings = np.load('data/embeddings/all_embeddings.npy')
    with open('data/embeddings/chunk_index.pkl', 'rb') as f:
        chunk_index = pickle.load(f)
    return embeddings, chunk_index

@st.cache_data
def load_chunk_text(case_id, chunk_id):
    """Load actual chunk text from JSON"""
    chunk_file = Path('data/chunks') / f"{case_id}_chunks.json"
    
    if not chunk_file.exists():
        return None
    
    try:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for chunk in data['chunks']:
            if chunk['chunk_id'] == chunk_id:
                return chunk['text']
    except:
        return None
    
    return None

# Initialize
try:
    model = load_model()
    embeddings, chunk_index = load_embeddings()
    
    # Calculate statistics
    unique_cases = len(set([c['case_id'] for c in chunk_index]))
    total_chunks = len(chunk_index)
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Search function
def search_cases(query, top_k=10, deduplicate=True, min_similarity=0.0):
    """Search for similar cases"""
    # Generate embedding
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    if deduplicate:
        top_indices = similarities.argsort()[-(top_k * 5):][::-1]
        results = []
        seen_cases = set()
        
        for idx in top_indices:
            metadata = chunk_index[idx]
            case_id = metadata['case_id']
            similarity = float(similarities[idx])
            
            if similarity < min_similarity or case_id in seen_cases:
                continue
            
            seen_cases.add(case_id)
            results.append({
                'case_id': case_id,
                'chunk_id': metadata['chunk_id'],
                'similarity': similarity,
                'date': metadata.get('date', 'Unknown'),
                'court': metadata.get('court', 'Unknown')
            })
            
            if len(results) >= top_k:
                break
    else:
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            metadata = chunk_index[idx]
            similarity = float(similarities[idx])
            
            if similarity < min_similarity:
                continue
            
            results.append({
                'case_id': metadata['case_id'],
                'chunk_id': metadata['chunk_id'],
                'similarity': similarity,
                'date': metadata.get('date', 'Unknown'),
                'court': metadata.get('court', 'Unknown')
            })
    
    return results

def get_similarity_class(score):
    """Get CSS class for similarity score"""
    if score >= 0.7:
        return "similarity-excellent", "🟢 Excellent", "#28a745"
    elif score >= 0.5:
        return "similarity-good", "🟡 Good", "#ffc107"
    elif score >= 0.3:
        return "similarity-fair", "🟠 Fair", "#fd7e14"
    else:
        return "similarity-poor", "🔴 Poor", "#dc3545"

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<h1 class="main-header">⚖️ Legal Precedent Search System</h1>', unsafe_allow_html=True)
st.markdown("**Semantic Search Across Supreme Court Judgments**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    # Search parameters
    top_k = st.slider("Number of results", 1, 50, 10)
    min_similarity = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.05)
    deduplicate = st.checkbox("Deduplicate (one per case)", value=True)
    show_text = st.checkbox("Show chunk text preview", value=False)
    
    st.markdown("---")
    
    # Statistics
    st.header("📊 Database Stats")
    st.metric("Total Cases", unique_cases)
    st.metric("Total Chunks", total_chunks)
    st.metric("Embedding Dimension", embeddings.shape[1])
    
    st.markdown("---")
    
    # Quick queries
    st.header("🔥 Example Queries")
    example_queries = [
        "contract breach damages",
        "arbitration tribunal",
        "employment termination",
        "property dispute ownership",
        "constitutional validity"
    ]
    
    for eq in example_queries:
        if st.button(eq, key=f"example_{eq}"):
            st.session_state['query'] = eq

# Main search interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔍 Enter your search query:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., breach of contract, arbitration award, property rights...",
        key='search_input'
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    search_button = st.button("🔎 Search", type="primary", use_container_width=True)

# Perform search
if search_button and query:
    with st.spinner("🔍 Searching through legal precedents..."):
        results = search_cases(
            query, 
            top_k=top_k, 
            deduplicate=deduplicate,
            min_similarity=min_similarity
        )
    
    # Display results
    if results:
        st.success(f"✅ Found {len(results)} relevant cases")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_similarity = sum(r['similarity'] for r in results) / len(results)
        excellent = sum(1 for r in results if r['similarity'] >= 0.7)
        good = sum(1 for r in results if 0.5 <= r['similarity'] < 0.7)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Results Found", len(results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🟢 Excellent", excellent)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🟡 Good", good)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display each result
        for i, result in enumerate(results, 1):
            css_class, quality_label, color = get_similarity_class(result['similarity'])
            
            with st.container():
                # Header row
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {i}. {result['case_id']}")
                
                with col2:
                    st.markdown(f"<span class='{css_class}'>{quality_label}</span>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**{result['similarity']:.3f}**")
                
                # Details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"📅 **Date:** {result['date']}")
                
                with col2:
                    st.write(f"🏛️ **Court:** {result['court']}")
                
                with col3:
                    st.write(f"📄 **Chunk ID:** {result['chunk_id']}")
                
                # Show chunk text if enabled
                if show_text:
                    chunk_text = load_chunk_text(result['case_id'], result['chunk_id'])
                    
                    if chunk_text:
                        with st.expander("📖 View Chunk Text"):
                            st.text_area(
                                "Content",
                                chunk_text,
                                height=200,
                                key=f"text_{i}",
                                label_visibility="collapsed"
                            )
                    else:
                        st.caption("⚠️ Text preview not available")
                
                st.markdown("---")
        
        # Export option
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create DataFrame
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create JSON
            export_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_results': len(results),
                'settings': {
                    'top_k': top_k,
                    'min_similarity': min_similarity,
                    'deduplicate': deduplicate
                },
                'results': results
            }
            
            json_str = json.dumps(export_data, indent=2)
            
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.warning("⚠️ No results found. Try:")
        st.write("- Lowering the minimum similarity threshold")
        st.write("- Using different search terms")
        st.write("- Broadening your query")

elif search_button:
    st.warning("⚠️ Please enter a search query")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏛️ Judicial AI Decision Support System | B.Tech CSE Capstone Project</p>
    <p>Powered by Sentence Transformers & Streamlit</p>
</div>
""", unsafe_allow_html=True)