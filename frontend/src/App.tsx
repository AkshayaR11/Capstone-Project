import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

interface CaseResult {
  case_id: string;
  score: number;
  date: string;
  severity: number;
  summary: string;
}

function App() {
  const [query, setQuery] = useState('')
  const [yearFilter, setYearFilter] = useState('')
  const [results, setResults] = useState<CaseResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedCase, setExpandedCase] = useState<string | null>(null)
  const [topK, setTopK] = useState<number>(6)

  // Upload Feature State
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadSummary, setUploadSummary] = useState<string | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [showUploadSummary, setShowUploadSummary] = useState(false)

  const toggleSummary = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedCase(expandedCase === id ? null : id);
  }

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query) return;

    setLoading(true)
    setError(null)
    setResults([])

    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          yearFilter: yearFilter,
          topK: topK
        })
      });

      if (!response.ok) {
        throw new Error("Failed to fetch from backend")
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError("An unknown error occurred.")
      }
    } finally {
      setLoading(false)
    }
  }

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setUploadLoading(true);
    setUploadSummary(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/summarize_upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process PDF upload. Ensure PyMuPDF is installed correctly.");
      }

      const data = await response.json();
      setUploadSummary(data.summary);
      setShowUploadSummary(true);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError("An error occurred during upload.")
      }
    } finally {
      setUploadLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="background-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>

      <header className="header glass">
        <h1 className="title animate-fade-in">⚖️ Judicial AI</h1>
        <p className="subtitle">Semantic Precedent Search Engine</p>
      </header>

      <main className="main-content">
        <div className="search-container glass">
          <form className="search-form" onSubmit={handleSearch}>
            <div className="input-group">
              <label>Search Query</label>
              <input 
                type="text" 
                className="input-field"
                placeholder="contract breach, arbitration, murder..." 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            
            <div className="input-group shrink">
              <label>Year (Opt)</label>
              <input 
                type="text" 
                className="input-field secondary-input"
                placeholder="e.g. 2012" 
                value={yearFilter}
                onChange={(e) => setYearFilter(e.target.value)}
              />
            </div>
            
            <div className="input-group shrink">
              <label>Matches: {topK}</label>
              <input 
                type="range" 
                min="1" max="15" 
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ marginTop: '10px' }}
              />
            </div>

            <button type="submit" className="search-btn" disabled={loading}>
              {loading ? (
                <span className="spinner"></span>
              ) : (
                "Search"
              )}
            </button>
          </form>
        </div>

        <div className="upload-container glass" style={{ marginTop: '20px', marginBottom: '20px', padding: '20px' }}>
          <h3 style={{ marginBottom: '10px', fontSize: '1.2rem', color: '#fff' }}>📄 Live Document Summarization</h3>
          <form className="upload-form" onSubmit={handleUploadSubmit} style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
            <input 
              type="file" 
              accept="application/pdf"
              onChange={(e) => setSelectedFile(e.target.files ? e.target.files[0] : null)}
              className="input-field"
              style={{ flex: 1 }}
            />
            <button type="submit" className="search-btn" disabled={!selectedFile || uploadLoading} style={{ background: '#10b981', minWidth: '150px' }}>
              {uploadLoading ? <span className="spinner" style={{ width: '20px', height: '20px', borderWidth: '3px' }}></span> : "Analyze PDF"}
            </button>
          </form>
          {uploadSummary && (
            <div style={{ marginTop: '15px' }}>
              <button 
                className="summary-btn" 
                onClick={(e) => { e.preventDefault(); setShowUploadSummary(!showUploadSummary); }}
                style={{ marginBottom: '10px', background: 'rgba(16, 185, 129, 0.1)', borderColor: '#10b981', color: '#10b981' }}
              >
                {showUploadSummary ? "Hide AI Summary ▲" : "View Live Summary ▼"}
              </button>
              
              {showUploadSummary && (
                <div className="summary-dropdown" style={{ background: 'rgba(0, 0, 0, 0.4)' }}>
                  <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#10b981'}}>✨ AI Extracted Document Framework</span></strong></p>
                  <div className="markdown-body">
                    <ReactMarkdown>{uploadSummary}</ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {error && (
          <div className="error-card glass">
            ⚠️ {error} - Ensure FastAPI is running on Port 8000.
          </div>
        )}

        <div className="results-grid">
          {results.map((r, index) => (
            <div 
              key={r.case_id} 
              className="result-card glass" 
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="card-header">
               <span className="similarity-badge">{(r.score * 100).toFixed(1)}% Match</span>
               <span className={`severity-badge level-${r.severity > 7 ? 'high' : r.severity > 4 ? 'med' : 'low'}`}>
                  Severity {r.severity}
               </span>
              </div>
              <h3 className="case-title">{r.case_id.replace(/_/g, ' ')}</h3>
              <div className="card-footer">
                <p><strong>Date:</strong> {r.date}</p>
                <button 
                  className="summary-btn" 
                  onClick={(e) => toggleSummary(r.case_id, e)}
                >
                  {expandedCase === r.case_id ? "Hide Summary ▲" : "⚖️ AI Summary ▼"}
                </button>
              </div>
              
              {expandedCase === r.case_id && (
                <div className="summary-dropdown">
                  <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#3b82f6'}}>🏛️ AI Legal Synopsis</span></strong></p>
                  <div className="markdown-body">
                    <ReactMarkdown>{r.summary}</ReactMarkdown>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
        
        {!loading && results.length === 0 && !error && query && (
          <div className="empty-state glass">
             No legal precedents found matching your query.
          </div>
        )}
      </main>
    </div>
  )
}

export default App
