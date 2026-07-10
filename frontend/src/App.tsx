import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

interface CaseResult {
  case_id: string;
  score: number;
  date: string;
  severity: number;
  summary: string;
  bns_sections?: string;
  ipc_sections?: string;
  priority_score?: number;
  case_type?: string;
  priority_explanation?: string;
  similar_cases?: CaseResult[];
}

function ProgressiveSummary({ summary }: { summary: string }) {
  const [showLevel2, setShowLevel2] = useState(false);

  if (!summary) return null;

  // Split by LEVEL 2 header marker case-insensitively
  const markerIndex = summary.toLowerCase().indexOf("level 2");
  let level1 = summary;
  let level2 = "";

  if (markerIndex !== -1) {
    level1 = summary.substring(0, markerIndex).replace(/level 1\s*[—-]\s*executive\s*summary/gi, "").trim();
    level2 = summary.substring(markerIndex).trim();
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
      <div style={{ background: 'rgba(255, 255, 255, 0.01)', padding: '15px', borderRadius: '8px', borderLeft: '3px solid #3b82f6' }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#60a5fa', fontSize: '1rem', fontWeight: 'bold' }}>📋 Level 1: Executive Summary</h4>
        <div className="markdown-body">
          <ReactMarkdown>{level1}</ReactMarkdown>
        </div>
      </div>

      {level2 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <button
            onClick={() => setShowLevel2(!showLevel2)}
            className="summary-btn"
            style={{
              background: 'rgba(129, 140, 248, 0.1)',
              borderColor: 'rgba(129, 140, 248, 0.3)',
              color: '#a5b4fc',
              padding: '6px 12px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.8rem',
              alignSelf: 'flex-start',
              transition: 'all 0.2s',
              margin: '5px 0'
            }}
          >
            {showLevel2 ? 'Hide Level 2: Detailed Analysis ▲' : 'Show Level 2: Detailed Analysis ▼'}
          </button>

          {showLevel2 && (
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '20px', borderRadius: '8px', borderLeft: '3px solid #818cf8', marginTop: '5px' }}>
              <h4 style={{ margin: '0 0 12px 0', color: '#a78bfa', fontSize: '1.05rem', fontWeight: 'bold' }}>🔍 Level 2: Detailed Legal Analysis</h4>
              <div className="markdown-body">
                <ReactMarkdown>{level2}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
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
  const [uploadResult, setUploadResult] = useState<CaseResult | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [showUploadSummary, setShowUploadSummary] = useState(false)

  // Priority Helpers
  const getPriorityCategory = (score?: number) => {
    if (score === undefined || score === null) return "Unknown";
    if (score >= 8.0) return "Critical";
    if (score >= 6.0) return "High";
    if (score >= 4.0) return "Medium";
    if (score >= 2.0) return "Low";
    return "Very Low";
  }

  const getPriorityClass = (score?: number) => {
    if (score === undefined || score === null) return "priority-low";
    if (score >= 8.0) return "priority-critical";
    if (score >= 6.0) return "priority-high";
    if (score >= 4.0) return "priority-medium";
    return "priority-low";
  }

  const formatCaseTitle = (caseId: string) => {
    if (!caseId) return "";
    let name = caseId.replace(/_/g, ' ');
    name = name.replace(/\.pdf$/i, "");
    name = name.replace(/\s+\d+$/, '');
    name = name.replace(/\s+on\s+\d+\s+[A-Za-z]+\s+\d{4}/gi, '');
    name = name.replace(/\s+on\s+\d+[-/]\d+[-/]\d+/gi, '');
    name = name.replace(/\s+vs\s+/gi, ' v. ');
    name = name.replace(/\s+versus\s+/gi, ' v. ');
    return name.replace(/\s+/g, ' ').trim();
  }

  const toggleSummary = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const isExpanding = expandedCase !== id;
    setExpandedCase(isExpanding ? id : null);

    if (isExpanding) {
      // Find the case in search results
      const idx = results.findIndex(r => r.case_id === id);
      if (idx !== -1) {
        const caseItem = results[idx];
        const isPending = !caseItem.summary || 
                          caseItem.summary === "AI Synopsis Pending Processing." || 
                          caseItem.summary.trim() === "";
        
        if (isPending) {
          // Set temp status to show loading
          const updatedResults = [...results];
          updatedResults[idx] = { ...caseItem, summary: "Generating AI Synopsis on-demand... Please wait..." };
          setResults(updatedResults);

          try {
            const response = await fetch(`http://localhost:8000/cases/${encodeURIComponent(id)}/summary`);
            if (response.ok) {
              const data = await response.json();
              const freshResults = [...results];
              const freshIdx = freshResults.findIndex(r => r.case_id === id);
              if (freshIdx !== -1) {
                freshResults[freshIdx] = { ...freshResults[freshIdx], summary: data.summary };
                setResults(freshResults);
              }
            } else {
              const freshResults = [...results];
              const freshIdx = freshResults.findIndex(r => r.case_id === id);
              if (freshIdx !== -1) {
                freshResults[freshIdx] = { ...freshResults[freshIdx], summary: "Failed to generate AI Synopsis. Please try again." };
                setResults(freshResults);
              }
            }
          } catch (err) {
            const freshResults = [...results];
            const freshIdx = freshResults.findIndex(r => r.case_id === id);
            if (freshIdx !== -1) {
              freshResults[freshIdx] = { ...freshResults[freshIdx], summary: "Network error occurred while fetching AI Synopsis." };
              setResults(freshResults);
            }
          }
        }
      }
    }
  }

  const formatPriorityExplanation = (explanation?: string, score?: number) => {
    if (!explanation) {
      return <div style={{ color: '#94a3b8' }}>Score: {score?.toFixed(2)} / 10.0 — {getPriorityCategory(score)}</div>;
    }
    const lines = explanation.split(' | ');

    let isML = false;
    let ruleScore = "0.0";
    let mlScore = "0.0";
    let finalScore = score !== undefined ? score.toFixed(2) : "0.0";
    let severity = "LOW";
    
    lines.forEach(line => {
      const idx = line.indexOf(': ');
      if (idx !== -1) {
        const lbl = line.substring(0, idx).toLowerCase();
        const val = line.substring(idx + 2);
        if (lbl.includes('baseline rule score')) ruleScore = val;
        if (lbl.includes('xgboost predicted score')) {
          mlScore = val;
          isML = true;
        }
        if (lbl.includes('constrained hybrid priority') || lbl.includes('final priority')) finalScore = val;
        if (lbl.includes('base priority')) {
          if (val.includes('CRITICAL')) severity = 'CRITICAL';
          else if (val.includes('HIGH')) severity = 'HIGH';
          else if (val.includes('MEDIUM')) severity = 'MEDIUM';
          else severity = 'LOW';
        }
      }
    });

    const floorMap: {[key: string]: number} = { CRITICAL: 8.0, HIGH: 6.5, MEDIUM: 4.5, LOW: 2.0 };
    const floor = floorMap[severity] || 2.0;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%', marginTop: '10px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
          {lines.map((line, i) => {
            const colonIndex = line.indexOf(': ');
            if (colonIndex === -1) return <div key={i} style={{ color: '#94a3b8', fontSize: '0.85rem' }}>• {line}</div>;
            const label = line.substring(0, colonIndex);
            const value = line.substring(colonIndex + 2);
            let color = '#fff';
            if (label.toLowerCase().includes('final') || label.toLowerCase().includes('hybrid')) color = '#10b981';
            if (label.toLowerCase().includes('action')) color = '#3b82f6';
            if (label.toLowerCase().includes('urgency')) color = '#f59e0b';
            if (label.toLowerCase().includes('category')) color = '#ec4899';
            return (
              <div key={i} className="glass" style={{ padding: '12px', borderRadius: '8px', borderLeft: `3px solid ${color === '#fff' ? '#475569' : color}`, background: 'rgba(255, 255, 255, 0.03)' }}>
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: '#94a3b8', letterSpacing: '0.05em', marginBottom: '4px' }}>{label}</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 'bold', color: color }}>{value}</div>
              </div>
            );
          })}
        </div>
        <div style={{ 
          padding: '12px 14px', 
          background: 'rgba(59, 130, 246, 0.07)', 
          borderRadius: '6px', 
          fontSize: '0.8rem', 
          color: '#60a5fa',
          fontFamily: 'monospace',
          border: '1px solid rgba(59, 130, 246, 0.15)',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px'
        }}>
          {isML ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontWeight: 'bold' }}>
                <span>⚙️ Constrained Hybrid Equation:</span>
                <span>FINAL = min(max(Safety Floor, XGBoost), Rule Score + 1.0)</span>
              </div>
              <div style={{ borderTop: '1px dashed rgba(59, 130, 246, 0.2)', paddingTop: '6px', display: 'flex', justifyContent: 'space-between', color: '#93c5fd', fontSize: '0.75rem' }}>
                <span>Execution Trace:</span>
                <span>min(max({floor} ({severity}), {mlScore}), {ruleScore} + 1.0) = {finalScore} / 10.0</span>
              </div>
            </>
          ) : (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontWeight: 'bold' }}>
                <span>⚙️ Rule-Based Fallback Equation:</span>
                <span>FINAL = min(10.0, Base × Age Multiplier + Boost)</span>
              </div>
              <div style={{ borderTop: '1px dashed rgba(59, 130, 246, 0.2)', paddingTop: '6px', display: 'flex', justifyContent: 'space-between', color: '#93c5fd', fontSize: '0.75rem' }}>
                <span>Execution Trace:</span>
                <span>{finalScore} / 10.0</span>
              </div>
            </>
          )}
        </div>
      </div>
    );
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
    setUploadResult(null);
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
      setUploadResult({
        case_id: selectedFile.name.replace(/\.pdf$/i, "").replace(/ /g, "_"),
        score: 1.0,
        date: "Today",
        severity: data.severity,
        summary: data.summary,
        bns_sections: data.bns_sections,
        ipc_sections: data.ipc_sections,
        priority_score: data.priority_score,
        case_type: data.case_type,
        priority_explanation: data.priority_explanation,
        similar_cases: data.similar_cases
      });
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
          {uploadResult && (
            <div style={{ marginTop: '20px' }}>
              <button 
                className="summary-btn" 
                onClick={(e) => { e.preventDefault(); setShowUploadSummary(!showUploadSummary); }}
                style={{ marginBottom: '15px', background: 'rgba(16, 185, 129, 0.1)', borderColor: '#10b981', color: '#10b981', display: 'block' }}
              >
                {showUploadSummary ? "Hide Live Summary & Agents ▲" : "View Live Summary & Agents ▼"}
              </button>
              
              {showUploadSummary && (
                <div className="summary-dropdown" style={{ background: 'rgba(20, 28, 44, 0.6)', borderLeft: '3px solid #10b981', padding: '20px' }}>
                  <div className="card-header" style={{ flexWrap: 'wrap', gap: '8px', marginBottom: '15px' }}>
                    <span className="similarity-badge" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }}>Live Processed File</span>
                    <span className="case-type-badge">{uploadResult.case_type}</span>
                    <span className={`priority-badge ${getPriorityClass(uploadResult.priority_score)}`}>
                       Priority: {uploadResult.priority_score?.toFixed(1)} — {getPriorityCategory(uploadResult.priority_score)}
                    </span>
                  </div>

                  <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#10b981'}}>🤖 Summarization Agent (Gemini-2.5-Flash)</span></strong></p>
                  <div style={{ marginBottom: '20px' }}>
                    <ProgressiveSummary summary={uploadResult.summary} />
                  </div>

                  <div className="agent-card" style={{ marginBottom: '20px', width: '100%', maxWidth: '100%' }}>
                    <div className="agent-card-title">🤖 Prioritization Agent <span className="badge-green">Active</span></div>
                    <div className="agent-card-content" style={{ display: 'block', width: '100%' }}>
                      {formatPriorityExplanation(uploadResult.priority_explanation, uploadResult.priority_score)}
                    </div>
                  </div>

                  {uploadResult.similar_cases && uploadResult.similar_cases.length > 0 && (
                    <div style={{ marginTop: '20px', paddingTop: '15px', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                      <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#10b981'}}>🔗 Related Precedents Found (Top 3 Similar Cases)</span></strong></p>
                      <div className="results-grid" style={{ gridTemplateColumns: '1fr', gap: '12px' }}>
                        {uploadResult.similar_cases.map((simCase) => (
                          <div key={simCase.case_id} className="result-card glass" style={{ opacity: 1, transform: 'none', padding: '15px', cursor: 'default' }}>
                            <div className="card-header" style={{ flexWrap: 'wrap', gap: '8px' }}>
                              <span className="similarity-badge">{(simCase.score * 100).toFixed(1)}% Match</span>
                              <span className="case-type-badge">{simCase.case_type}</span>
                              <span className={`priority-badge ${getPriorityClass(simCase.priority_score)}`}>
                                Priority: {simCase.priority_score?.toFixed(1)}
                              </span>
                            </div>
                            <h4 style={{ margin: '8px 0 4px 0', color: '#fff', fontSize: '1.05rem' }}>{formatCaseTitle(simCase.case_id)}</h4>
                            <p style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: '1.4' }}><strong>Verdict Synopsis:</strong> {simCase.summary.substring(0, 160)}...</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
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
              <div className="card-header" style={{ flexWrap: 'wrap', gap: '8px' }}>
                <span className="similarity-badge">{(r.score * 100).toFixed(1)}% Match</span>
                <span className="case-type-badge">{r.case_type}</span>
                <span className={`priority-badge ${getPriorityClass(r.priority_score)}`}>
                   Priority: {r.priority_score?.toFixed(1)} — {getPriorityCategory(r.priority_score)}
                </span>
              </div>
              <h3 className="case-title">{formatCaseTitle(r.case_id)}</h3>
              <div className="card-footer">
                <p><strong>Date:</strong> {r.date}</p>
                <button 
                  className="summary-btn" 
                  onClick={(e) => toggleSummary(r.case_id, e)}
                >
                  {expandedCase === r.case_id ? "Hide Case Insights ▲" : "⚖️ AI Insights & Agents ▼"}
                </button>
              </div>
              
              {expandedCase === r.case_id && (
                <div className="summary-dropdown" style={{ borderLeft: '3px solid #3b82f6' }}>
                  <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#3b82f6'}}>🏛️ Summarization Agent (Gemini-2.5-Flash)</span></strong></p>
                  <div style={{ marginBottom: '20px' }}>
                    <ProgressiveSummary summary={r.summary} />
                  </div>

                  <div className="agent-card" style={{ width: '100%', maxWidth: '100%' }}>
                    <div className="agent-card-title">🤖 Prioritization Agent <span className="badge-green">Active</span></div>
                    <div className="agent-card-content" style={{ display: 'block', width: '100%' }}>
                      {formatPriorityExplanation(r.priority_explanation, r.priority_score)}
                    </div>
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
