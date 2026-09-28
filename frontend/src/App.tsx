import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './index.css'

interface ReconciliationData {
  status: "APPROVED" | "ADJUSTED" | "FLAGGED_FOR_BENCH";
  reconciled_score: number;
  conflict_detected: boolean;
  conflict_reasons: string[];
  deliberation_notes: string;
  memo: string;
  recommendation: string;
}

interface CaseResult {
  rank?: number;
  case_id: string;
  score?: number;
  date: string;
  severity: number;
  summary: string;
  bns_sections?: string;
  ipc_sections?: string;
  priority_score?: number;
  case_type?: string;
  legal_regime?: string;
  priority_explanation?: string;
  similar_cases?: CaseResult[];
  contributions?: Record<string, number>;
  bias_score?: number | null;
  bias_details?: string;
  bias_flags?: string[];
  reconciliation?: ReconciliationData;
}

const renderContributions = (contribs: Record<string, number> | undefined) => {
  if (!contribs) return <p style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Attributions loading...</p>;
  
  const labelMap: Record<string, string> = {
    case_type: "Case Type Risk",
    num_ipc_sections: "IPC Count Impact",
    num_cpc_sections: "CPC Count Impact",
    num_precedents: "Precedents Citations",
    total_words: "Docket Length",
    case_age_days: "Pending Age Factor",
    max_severity_score: "Severity Rating",
    immediate_threat_flag: "Threat Multiplier",
    societal_impact_score: "Societal Impact Weight",
    severity_contribution: "Severity contribution",
    societal_impact_contribution: "Societal impact contribution",
    immediate_threat_contribution: "Immediate threat contribution",
    recency_contribution: "Recency contribution"
  };

  const activeContribs = Object.entries(contribs)
    .sort((a, b) => b[1] - a[1]);

  if (activeContribs.length === 0) {
    return <p style={{ fontSize: '0.85rem', color: '#cbd5e1', fontStyle: 'italic', marginTop: '6px' }}>Neutral priority profile.</p>;
  }

  return (
    <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {activeContribs.map(([key, val]) => {
        const pct = Math.min(Math.abs(val) * 10, 100);
        return (
          <div key={key} style={{ fontSize: '0.8rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px', color: '#cbd5e1' }}>
              <span>{labelMap[key] || key}</span>
              <span style={{ fontWeight: 'bold', color: '#ffffff' }}>
                {val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
              </span>
            </div>
            <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: '#ffffff', borderRadius: '3px' }} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

const renderDecisionTrace = (trace: string | undefined) => {
  const entries = (trace || "")
    .split(';')
    .map(point => point.trim())
    .filter(Boolean);
  const method = entries.find(point => point.startsWith("Evaluated using"));
  const driverOrder = [/immediate threat|liberty/i, /severity offence/i, /societal impact/i, /precedential complexity/i, /pending|litigation duration|case pending/i, /sections implicated|procedure codes cited/i, /criminal case|civil case/i];
  const drivers = entries
    .filter(point => !point.startsWith("Evaluated using"))
    .sort((a, b) => {
      const rankA = driverOrder.findIndex(pattern => pattern.test(a));
      const rankB = driverOrder.findIndex(pattern => pattern.test(b));
      return (rankA < 0 ? driverOrder.length : rankA) - (rankB < 0 ? driverOrder.length : rankB);
    })
    .slice(0, 3);

  return (
    <div>
      {drivers.length > 0 ? (
        <ul style={{ margin: '6px 0 0', paddingLeft: '20px' }}>
          {drivers.map((point, index) => <li key={index} style={{ marginBottom: '4px' }}>{point}</li>)}
        </ul>
      ) : <p style={{ margin: '6px 0', color: '#94a3b8' }}>No additional priority signals were detected.</p>}
      {method && <p style={{ margin: '6px 0 0', fontSize: '0.78rem', color: '#94a3b8' }}>Scoring method: {method.replace("Evaluated using ", "")}</p>}
    </div>
  );
};

const isRuleBasedTrace = (trace: string | undefined) =>
  Boolean(trace?.includes("baseline Rule-Based framework"));

const renderPriorityComponents = (r: CaseResult) => (
  <div style={{ marginTop: '12px', borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '8px' }}>
    <strong style={{ display: 'block', marginBottom: '6px', color: '#cbd5e1' }}>
      {isRuleBasedTrace(r.priority_explanation) ? 'Priority Score Components:' : 'Model Feature Attributions:'}
    </strong>
    {isRuleBasedTrace(r.priority_explanation) && (
      <p style={{ margin: '0 0 8px', fontSize: '0.78rem', color: '#94a3b8' }}>
        Score = (severity / 10 * 3.5) + (impact / 5 * 2.5) + (threat * 3) + recency; capped at 10. Values shown are score points.
      </p>
    )}
    {renderContributions(r.contributions)}
  </div>
);

const renderBiasCard = (r: CaseResult) => (
  <div className="agent-card" style={{ gridColumn: 'span 2' }}>
    <div className="agent-card-title">
      ⚖️ Bias & Fairness Scan 
      {r.bias_score === null || r.bias_score === undefined ? (
        <span className="badge-orange" style={{ marginLeft: '8px' }}>Pending Audit</span>
      ) : (
        <span className={`badge-${r.bias_score >= 0.85 ? 'green' : 'red'}`} style={{ marginLeft: '8px' }}>
          {r.bias_score >= 0.85 ? 'Neutral' : 'Profiling Alert'}
        </span>
      )}
    </div>
    <div className="agent-card-content">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <strong>Demographic Neutrality Index:</strong>
        <span style={{ fontSize: '1.2rem', fontWeight: 'bold', color: r.bias_score === null || r.bias_score === undefined ? '#94a3b8' : (r.bias_score >= 0.85 ? '#86efac' : '#fca5a5') }}>
          {r.bias_score === null || r.bias_score === undefined ? 'N/A' : `${(r.bias_score * 100).toFixed(0)}%`}
        </span>
      </div>
      <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '4px', overflow: 'hidden', marginBottom: '10px' }}>
        <div style={{ 
          width: r.bias_score === null || r.bias_score === undefined ? '0%' : `${r.bias_score * 100}%`, 
          height: '100%', 
          background: r.bias_score === null || r.bias_score === undefined ? '#475569' : (r.bias_score >= 0.85 ? '#10b981' : '#ef4444'), 
          borderRadius: '4px' 
        }} />
      </div>
      <strong>Audit Log:</strong> {r.bias_details || "Automated audit unavailable. Manual review recommended."}
      {r.bias_flags && r.bias_flags.length > 0 && (
        <div style={{ marginTop: '10px', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '8px' }}>
          <strong style={{ color: '#fca5a5', fontSize: '0.8rem' }}>Flagged Contexts:</strong>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
            {r.bias_flags.map((flag, idx) => (
              <span key={idx} style={{ fontSize: '0.75rem', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '2px 6px', borderRadius: '4px', color: '#fca5a5' }}>
                "{flag}"
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  </div>
);

const renderReconciliationCard = (
  r: CaseResult,
  onRunAudit?: (caseId: string) => void,
  isAuditing?: boolean
) => {
  if (!r.reconciliation || !r.reconciliation.status) {
    if (!onRunAudit) return null;
    return (
      <div className="agent-card" style={{ gridColumn: 'span 2', background: 'rgba(99, 102, 241, 0.05)', border: '1px dashed rgba(99, 102, 241, 0.3)', textAlign: 'center', padding: '16px' }}>
        <p style={{ fontSize: '0.88rem', color: '#c7d2fe', marginBottom: '10px' }}>
          🏛️ <strong>LangGraph Judicial Review & Reconciliation Agent</strong> is ready to audit this docket record.
        </p>
        <button
          className="search-btn"
          style={{ margin: '0 auto', padding: '8px 18px', fontSize: '0.85rem' }}
          onClick={(e) => {
            e.stopPropagation();
            onRunAudit(r.case_id);
          }}
          disabled={isAuditing}
        >
          {isAuditing ? "⚡ Running LangGraph State Machine..." : "⚡ Run LangGraph Judicial Audit"}
        </button>
      </div>
    );
  }

  const { status, reconciled_score, conflict_detected, conflict_reasons, deliberation_notes, memo, recommendation } = r.reconciliation;
  const statusColor = status === "APPROVED" ? "#10b981" : status === "ADJUSTED" ? "#f59e0b" : "#ef4444";
  const statusBg = status === "APPROVED" ? "rgba(16, 185, 129, 0.15)" : status === "ADJUSTED" ? "rgba(245, 158, 11, 0.15)" : "rgba(239, 68, 68, 0.15)";
  const statusBadgeText = status === "APPROVED" ? "Validated (Consistent)" : status === "ADJUSTED" ? `Adjusted to ${reconciled_score?.toFixed(1)}` : "Flagged for Bench Caution";

  return (
    <div className="agent-card" style={{ gridColumn: 'span 2', borderLeft: `3px solid ${statusColor}`, background: 'rgba(255, 255, 255, 0.03)' }}>
      <div className="agent-card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <span>🏛️ Judicial Review & Reconciliation Agent <span style={{ fontSize: '0.75rem', opacity: 0.7, fontWeight: 'normal' }}>(LangGraph State Machine)</span></span>
        <span style={{ fontSize: '0.75rem', padding: '3px 8px', borderRadius: '4px', background: statusBg, color: statusColor, fontWeight: 600, border: `1px solid ${statusColor}40` }}>
          {statusBadgeText}
        </span>
      </div>
      <div className="agent-card-content" style={{ marginTop: '10px' }}>
        <div style={{ marginBottom: '10px', padding: '8px 12px', borderRadius: '6px', background: 'rgba(99, 102, 241, 0.12)', border: '1px solid rgba(99, 102, 241, 0.25)', fontSize: '0.85rem', color: '#c7d2fe' }}>
          <strong>Scheduling Directive:</strong> {recommendation || "Standard Docket Scheduling"}
        </div>

        {conflict_detected && conflict_reasons && conflict_reasons.length > 0 && (
          <div style={{ marginBottom: '10px', padding: '10px', borderRadius: '6px', background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
            <strong style={{ color: '#fca5a5', fontSize: '0.82rem', display: 'block', marginBottom: '6px' }}>⚠️ Reconciled Contradictions:</strong>
            {conflict_reasons.map((reason, idx) => (
              <p key={idx} style={{ fontSize: '0.8rem', color: '#fecaca', margin: '3px 0' }}>• {reason}</p>
            ))}
          </div>
        )}

        {deliberation_notes && (
          <div style={{ marginBottom: '10px', fontSize: '0.82rem', color: '#cbd5e1' }}>
            <strong style={{ color: '#93c5fd' }}>Agent Deliberation Trace:</strong> {deliberation_notes}
          </div>
        )}

        {memo && (
          <div style={{ marginTop: '10px', borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '10px' }}>
            <strong style={{ fontSize: '0.82rem', color: '#cbd5e1', display: 'block', marginBottom: '6px' }}>📋 Bench Triage Memo:</strong>
            <p style={{ fontSize: '0.82rem', color: '#e2e8f0', lineHeight: 1.5, whiteSpace: 'pre-line' }}>{memo}</p>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState<'search' | 'queue' | 'batch'>('search');

  // Tab 1: Search State
  const [query, setQuery] = useState('')
  const [yearFilter, setYearFilter] = useState('')
  const [results, setResults] = useState<CaseResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedCase, setExpandedCase] = useState<string | null>(null)
  const [topK, setTopK] = useState<number>(6)

  // Single PDF Upload State
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadResult, setUploadResult] = useState<CaseResult | null>(null)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [showUploadSummary, setShowUploadSummary] = useState(false)

  // Tab 2: Pending Queue State
  const [queueResults, setQueueResults] = useState<CaseResult[]>([]);
  const [queueLoading, setQueueLoading] = useState(false);
  const [queueCaseType, setQueueCaseType] = useState<string>('all');
  const [queueRegime, setQueueRegime] = useState<string>('all');

  // Tab 3: Batch Upload State
  const [batchFiles, setBatchFiles] = useState<FileList | null>(null);
  const [batchResults, setBatchResults] = useState<CaseResult[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);

  // On-Demand LangGraph Reconciliation Audit State
  const [auditingCaseId, setAuditingCaseId] = useState<string | null>(null);

  const handleRunReconciliation = async (caseId: string) => {
    setAuditingCaseId(caseId);
    try {
      const res = await fetch(`http://localhost:8000/cases/${encodeURIComponent(caseId)}/reconcile`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error("Reconciliation failed");
      const data = await res.json();
      
      setQueueResults(prev => prev.map(c => c.case_id === caseId ? { ...c, reconciliation: data.reconciliation } : c));
      setResults(prev => prev.map(c => c.case_id === caseId ? { ...c, reconciliation: data.reconciliation } : c));
      setBatchResults(prev => prev.map(c => c.case_id === caseId ? { ...c, reconciliation: data.reconciliation } : c));
      setUploadResult(prev => prev && prev.case_id === caseId ? { ...prev, reconciliation: data.reconciliation } : prev);
    } catch (err: unknown) {
      console.error("Audit error:", err);
    } finally {
      setAuditingCaseId(null);
    }
  };

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

  const getRankBadgeClass = (score?: number) => {
    if (score === undefined || score === null) return "rank-badge-low";
    if (score >= 8.0) return "rank-badge-critical";
    if (score >= 6.0) return "rank-badge-high";
    if (score >= 4.0) return "rank-badge-med";
    return "rank-badge-low";
  }

  const toggleSummary = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedCase(expandedCase === id ? null : id);
  }

  // Fetch Pending Queue
  const fetchPendingQueue = async (cType = queueCaseType, lRegime = queueRegime) => {
    setQueueLoading(true);
    setError(null);
    try {
      let url = `http://localhost:8000/cases/pending?limit=50`;
      if (cType !== 'all') url += `&case_type=${cType}`;
      if (lRegime !== 'all') url += `&legal_regime=${lRegime}`;

      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch pending queue");
      const data = await res.json();
      setQueueResults(data.results);
    } catch (err: unknown) {
      if (err instanceof Error) setError(err.message);
    } finally {
      setQueueLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'queue') {
      fetchPendingQueue(queueCaseType, queueRegime);
    }
  }, [activeTab, queueCaseType, queueRegime]);

  // Handle Semantic Search
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query) return;

    setLoading(true)
    setError(null)
    setResults([])

    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, yearFilter: yearFilter, topK: topK })
      });

      if (!response.ok) throw new Error("Failed to fetch search results from backend");
      const data = await response.json();
      setResults(data.results);
    } catch (err: unknown) {
      if (err instanceof Error) setError(err.message);
      else setError("An unknown error occurred.");
    } finally {
      setLoading(false)
    }
  }

  // Handle Single PDF Upload
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

      if (!response.ok) throw new Error("Failed to process PDF upload.");

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
        similar_cases: data.similar_cases,
        contributions: data.contributions,
        bias_score: data.bias_score,
        bias_details: data.bias_details,
        bias_flags: data.bias_flags,
        reconciliation: data.reconciliation
      });
      setShowUploadSummary(true);
    } catch (err: unknown) {
      if (err instanceof Error) setError(err.message);
      else setError("An error occurred during upload.");
    } finally {
      setUploadLoading(false);
    }
  }

  // Handle Batch PDF Upload
  const handleBatchUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!batchFiles || batchFiles.length === 0) return;

    setBatchLoading(true);
    setBatchResults([]);
    setError(null);

    const formData = new FormData();
    for (let i = 0; i < batchFiles.length; i++) {
      formData.append('files', batchFiles[i]);
    }

    try {
      const response = await fetch('http://localhost:8000/batch_summarize_upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to process batch PDF upload.");

      const data = await response.json();
      setBatchResults(data.results);
    } catch (err: unknown) {
      if (err instanceof Error) setError(err.message);
      else setError("An error occurred during batch upload.");
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="background-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>

      <header className="header glass">
        <h1 className="title animate-fade-in">⚖️ Judicial AI</h1>
        <p className="subtitle">Precedent Search & Intelligent Docket Prioritization System</p>
      </header>

      {/* Navigation Tabs */}
      <nav className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          🔍 Precedent Search
        </button>
        <button 
          className={`tab-btn ${activeTab === 'queue' ? 'active' : ''}`}
          onClick={() => setActiveTab('queue')}
        >
          📋 Pending Priority Queue
        </button>
        <button 
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          📁 Batch File Prioritizer
        </button>
      </nav>

      <main className="main-content">
        {error && (
          <div className="error-card glass" style={{ marginBottom: '20px' }}>
            ⚠️ {error} - Ensure FastAPI is running on Port 8000.
          </div>
        )}

        {/* TAB 1: PRECEDENT SEARCH */}
        {activeTab === 'search' && (
          <>
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
                  {loading ? <span className="spinner"></span> : "Search"}
                </button>
              </form>
            </div>

            <div className="upload-container glass" style={{ marginTop: '20px', marginBottom: '20px', padding: '20px' }}>
              <h3 style={{ marginBottom: '10px', fontSize: '1.2rem', color: '#fff' }}>📄 Live Single Document Summarization</h3>
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
                        <span className={`severity-badge level-${uploadResult.severity > 7 ? 'high' : uploadResult.severity > 4 ? 'med' : 'low'}`}>
                          Severity {uploadResult.severity}
                        </span>
                        <span className="case-type-badge">{uploadResult.case_type}</span>
                        <span className={`priority-badge ${getPriorityClass(uploadResult.priority_score)}`}>
                          Priority: {uploadResult.priority_score?.toFixed(1)} ({getPriorityCategory(uploadResult.priority_score)})
                        </span>
                      </div>

                      <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#10b981'}}>🤖 Summarization Agent (Gemini-2.5-Flash)</span></strong></p>
                      <div className="markdown-body" style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                        <ReactMarkdown>{uploadResult.summary}</ReactMarkdown>
                      </div>

                      <p style={{ marginBottom: '10px' }}><strong><span style={{ fontSize: '1.1rem', color: '#818cf8'}}>⚙️ Multi-Agent Intelligence Scan</span></strong></p>
                      <div className="agent-grid">
                        <div className="agent-card">
                          <div className="agent-card-title">🤖 Prioritization Agent <span className="badge-green">Active</span></div>
                          <div className="agent-card-content">
                            <strong>Score:</strong> {uploadResult.priority_score?.toFixed(2)} / 10.0<br/>
                            <strong>Category:</strong> {getPriorityCategory(uploadResult.priority_score)}
                            {renderPriorityComponents(uploadResult)}
                          </div>
                        </div>
                        <div className="agent-card">
                          <div className="agent-card-title">🔍 Explainability Agent <span className="badge-green">Active</span></div>
                          <div className="agent-card-content">
                            <strong>Key Reasons:</strong>
                            {renderDecisionTrace(uploadResult.priority_explanation)}
                          </div>
                        </div>
                        {renderBiasCard(uploadResult)}
                        {renderReconciliationCard(uploadResult, handleRunReconciliation, auditingCaseId === uploadResult.case_id)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="results-grid">
              {results.map((r, index) => (
                <div key={r.case_id} className="result-card glass" style={{ animationDelay: `${index * 0.1}s` }}>
                  <div className="card-header" style={{ flexWrap: 'wrap', gap: '8px' }}>
                    <span className="similarity-badge">{((r.score || 0) * 100).toFixed(1)}% Match</span>
                    <span className={`severity-badge level-${r.severity > 7 ? 'high' : r.severity > 4 ? 'med' : 'low'}`}>
                      Severity {r.severity}
                    </span>
                    <span className="case-type-badge">{r.case_type}</span>
                    <span className={`priority-badge ${getPriorityClass(r.priority_score)}`}>
                      Priority: {r.priority_score?.toFixed(1)} ({getPriorityCategory(r.priority_score)})
                    </span>
                  </div>
                  <h3 className="case-title">{r.case_id.replace(/_/g, ' ')}</h3>
                  <div className="card-footer">
                    <p><strong>Date:</strong> {r.date}</p>
                    <button className="summary-btn" onClick={(e) => toggleSummary(r.case_id, e)}>
                      {expandedCase === r.case_id ? "Hide Case Insights ▲" : "⚖️ AI Insights & Agents ▼"}
                    </button>
                  </div>
                  
                  {expandedCase === r.case_id && (
                    <div className="summary-dropdown" style={{ borderLeft: '3px solid #3b82f6' }}>
                      <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#3b82f6'}}>🏛️ Summarization Agent</span></strong></p>
                      <div className="markdown-body" style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                        <ReactMarkdown>{r.summary}</ReactMarkdown>
                      </div>

                      <p style={{ marginBottom: '10px' }}><strong><span style={{ fontSize: '1.1rem', color: '#818cf8'}}>⚙️ Multi-Agent Intelligence Scan</span></strong></p>
                      <div className="agent-grid">
                        <div className="agent-card">
                          <div className="agent-card-title">🤖 Prioritization Agent <span className="badge-green">Active</span></div>
                          <div className="agent-card-content">
                            <strong>Score:</strong> {r.priority_score?.toFixed(2)} / 10.0<br/>
                            <strong>Category:</strong> {getPriorityCategory(r.priority_score)}
                            {renderPriorityComponents(r)}
                          </div>
                        </div>
                        <div className="agent-card">
                          <div className="agent-card-title">🔍 Explainability Agent <span className="badge-green">Active</span></div>
                          <div className="agent-card-content">
                            <strong>Key Reasons:</strong>
                            {renderDecisionTrace(r.priority_explanation)}
                          </div>
                        </div>
                        {renderBiasCard(r)}
                        {renderReconciliationCard(r, handleRunReconciliation, auditingCaseId === r.case_id)}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {/* TAB 2: PENDING PRIORITY QUEUE */}
        {activeTab === 'queue' && (
          <div>
            <div className="filter-bar glass" style={{ padding: '15px 20px' }}>
              <label style={{ fontSize: '0.9rem', color: '#94a3b8', fontWeight: 600 }}>Filter Queue:</label>
              <select 
                className="filter-select" 
                value={queueCaseType}
                onChange={(e) => setQueueCaseType(e.target.value)}
              >
                <option value="all">All Case Types</option>
                <option value="criminal">Criminal Cases</option>
                <option value="civil">Civil Cases</option>
              </select>

              <select 
                className="filter-select" 
                value={queueRegime}
                onChange={(e) => setQueueRegime(e.target.value)}
              >
                <option value="all">All Legal Regimes</option>
                <option value="ipc">IPC Regime</option>
                <option value="bns">BNS Regime</option>
              </select>
            </div>

            {queueLoading ? (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <span className="spinner" style={{ width: '40px', height: '40px', borderWidth: '4px', margin: '0 auto' }}></span>
                <p style={{ marginTop: '15px', color: '#94a3b8' }}>Ordering pending docket cases by priority score...</p>
              </div>
            ) : (
              <div className="results-grid">
                {queueResults.map((r) => (
                  <div key={r.case_id} className="result-card glass" style={{ opacity: 1, transform: 'none' }}>
                    <div className="card-header" style={{ flexWrap: 'wrap', gap: '8px' }}>
                      <span className={`rank-badge ${getRankBadgeClass(r.priority_score)}`}>
                        #{r.rank} RANK
                      </span>
                      <span className={`priority-badge ${getPriorityClass(r.priority_score)}`}>
                        Priority: {r.priority_score?.toFixed(1)} ({getPriorityCategory(r.priority_score)})
                      </span>
                      <span className="case-type-badge">{r.case_type}</span>
                      <span className="severity-badge" style={{ background: 'rgba(99, 102, 241, 0.1)', color: '#818cf8', borderColor: 'rgba(99, 102, 241, 0.2)' }}>
                        Regime: {r.legal_regime}
                      </span>
                    </div>

                    <h3 className="case-title">{r.case_id.replace(/_/g, ' ')}</h3>
                    
                    <div className="card-footer">
                      <p><strong>Date:</strong> {r.date}</p>
                      <button className="summary-btn" onClick={(e) => toggleSummary(r.case_id, e)}>
                        {expandedCase === r.case_id ? "Hide Case Insights ▲" : "⚖️ AI Insights & Agents ▼"}
                      </button>
                    </div>

                    {expandedCase === r.case_id && (
                      <div className="summary-dropdown" style={{ borderLeft: '3px solid #ef4444' }}>
                        <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#ef4444'}}>🏛️ Case Synopsis & Priority Breakdown</span></strong></p>
                        <div className="markdown-body" style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                          <ReactMarkdown>{r.summary}</ReactMarkdown>
                        </div>

                        <div className="agent-grid">
                          <div className="agent-card">
                            <div className="agent-card-title">🤖 Prioritization Agent <span className="badge-green">Active</span></div>
                            <div className="agent-card-content">
                              <strong>Queue Priority Rank:</strong> #{r.rank}<br/>
                              <strong>Score:</strong> {r.priority_score?.toFixed(2)} / 10.0<br/>
                              <strong>Category:</strong> {getPriorityCategory(r.priority_score)}
                              {renderPriorityComponents(r)}
                            </div>
                          </div>
                          <div className="agent-card">
                            <div className="agent-card-title">🔍 Explainability Agent <span className="badge-green">Active</span></div>
                            <div className="agent-card-content">
                              <strong>Key Reasons:</strong>
                              {renderDecisionTrace(r.priority_explanation)}
                            </div>
                          </div>
                          {renderBiasCard(r)}
                          {renderReconciliationCard(r, handleRunReconciliation, auditingCaseId === r.case_id)}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: BATCH FILE PRIORITIZER */}
        {activeTab === 'batch' && (
          <div>
            <div className="upload-container glass" style={{ padding: '25px', marginBottom: '25px' }}>
              <h3 style={{ marginBottom: '12px', fontSize: '1.3rem', color: '#fff' }}>📁 Batch File Prioritizer & Ranker</h3>
              <p style={{ color: '#94a3b8', fontSize: '0.9rem', marginBottom: '20px' }}>
                Select multiple case PDF dockets at once. The multi-agent pipeline will process, summarize, score, and automatically rank them from highest priority to lowest priority.
              </p>

              <form onSubmit={handleBatchUploadSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                <input 
                  type="file" 
                  accept="application/pdf"
                  multiple
                  onChange={(e) => setBatchFiles(e.target.files)}
                  className="input-field"
                />
                
                <button 
                  type="submit" 
                  className="search-btn" 
                  disabled={!batchFiles || batchFiles.length === 0 || batchLoading}
                  style={{ background: 'linear-gradient(135deg, #6366f1, #4f46e5)', height: '54px' }}
                >
                  {batchLoading ? (
                    <span style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span className="spinner"></span>
                      Processing {batchFiles ? batchFiles.length : 0} PDFs in Batch...
                    </span>
                  ) : (
                    `Upload & Rank ${batchFiles ? batchFiles.length : 0} PDF Files`
                  )}
                </button>
              </form>
            </div>

            {batchResults.length > 0 && (
              <div>
                <h3 style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '15px' }}>
                  📊 Batch Priority Ranking Results ({batchResults.length} Files Processed)
                </h3>

                <div className="results-grid">
                  {batchResults.map((r) => (
                    <div key={r.case_id} className="result-card glass" style={{ opacity: 1, transform: 'none' }}>
                      <div className="card-header" style={{ flexWrap: 'wrap', gap: '8px' }}>
                        <span className={`rank-badge ${getRankBadgeClass(r.priority_score)}`}>
                          #{r.rank} PRIORITY RANK
                        </span>
                        <span className={`priority-badge ${getPriorityClass(r.priority_score)}`}>
                          Priority: {r.priority_score?.toFixed(1)} ({getPriorityCategory(r.priority_score)})
                        </span>
                        <span className="case-type-badge">{r.case_type}</span>
                      </div>

                      <h3 className="case-title">{(r.case_id || 'Uploaded_Case').replace(/_/g, ' ')}</h3>

                      <div className="card-footer">
                        <p><strong>File Name:</strong> {r.case_id || 'Uploaded Case'}.pdf</p>
                        <button className="summary-btn" onClick={(e) => toggleSummary(r.case_id || 'Uploaded_Case', e)}>
                          {expandedCase === (r.case_id || 'Uploaded_Case') ? "Hide Case Insights ▲" : "⚖️ AI Insights & Agents ▼"}
                        </button>
                      </div>

                      {expandedCase === r.case_id && (
                        <div className="summary-dropdown" style={{ borderLeft: '3px solid #6366f1' }}>
                          <p style={{ marginBottom: '15px' }}><strong><span style={{ fontSize: '1.1rem', color: '#818cf8'}}>🏛️ Batch File Summary</span></strong></p>
                          <div className="markdown-body" style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                            <ReactMarkdown>{r.summary}</ReactMarkdown>
                          </div>

                          <div className="agent-grid">
                            <div className="agent-card">
                              <div className="agent-card-title">🤖 Prioritization Agent <span className="badge-green">Active</span></div>
                              <div className="agent-card-content">
                                <strong>Score:</strong> {r.priority_score?.toFixed(2)} / 10.0<br/>
                                <strong>Category:</strong> {getPriorityCategory(r.priority_score)}
                                {renderPriorityComponents(r)}
                              </div>
                            </div>
                            <div className="agent-card">
                              <div className="agent-card-title">🔍 Explainability Agent <span className="badge-green">Active</span></div>
                              <div className="agent-card-content">
                                <strong>Key Reasons:</strong>
                                {renderDecisionTrace(r.priority_explanation)}
                              </div>
                            </div>
                            {renderBiasCard(r)}
                            {renderReconciliationCard(r, handleRunReconciliation, auditingCaseId === r.case_id)}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
