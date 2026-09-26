import { useRef, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const fileInputRef = useRef(null)

  const [video, setVideo] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragActive, setDragActive] = useState(false)

  const selectVideo = (file) => {
    if (!file) return

    if (!file.type.startsWith('video/')) {
      setError('Please select a valid echocardiogram video.')
      return
    }

    setVideo(file)
    setResult(null)
    setError('')
  }

  const handleFileChange = (event) => {
    selectVideo(event.target.files?.[0])
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    selectVideo(event.dataTransfer.files?.[0])
  }

  const handleAnalyze = async () => {
    if (!video) {
      setError('Please select an echocardiogram video first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', video)

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Analysis failed (${response.status})`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Unable to connect to the analysis server.')
    } finally {
      setLoading(false)
    }
  }

  const removeVideo = () => {
    setVideo(null)
    setResult(null)
    setError('')

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return ''
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-name">Cardiac Function AI</div>
          <div className="brand-divider" />
          <div className="brand-subtitle">Echocardiogram Analysis</div>
        </div>

        <div className="prototype-label">
          <span className="status-dot" />
          Research Prototype
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="hero-grid" />

          <div className="hero-content">
            <div className="eyebrow">
              <span className="eyebrow-line" />
              CARDIAC FUNCTION ASSESSMENT
              <span className="eyebrow-line" />
            </div>

            <h1>
              Automated cardiac function
              <span> assessment from echocardiography.</span>
            </h1>

            <p>
              Analyze echocardiogram videos to estimate left-ventricular
              function and generate key cardiac measurements.
            </p>
          </div>
        </section>

        <section className="workspace">
          <div className="analysis-panel">
            <div className="panel-top">
              <div>
                <span className="panel-index">01</span>
                <div>
                  <h2>Upload echocardiogram</h2>
                  <p>Provide an echo video to begin the analysis.</p>
                </div>
              </div>

              <span className="panel-status">READY</span>
            </div>

            <input
              ref={fileInputRef}
              className="hidden-input"
              type="file"
              accept="video/*"
              onChange={handleFileChange}
            />

            {!video ? (
              <div
                className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
                onDragOver={(event) => {
                  event.preventDefault()
                  setDragActive(true)
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="scan-decoration">
                  <span />
                  <span />
                  <span />
                </div>

                <div className="upload-symbol">↑</div>

                <h3>Drop echocardiogram video</h3>

                <p>
                  Drag and drop your file here or
                  <strong> browse</strong>
                </p>

                <span className="file-hint">
                  Video files supported
                </span>
              </div>
            ) : (
              <div className="selected-file">
                <div className="file-visual">
                  <div className="play-symbol">▶</div>
                  <span className="file-type">VIDEO</span>
                </div>

                <div className="file-details">
                  <span className="file-label">SELECTED STUDY</span>
                  <strong>{video.name}</strong>
                  <span>{formatFileSize(video.size)}</span>
                </div>

                <button
                  type="button"
                  className="remove-button"
                  onClick={removeVideo}
                >
                  Remove
                </button>
              </div>
            )}

            {error && (
              <div className="error-message">
                <span>!</span>
                {error}
              </div>
            )}

            <button
              type="button"
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={!video || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Processing echocardiogram
                  <span className="loading-dots">...</span>
                </>
              ) : (
                <>
                  Analyze echocardiogram
                  <span className="button-arrow">↗</span>
                </>
              )}
            </button>
          </div>

          {result && (
            <section className="results-panel">
              <div className="results-top">
                <div>
                  <span className="panel-index">02</span>
                  <div>
                    <span className="results-eyebrow">ANALYSIS COMPLETE</span>
                    <h2>Cardiac function results</h2>
                  </div>
                </div>

                <div className="completed-badge">
                  <span>✓</span>
                  Complete
                </div>
              </div>

              <div className="ef-section">
                <div className="ef-background-grid" />

                <div className="ef-content">
                  <span className="metric-label">LEFT VENTRICULAR FUNCTION</span>

                  <div className="ef-value">
                    {result.ef_percent}
                    <span>%</span>
                  </div>

                  <div className="ef-title">Ejection Fraction</div>

                  <p>
                    Estimated left-ventricular ejection fraction from the
                    analyzed echocardiogram.
                  </p>
                </div>

                <div className="waveform">
                  <svg
                    viewBox="0 0 300 90"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    aria-hidden="true"
                  >
                    <path
                      d="M0 47H43L51 46L59 48L67 47L75 46L82 47L90 47L101 46L110 47L118 47L126 46L136 47L145 47L153 46L162 47L170 47L178 46L187 47L196 47L205 46L213 47L221 47L230 46L238 47L247 47L255 46L264 47L273 47L281 46L290 47L300 47"
                      stroke="currentColor"
                      strokeWidth="1.5"
                    />
                    <path
                      d="M0 47H32L42 47L48 44L54 50L60 47L67 47L76 47L84 47L92 47L99 46L105 47L112 47L120 47L128 47L135 47L142 47L149 47L157 47L164 47L171 47L179 47L186 47L193 47L201 47L208 47L215 47L223 47L230 47L237 47L245 47L252 47L260 47L268 47L276 47L284 47L292 47L300 47"
                      stroke="currentColor"
                      strokeWidth="2"
                    />
                  </svg>
                  <span>VENTRICULAR ANALYSIS</span>
                </div>
              </div>

              <div className="metrics-section">
                <div className="metric-card">
                  <span className="metric-number">01</span>
                  <span className="metric-label">EDV</span>
                  <strong>
                    {result.edv}
                    <small>mL</small>
                  </strong>
                  <span className="metric-description">
                    End-diastolic volume
                  </span>
                </div>

                <div className="metric-card">
                  <span className="metric-number">02</span>
                  <span className="metric-label">ESV</span>
                  <strong>
                    {result.esv}
                    <small>mL</small>
                  </strong>
                  <span className="metric-description">
                    End-systolic volume
                  </span>
                </div>

                <div className="metric-card">
                  <span className="metric-number">03</span>
                  <span className="metric-label">ED FRAME</span>
                  <strong>{result.ed_frame}</strong>
                  <span className="metric-description">
                    End-diastolic frame
                  </span>
                </div>

                <div className="metric-card">
                  <span className="metric-number">04</span>
                  <span className="metric-label">ES FRAME</span>
                  <strong>{result.es_frame}</strong>
                  <span className="metric-description">
                    End-systolic frame
                  </span>
                </div>
              </div>

              <div className="risk-section">
                <div className="risk-marker">!</div>

                <div className="risk-content">
                  <div>
                    <span className="metric-label">RISK FLAG</span>
                    <strong>{result.risk_flag}</strong>
                  </div>

                  <p>
                    Automated prototype output. This result should not be
                    used as a clinical diagnosis.
                  </p>
                </div>
              </div>
            </section>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Cardiac Function AI Assistant</span>
        <span>Research prototype · Automated analysis</span>
      </footer>
    </div>
  )
}

export default App