import React, { useState, useEffect, useCallback } from 'react'
import SearchBox from './SearchBox'
import Results from './Results'

const API_BASE = '/api'

export default function App() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [globalMetrics, setGlobalMetrics] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)

  // Fetch global metrics on mount + every 30s
  const fetchGlobalMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/metrics`)
      if (res.ok) {
        setGlobalMetrics(await res.json())
      }
    } catch {
      // Silently ignore — backend may be down
    }
  }, [])

  useEffect(() => {
    fetchGlobalMetrics()
    const interval = setInterval(fetchGlobalMetrics, 30000)
    return () => clearInterval(interval)
  }, [fetchGlobalMetrics])

  const handleSearch = useCallback(async (searchQuery) => {
    setIsLoading(true)
    setAnswer('')
    setSources([])
    setMetrics(null)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/search/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, top_k: 5 }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.detail || `Search failed (${response.status})`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      setIsLoading(false)
      setIsStreaming(true)

      let buffer = ''

      const processEvent = (eventText) => {
        const trimmed = eventText.trim()
        if (!trimmed.startsWith('data: ')) return
        try {
          const data = JSON.parse(trimmed.slice(6))
          if (data.token) {
            setAnswer((prev) => prev + data.token)
          }
          if (data.done) {
            if (data.answer) setAnswer(data.answer)
            setSources(data.sources || [])
            setMetrics(data.metrics || {})
            setIsStreaming(false)
            fetchGlobalMetrics()
          }
        } catch {
          // Incomplete JSON — will retry when more data arrives
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // SSE events are separated by double newlines
        let boundary = buffer.indexOf('\n\n')
        while (boundary !== -1) {
          const event = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)
          processEvent(event)
          boundary = buffer.indexOf('\n\n')
        }
      }

      // Flush remaining buffer (final event may lack trailing \n\n)
      if (buffer.trim()) {
        processEvent(buffer)
      }

      setIsStreaming(false)
    } catch (err) {
      setError(err.message)
      setIsLoading(false)
      setIsStreaming(false)
    }
  }, [fetchGlobalMetrics])

  return (
    <div style={{
      maxWidth: 800,
      margin: '0 auto',
      padding: '40px 20px',
      minHeight: '100vh',
    }}>
      <header style={{ textAlign: 'center', marginBottom: 40 }}>
        <h1 style={{
          fontSize: 28,
          fontWeight: 600,
          letterSpacing: '-0.02em',
          marginBottom: 6,
        }}>
          <span style={{ color: 'var(--accent)' }}>{'</>'}</span>{' '}
          CodeQuery
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>
          AI-powered Python docs search with citations
        </p>
      </header>

      <SearchBox
        query={query}
        setQuery={setQuery}
        onSearch={handleSearch}
        isLoading={isLoading}
        isStreaming={isStreaming}
      />

      {error && (
        <div style={{
          marginTop: 16,
          padding: '10px 14px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid var(--red)',
          borderRadius: 6,
          color: 'var(--red)',
          fontSize: 13,
        }}>
          {error}
        </div>
      )}

      <Results
        answer={answer}
        sources={sources}
        metrics={metrics}
        isStreaming={isStreaming}
        globalMetrics={globalMetrics}
      />
    </div>
  )
}
