import React from 'react'

const EXAMPLE_QUERIES = [
  'How do I make a POST request?',
  'How do I read a CSV with pandas?',
  'How do I define a FastAPI route?',
]

export default function SearchBox({ query, setQuery, onSearch, isLoading, isStreaming }) {
  const disabled = isLoading || isStreaming

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!query.trim() || disabled) return
    onSearch(query.trim())
  }

  const handleChipClick = (example) => {
    setQuery(example)
    onSearch(example)
  }

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 8 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything about Python libraries..."
          disabled={disabled}
          style={{
            flex: 1,
            height: 48,
            padding: '0 16px',
            fontSize: 15,
            fontFamily: 'var(--sans)',
            background: 'var(--surface)',
            color: 'var(--text)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            outline: 'none',
            transition: 'border-color 0.15s',
          }}
          onFocus={(e) => (e.target.style.borderColor = 'var(--accent)')}
          onBlur={(e) => (e.target.style.borderColor = 'var(--border)')}
        />
        <button
          type="submit"
          disabled={disabled || !query.trim()}
          style={{
            height: 48,
            padding: '0 24px',
            fontSize: 14,
            fontWeight: 500,
            fontFamily: 'var(--sans)',
            background: disabled ? 'var(--border)' : 'var(--accent)',
            color: disabled ? 'var(--text-muted)' : '#fff',
            border: 'none',
            borderRadius: 8,
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'background 0.15s',
            whiteSpace: 'nowrap',
          }}
          onMouseEnter={(e) => {
            if (!disabled) e.target.style.background = 'var(--accent-hover)'
          }}
          onMouseLeave={(e) => {
            if (!disabled) e.target.style.background = 'var(--accent)'
          }}
        >
          {isLoading ? 'Searching...' : isStreaming ? 'Streaming...' : 'Search'}
        </button>
      </form>

      {/* Streaming indicator */}
      {isStreaming && (
        <div style={{
          marginTop: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          color: 'var(--accent)',
          fontSize: 12,
        }}>
          <span style={{
            display: 'inline-block',
            width: 6,
            height: 14,
            background: 'var(--accent)',
            animation: 'blink 1s infinite',
          }} />
          Streaming response...
        </div>
      )}

      {/* Example query chips */}
      {!isStreaming && !isLoading && (
        <div style={{
          marginTop: 12,
          display: 'flex',
          flexWrap: 'wrap',
          gap: 8,
        }}>
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              onClick={() => handleChipClick(example)}
              style={{
                padding: '5px 12px',
                fontSize: 12,
                fontFamily: 'var(--sans)',
                color: 'var(--text-muted)',
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: 16,
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}
              onMouseEnter={(e) => {
                e.target.style.borderColor = 'var(--accent)'
                e.target.style.color = 'var(--text)'
              }}
              onMouseLeave={(e) => {
                e.target.style.borderColor = 'var(--border)'
                e.target.style.color = 'var(--text-muted)'
              }}
            >
              {example}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
