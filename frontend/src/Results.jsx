import React, { useMemo } from 'react'
import { marked } from 'marked'

// Configure marked for safe rendering
marked.setOptions({
  breaks: true,
  gfm: true,
})

const LIBRARY_COLORS = {
  requests: '#3b82f6',
  pandas: '#10b981',
  fastapi: '#f59e0b',
}

function LibraryBadge({ library }) {
  const color = LIBRARY_COLORS[library?.toLowerCase()] || 'var(--text-muted)'
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 8px',
      fontSize: 11,
      fontWeight: 600,
      fontFamily: 'var(--mono)',
      color,
      background: `${color}18`,
      border: `1px solid ${color}40`,
      borderRadius: 4,
      textTransform: 'lowercase',
    }}>
      {library}
    </span>
  )
}

function SourceCard({ source }) {
  return (
    <div style={{
      flex: '0 0 auto',
      padding: '10px 14px',
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      maxWidth: 280,
      minWidth: 200,
    }}>
      <div style={{ marginBottom: 6 }}>
        <LibraryBadge library={source.library} />
      </div>
      <div style={{
        fontSize: 13,
        fontWeight: 500,
        marginBottom: 4,
        lineHeight: 1.3,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        display: '-webkit-box',
        WebkitLineClamp: 2,
        WebkitBoxOrient: 'vertical',
      }}>
        {source.page_title}
      </div>
      <a
        href={source.source_url}
        target="_blank"
        rel="noopener noreferrer"
        style={{ fontSize: 11, color: 'var(--text-muted)' }}
      >
        View source &rarr;
      </a>
    </div>
  )
}

function MetricsBar({ metrics }) {
  if (!metrics) return null

  const items = [
    {
      label: 'Latency',
      value: `${metrics.latency_ms ?? '—'}ms`,
      color: (metrics.latency_ms ?? 0) < 200 ? 'var(--green)' : 'var(--amber)',
    },
    {
      label: 'Cache',
      value: metrics.cache_hit ? 'HIT' : 'MISS',
      color: metrics.cache_hit ? 'var(--green)' : 'var(--text-muted)',
    },
    {
      label: 'Mode',
      value: metrics.retrieval_mode || 'hybrid',
      color: 'var(--text-muted)',
    },
  ]

  return (
    <div style={{
      display: 'flex',
      gap: 20,
      padding: '8px 0',
      marginTop: 16,
      borderTop: '1px solid var(--border)',
      fontSize: 12,
      fontFamily: 'var(--mono)',
    }}>
      {items.map(({ label, value, color }) => (
        <span key={label} style={{ color: 'var(--text-muted)' }}>
          {label}:{' '}
          <span style={{ color, fontWeight: 500 }}>{value}</span>
        </span>
      ))}
    </div>
  )
}

function GlobalMetricsDashboard({ globalMetrics }) {
  if (!globalMetrics) return null

  const { cache, performance, usage } = globalMetrics

  const stats = [
    {
      label: 'Cache Hit Rate',
      value: `${(cache.hit_rate * 100).toFixed(1)}%`,
    },
    {
      label: 'Total Queries',
      value: usage.total_queries.toLocaleString(),
    },
    {
      label: 'Avg Latency',
      value: `${performance.avg_latency_ms.toFixed(0)}ms`,
    },
    {
      label: 'Est. Cost',
      value: `$${usage.estimated_cost_usd.toFixed(4)}`,
    },
  ]

  return (
    <div style={{
      marginTop: 32,
      padding: 16,
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
    }}>
      <div style={{
        fontSize: 11,
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        color: 'var(--text-muted)',
        marginBottom: 12,
      }}>
        Live Metrics
        <span style={{
          display: 'inline-block',
          width: 6,
          height: 6,
          background: 'var(--green)',
          borderRadius: '50%',
          marginLeft: 8,
          animation: 'pulse 2s infinite',
        }} />
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 16,
      }}>
        {stats.map(({ label, value }) => (
          <div key={label}>
            <div style={{
              fontSize: 18,
              fontWeight: 600,
              fontFamily: 'var(--mono)',
            }}>
              {value}
            </div>
            <div style={{
              fontSize: 11,
              color: 'var(--text-muted)',
              marginTop: 2,
            }}>
              {label}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Results({ answer, sources, metrics, isStreaming, globalMetrics }) {
  const renderedHtml = useMemo(() => {
    if (!answer) return ''
    return marked.parse(answer)
  }, [answer])

  const hasAnswer = answer.length > 0

  return (
    <div style={{ marginTop: hasAnswer ? 28 : 0 }}>
      {/* Answer section */}
      {hasAnswer && (
        <div style={{
          padding: '20px 24px',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          lineHeight: 1.7,
          fontSize: 14,
        }}>
          <div
            dangerouslySetInnerHTML={{ __html: renderedHtml }}
            style={{ wordBreak: 'break-word' }}
          />
          {isStreaming && (
            <span style={{
              display: 'inline-block',
              width: 8,
              height: 16,
              background: 'var(--accent)',
              marginLeft: 2,
              animation: 'blink 0.8s infinite',
              verticalAlign: 'text-bottom',
              borderRadius: 1,
            }} />
          )}
        </div>
      )}

      {/* Sources section */}
      {sources.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <div style={{
            fontSize: 11,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--text-muted)',
            marginBottom: 8,
          }}>
            Sources used
          </div>
          <div style={{
            display: 'flex',
            gap: 10,
            overflowX: 'auto',
            paddingBottom: 4,
          }}>
            {sources.slice(0, 5).map((source, i) => (
              <SourceCard key={i} source={source} />
            ))}
          </div>
        </div>
      )}

      {/* Per-query metrics bar */}
      {hasAnswer && !isStreaming && <MetricsBar metrics={metrics} />}

      {/* Global metrics dashboard */}
      <GlobalMetricsDashboard globalMetrics={globalMetrics} />
    </div>
  )
}
