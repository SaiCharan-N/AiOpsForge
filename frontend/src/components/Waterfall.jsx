import { fmtMs } from '../lib/metricsApi.js'

/** Per-node timeline ("flamegraph") for one run. One lane per node
 * execution, positioned by start time and split into what the time was
 * spent on: LLM generation, MCP tools, DB/memory, embeddings, and "other"
 * (everything unaccounted for — Python, event emission, waiting).
 */
export const SEGMENTS = [
  { key: 'llm_ms', label: 'LLM', color: '#6366f1' },
  { key: 'tool_ms', label: 'Tools (MCP)', color: '#f59e0b' },
  { key: 'db_ms', label: 'DB / memory', color: '#34d399' },
  { key: 'embed_ms', label: 'Embeddings', color: '#38bdf8' },
  { key: 'other_ms', label: 'Other', color: '#4b5563' },
]

export default function Waterfall({ nodes, totalMs }) {
  if (!nodes?.length) return <div className="chart-empty">No node timings recorded</div>
  const span = Math.max(totalMs || 0, ...nodes.map((n) => n.start_ms + n.duration_ms), 1)
  return (
    <div className="waterfall">
      <div className="waterfall-legend">
        {SEGMENTS.map((s) => (
          <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>
        ))}
      </div>
      {nodes.map((n, i) => {
        const title = SEGMENTS.map((s) => `${s.label}: ${fmtMs(n[s.key])}`).join('\n')
        return (
          <div className="waterfall-row" key={i}>
            <div className="waterfall-label">
              {n.node}{n.task_order != null ? ` · t${n.task_order}` : ''}{n.attempt > 1 ? ` · #${n.attempt}` : ''}
            </div>
            <div className="waterfall-track">
              <div className={`waterfall-bar${n.ok ? '' : ' failed'}`} title={`${n.node}: ${fmtMs(n.duration_ms)}\n${title}`}
                   style={{ left: `${(n.start_ms / span) * 100}%`, width: `${Math.max((n.duration_ms / span) * 100, 0.6)}%` }}>
                {SEGMENTS.map((s) => (n[s.key] > 0 ? (
                  <span key={s.key} style={{ flex: n[s.key], background: s.color }} />
                ) : null))}
              </div>
            </div>
            <div className="waterfall-ms">{fmtMs(n.duration_ms)}</div>
          </div>
        )
      })}
    </div>
  )
}
