import { useCallback, useEffect, useState } from 'react'
import { BarCard } from '../components/Chart.jsx'
import ResultsTable from '../components/ResultsTable.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { batchApi, isActive } from '../lib/batchApi.js'
import { fmtCost, fmtMs, fmtNum, fmtPct } from '../lib/metricsApi.js'
import { frameworkLabel } from '../constants/frameworks.js'

/** One batch: live progress while it runs, then the comparison table of every
 * project (sort, filter, drill down into its metrics), a tokens-per-project
 * chart and CSV export. */
function Kpi({ label, value }) {
  return <div className="kpi"><div className="kpi-value">{value}</div><div className="kpi-label">{label}</div></div>
}

export default function BatchResults({ batchId, onBack, onOpenProject }) {
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try { setRes(await batchApi.batchResults(batchId)); setError(null) }
    catch (e) { setError(e.message) }
  }, [batchId])

  useEffect(() => { load() }, [load])
  const running = res ? isActive(res.batch.status) : true
  useEffect(() => {
    if (!running) return undefined
    const id = setInterval(load, 2000)
    return () => clearInterval(id)
  }, [running, load])

  async function cancel() {
    setBusy(true)
    try { await batchApi.cancelBatch(batchId); await load() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  if (!res) return <div className="panel">{error ? <div className="error-banner">{error}</div> : 'Loading…'}</div>
  const { batch, rows, summary } = res

  const columns = [
    { key: 'idx', label: '#', render: (r) => r.idx + 1 },
    { key: 'request', label: 'Request', render: (r) => <span title={r.request}>{r.request.slice(0, 48)}</span> },
    { key: 'status', label: 'Status', render: (r) => <StatusBadge status={r.status} /> },
    { key: 'framework', label: 'Framework', render: (r) => (r.framework ? frameworkLabel(r.framework) : '—') },
    { key: 'communication_mode', label: 'Mode' },
    { key: 'tokens_total', label: 'Tokens', align: 'right', render: (r) => fmtNum(r.tokens_total) },
    { key: 'total_ms', label: 'Latency', align: 'right', render: (r) => fmtMs(r.total_ms) },
    { key: 'est_cost_usd', label: 'Cost', align: 'right', render: (r) => fmtCost(r.est_cost_usd) },
    { key: 'pass_rate', label: 'Task pass', align: 'right', render: (r) => fmtPct(r.pass_rate) },
    { key: 'avg_attempts', label: 'Attempts', align: 'right' },
    { key: 'open', label: '', sortValue: () => 0,
      render: (r) => (r.project_id ? <button className="btn small" onClick={(e) => { e.stopPropagation(); onOpenProject?.(r.project_id) }}>Metrics →</button> : '') },
  ]
  const chart = rows.map((r) => ({ name: `#${r.idx + 1}`, tokens: r.tokens_total || 0 }))

  return (
    <div className="batch-results">
      <div className="panel-head">
        <div>
          <button className="btn small" onClick={onBack}>← Back</button>{' '}
          <strong>{batch.name || batch.batch_id.slice(0, 8)}</strong> <StatusBadge status={batch.status} />
          {batch.level && <span className="hint"> · condition: {batch.level}</span>}
        </div>
        <div className="metrics-actions">
          {running && <button className="btn" disabled={busy} onClick={cancel}>✕ Cancel</button>}
          <a className="btn" href={batchApi.batchCsvUrl(batchId)}>⬇ CSV</a>
        </div>
      </div>
      {error && <div className="error-banner">{error}</div>}

      <div className="progress" role="progressbar" aria-valuenow={Math.round(batch.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-bar" style={{ width: `${batch.progress * 100}%` }} />
        <span className="progress-label">{batch.finished}/{batch.total} finished · {batch.running} running · {batch.pending} waiting</span>
      </div>

      <div className="kpi-row">
        <Kpi label="Passed" value={`${summary.passed}/${summary.runs}`} />
        <Kpi label="Needs review" value={summary.needs_review} />
        <Kpi label="Errors" value={summary.error} />
        <Kpi label="Total tokens" value={fmtNum(summary.tokens_total)} />
        <Kpi label="Est. cost" value={fmtCost(summary.est_cost_usd)} />
        <Kpi label="Mean latency" value={fmtMs(summary.total_ms?.mean)} />
      </div>

      <BarCard title="Tokens per project" data={chart} xKey="name" series={[{ key: 'tokens', label: 'tokens' }]} yFormatter={fmtNum} height={200} />

      <ResultsTable rows={rows} columns={columns} rowKey="idx" costKey="est_cost_usd"
                    filters={[{ key: 'status', label: 'status' }, { key: 'framework', label: 'framework' }]}
                    onRowClick={(r) => r.project_id && onOpenProject?.(r.project_id)}
                    emptyText="No projects yet" />
    </div>
  )
}
