import { useCallback, useEffect, useMemo, useState } from 'react'
import { BarCard, LineCard, PieCard } from '../components/Chart.jsx'
import SortableTable from '../components/SortableTable.jsx'
import Waterfall, { SEGMENTS } from '../components/Waterfall.jsx'
import { AGENT_META } from '../constants/agents.js'
import { frameworkLabel } from '../constants/frameworks.js'
import { fmtCost, fmtMs, fmtNum, fmtPct, metricsApi } from '../lib/metricsApi.js'

/** Metrics dashboard: cost, latency and success across projects, with
 * drill-down into one run (per-agent tokens, cost split, latency
 * breakdown, per-node timeline), a two-project comparison, per-model
 * speed + regression check, and CSV/JSON export. All data comes from the
 * GET /metrics/* endpoints — this file only lays it out.
 */
const agentColor = (a) => AGENT_META[a]?.color || '#9ca3af'
const shortId = (id) => id.slice(0, 6)

function Kpi({ label, value, hint }) {
  return (
    <div className="kpi">
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  )
}

export default function MetricsDash({ focusProjectId }) {
  const [projects, setProjects] = useState([])
  const [summary, setSummary] = useState(null)
  const [models, setModels] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [compareIds, setCompareIds] = useState([])
  const [compare, setCompare] = useState(null)
  const [regression, setRegression] = useState(null)
  const [filter, setFilter] = useState({ status: '', mode: '' })
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [p, s, m] = await Promise.all([metricsApi.projects(), metricsApi.summary(), metricsApi.models()])
      setProjects(p)
      setSummary(s)
      setModels(m.models)
    } catch (e) {
      setError(`Could not load metrics: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])
  useEffect(() => { if (focusProjectId) setSelectedId(focusProjectId) }, [focusProjectId])

  useEffect(() => {
    if (!selectedId) { setDetail(null); return }
    let cancelled = false
    metricsApi.project(selectedId)
      .then((d) => { if (!cancelled) setDetail(d) })
      .catch(() => { if (!cancelled) setDetail(null) })
    return () => { cancelled = true }
  }, [selectedId])

  useEffect(() => {
    if (compareIds.length < 2) { setCompare(null); return }
    let cancelled = false
    metricsApi.compare(compareIds).then((d) => { if (!cancelled) setCompare(d) }).catch(() => {})
    return () => { cancelled = true }
  }, [compareIds])

  const visible = useMemo(() => projects.filter((p) =>
    (!filter.status || p.status === filter.status) && (!filter.mode || p.communication_mode === filter.mode)),
  [projects, filter])

  const totals = useMemo(() => {
    const t = visible.reduce((a, p) => ({
      tokens: a.tokens + p.tokens_total, cost: a.cost + (p.est_cost_usd || 0), ms: a.ms + (p.total_ms || 0),
      timed: a.timed + (p.total_ms ? 1 : 0),
    }), { tokens: 0, cost: 0, ms: 0, timed: 0 })
    return { ...t, avgMs: t.timed ? t.ms / t.timed : null }
  }, [visible])

  const trend = useMemo(() => [...visible].reverse().map((p) => ({
    name: shortId(p.project_id), tokens: p.tokens_total, seconds: p.total_ms ? +(p.total_ms / 1000).toFixed(1) : null,
  })), [visible])

  function toggleCompare(id) {
    setCompareIds((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id].slice(-2)))
  }

  async function checkRegression(model) {
    setRegression({ model, loading: true })
    try { setRegression(await metricsApi.regression(model)) }
    catch (e) { setRegression({ model, error: e.message }) }
  }

  const columns = [
    { key: 'name', label: 'Project', render: (p) => <span title={p.request}>{(p.name || p.project_id).slice(0, 34)}</span> },
    { key: 'status', label: 'Status' },
    { key: 'communication_mode', label: 'Mode' },
    { key: 'framework', label: 'Framework', render: (p) => (p.framework ? frameworkLabel(p.framework) : '—') },
    { key: 'task_types', label: 'Task types' },
    { key: 'model', label: 'Model' },
    { key: 'tokens_total', label: 'Tokens', align: 'right', render: (p) => fmtNum(p.tokens_total) },
    { key: 'est_cost_usd', label: 'Est. cost', align: 'right', render: (p) => fmtCost(p.est_cost_usd) },
    { key: 'total_ms', label: 'Latency', align: 'right', render: (p) => fmtMs(p.total_ms) },
    { key: 'pass_rate', label: 'Pass rate', align: 'right', render: (p) => fmtPct(p.pass_rate) },
    { key: 'avg_attempts', label: 'Avg attempts', align: 'right' },
  ]

  const d = detail
  const agentRows = d ? Object.entries(d.tokens.by_agent).map(([agent, v]) => ({ agent, ...v })) : []
  const latencyPie = d ? SEGMENTS.map((s) => ({ name: s.label, value: d.latency[s.key], color: s.color })) : []

  return (
    <div className="metrics-page">
      <div className="metrics-head">
        <div>
          <h2>Metrics</h2>
          <p className="subtitle">Token usage, estimated cost, latency and success across runs.</p>
        </div>
        <div className="metrics-actions">
          <button className="btn" onClick={refresh} disabled={loading}>{loading ? 'Loading…' : '↻ Refresh'}</button>
          <a className="btn" href={metricsApi.exportUrl('csv')}>⬇ CSV</a>
          <a className="btn" href={metricsApi.exportUrl('json')} target="_blank" rel="noreferrer">⬇ JSON</a>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="kpi-row">
        <Kpi label="Projects" value={fmtNum(visible.length)} />
        <Kpi label="Total tokens" value={fmtNum(totals.tokens)} />
        <Kpi label="Est. cost" value={fmtCost(totals.cost)} hint="cloud-equivalent rate" />
        <Kpi label="Avg run time" value={fmtMs(totals.avgMs)} />
        <Kpi label="Task pass rate" value={fmtPct(summary?.overall?.pass_rate)}
             hint={summary?.overall?.tasks ? `${summary.overall.tasks} tasks` : null} />
        <Kpi label="Avg attempts / task" value={summary?.overall?.avg_attempts ?? '—'} />
      </div>

      <div className="chart-grid">
        <BarCard title="Tokens per project" subtitle="oldest → newest" data={trend} xKey="name"
                 series={[{ key: 'tokens', label: 'tokens' }]} yFormatter={fmtNum} />
        <LineCard title="Run time trend" subtitle="seconds per run — rising = slower" data={trend} xKey="name"
                  series={[{ key: 'seconds', label: 'seconds', color: '#f59e0b' }]} />
      </div>

      <section className="panel">
        <div className="panel-head">
          <h3>Projects</h3>
          <div className="filters">
            <select value={filter.status} onChange={(e) => setFilter({ ...filter, status: e.target.value })}>
              <option value="">any status</option>
              {[...new Set(projects.map((p) => p.status))].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <select value={filter.mode} onChange={(e) => setFilter({ ...filter, mode: e.target.value })}>
              <option value="">any mode</option>
              {[...new Set(projects.map((p) => p.communication_mode).filter(Boolean))].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>
        <p className="hint">Click a row for its breakdown · tick two rows to compare them.</p>
        <SortableTable
          columns={columns} rows={visible} rowKey="project_id" selectedKey={selectedId}
          onRowClick={(p) => setSelectedId(p.project_id)} initialSort={null}
          renderPrefix={(p) => (
            <input type="checkbox" checked={compareIds.includes(p.project_id)} onChange={() => toggleCompare(p.project_id)} />
          )}
          emptyText="No runs with metrics yet — submit a request in the Chat tab."
        />
      </section>

      {d && (
        <section className="panel">
          <div className="panel-head">
            <h3>Run detail · <span className="mono">{shortId(d.project_id)}</span></h3>
            <span className="hint">{d.communication_mode} · {d.model}</span>
          </div>
          <div className="kpi-row">
            <Kpi label="Tokens in / out" value={`${fmtNum(d.tokens.tokens_in)} / ${fmtNum(d.tokens.tokens_out)}`} hint={`${d.tokens.llm_calls} LLM calls`} />
            <Kpi label="Est. cost" value={fmtCost(d.cost.est_cost_usd)} />
            <Kpi label="Total time" value={fmtMs(d.latency.total_ms)} />
            <Kpi label="Model load" value={fmtMs(d.latency.model_load_ms)} hint="cold start time" />
            <Kpi label="Transport overhead" value={fmtMs(d.latency.network_ms)} hint="wall − Ollama server time" />
            <Kpi label="Pass rate" value={fmtPct(d.success.pass_rate)} hint={`avg ${d.success.avg_attempts ?? '—'} attempts`} />
          </div>
          <div className="chart-grid three">
            <BarCard title="Tokens by agent" data={agentRows} xKey="agent" stacked
                     series={[{ key: 'tokens_in', label: 'in', color: '#6366f1' }, { key: 'tokens_out', label: 'out', color: '#34d399' }]}
                     yFormatter={fmtNum} />
            <PieCard title="Cost by agent" data={agentRows.map((a) => ({ name: a.agent, value: a.est_cost, color: agentColor(a.agent) }))}
                     valueFormatter={fmtCost} />
            <PieCard title="Where the time went" data={latencyPie} valueFormatter={fmtMs} />
          </div>
          <h4>Per-node timeline</h4>
          <Waterfall nodes={d.nodes} totalMs={d.latency.total_ms} />

          <div className="two-col">
            <div>
              <h4>Tools</h4>
              <SortableTable rowKey="name" rows={Object.entries(d.tools).map(([name, v]) => ({ name, ...v }))}
                columns={[{ key: 'name', label: 'Tool' }, { key: 'calls', label: 'Calls', align: 'right' },
                          { key: 'avg_ms', label: 'Avg', align: 'right', render: (r) => fmtMs(r.avg_ms) },
                          { key: 'total_ms', label: 'Total', align: 'right', render: (r) => fmtMs(r.total_ms) },
                          { key: 'errors', label: 'Errors', align: 'right' }]} emptyText="No tool calls" />
            </div>
            <div>
              <h4>Tasks</h4>
              <SortableTable rowKey="order" rows={d.tasks}
                columns={[{ key: 'order', label: '#' }, { key: 'type', label: 'Type' },
                          { key: 'passed', label: 'Result', render: (t) => (t.passed ? '✅ passed' : t.escalated ? '🚩 escalated' : '❌ failed') },
                          { key: 'attempts', label: 'Attempts', align: 'right' },
                          { key: 'repeated_errors', label: 'Repeat errs', align: 'right' }]} emptyText="No finished tasks" />
            </div>
          </div>
        </section>
      )}

      {compare && compare.deltas?.length > 0 && (
        <section className="panel">
          <div className="panel-head"><h3>Comparison</h3>
            <span className="hint">{compare.projects.map((p) => shortId(p.project_id)).join(' vs ')} · % change vs first</span></div>
          <div className="chart-grid">
            <BarCard title="Tokens by agent" xKey="agent"
              data={[...new Set(compare.projects.flatMap((p) => Object.keys(p.tokens.by_agent)))].map((agent) => ({
                agent, ...Object.fromEntries(compare.projects.map((p) => [shortId(p.project_id),
                  (p.tokens.by_agent[agent]?.tokens_in || 0) + (p.tokens.by_agent[agent]?.tokens_out || 0)])),
              }))}
              series={compare.projects.map((p, i) => ({ key: shortId(p.project_id), label: shortId(p.project_id), color: i ? '#f59e0b' : '#6366f1' }))}
              yFormatter={fmtNum} />
            <div className="chart-card">
              <div className="chart-title">Change vs baseline</div>
              <table className="metrics-table">
                <thead><tr><th>Metric</th><th className="num">Baseline</th><th className="num">Other</th><th className="num">Change</th></tr></thead>
                <tbody>
                  {Object.entries(compare.deltas[0]).filter(([k]) => k !== 'project_id').map(([k, v]) => (
                    <tr key={k}>
                      <td>{k.replace(/_/g, ' ')}</td>
                      <td className="num">{v.baseline ?? '—'}</td><td className="num">{v.value ?? '—'}</td>
                      <td className={`num ${v.pct_change > 0 ? 'up' : v.pct_change < 0 ? 'down' : ''}`}>
                        {v.pct_change == null ? '—' : `${v.pct_change > 0 ? '+' : ''}${v.pct_change}%`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {summary?.overall?.tasks > 0 && (
        <section className="panel">
          <div className="panel-head"><h3>Success</h3></div>
          <div className="chart-grid three">
            <BarCard title="Pass rate by task type" xKey="type" yFormatter={(v) => `${Math.round(v * 100)}%`}
              data={Object.entries(summary.by_task_type).map(([type, s]) => ({ type, rate: s.pass_rate }))}
              series={[{ key: 'rate', label: 'pass rate', color: '#34d399' }]} />
            <BarCard title="Pass rate by framework" subtitle="plain code tasks = none" xKey="framework" yFormatter={(v) => `${Math.round(v * 100)}%`}
              data={Object.entries(summary.by_framework || {}).map(([framework, s]) => ({ framework, rate: s.pass_rate }))}
              series={[{ key: 'rate', label: 'pass rate', color: '#38bdf8' }]} />
            <BarCard title="Tasks by attempts used" subtitle="1 = passed first time" xKey="attempts"
              data={Object.entries(summary.attempt_histogram).map(([attempts, tasks]) => ({ attempts, tasks }))}
              series={[{ key: 'tasks', label: 'tasks', color: '#6366f1' }]} />
            <div className="chart-card">
              <div className="chart-title">Rates</div>
              <table className="metrics-table"><tbody>
                <tr><td>Escalation rate</td><td className="num">{fmtPct(summary.overall.escalation_rate)}</td></tr>
                <tr><td>Timeout rate</td><td className="num">{fmtPct(summary.overall.timeout_rate)}</td></tr>
                <tr><td>Repeat-error rate</td><td className="num">{fmtPct(summary.overall.repeat_error_rate)}</td></tr>
                <tr><td>Avg attempts (with similar past fix)</td><td className="num">{summary.failure_recovery.avg_attempts_with_similar_fix ?? '—'}</td></tr>
                <tr><td>Avg attempts (without)</td><td className="num">{summary.failure_recovery.avg_attempts_without_similar_fix ?? '—'}</td></tr>
              </tbody></table>
            </div>
          </div>
        </section>
      )}

      {models && Object.keys(models).length > 0 && (
        <section className="panel">
          <div className="panel-head"><h3>Models</h3><span className="hint">speed, quality and regression check</span></div>
          <SortableTable rowKey="model"
            rows={Object.entries(models).map(([model, m]) => ({ model, ...m }))}
            columns={[
              { key: 'model', label: 'Model' },
              { key: 'llm_calls', label: 'Calls', align: 'right' },
              { key: 'ms_per_output_token', label: 'ms / output token', align: 'right' },
              { key: 'tokens_per_s_generation', label: 'tok/s (gen)', align: 'right' },
              { key: 'cold_loads', label: 'Cold loads', align: 'right' },
              { key: 'task_pass_rate', label: 'Pass rate', align: 'right', render: (m) => fmtPct(m.task_pass_rate) },
              { key: 'task_avg_attempts', label: 'Avg attempts', align: 'right' },
              { key: 'slower_than_fastest', label: 'Speed', render: (m) => (m.slower_than_fastest ? `🐢 ${m.slowdown_vs_fastest}× slower` : m.ms_per_output_token ? '⚡ fastest tier' : '—') },
              { key: 'check', label: '', sortValue: () => 0, render: (m) => <button className="btn small" onClick={() => checkRegression(m.model)}>Check regression</button> },
            ]} />
          {regression && (
            <div className={`regression ${regression.regression ? 'bad' : ''}`}>
              {regression.loading && 'Checking…'}
              {regression.error && `Could not check: ${regression.error}`}
              {!regression.loading && !regression.error && (regression.note
                ? `${regression.model}: ${regression.note}`
                : `${regression.model}: ${regression.recent_ms_per_token} ms/token recently vs ${regression.baseline_ms_per_token} before (${regression.pct_change > 0 ? '+' : ''}${regression.pct_change}%) — ${regression.regression ? '⚠ REGRESSION (slower than threshold)' : 'no regression'}`)}
            </div>
          )}
        </section>
      )}
    </div>
  )
}
