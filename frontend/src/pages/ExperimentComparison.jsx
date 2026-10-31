import { useCallback, useEffect, useState } from 'react'
import { BarCard, LineCard } from '../components/Chart.jsx'
import SortableTable from '../components/SortableTable.jsx'
import StatusBadge from '../components/StatusBadge.jsx'
import { batchApi, fmtNumber, fmtP, isActive } from '../lib/batchApi.js'

/** One experiment: the verdict in plain words, then per-metric bars with the
 * significance of each comparison, the full statistics table, and exports
 * (raw CSV for re-analysis, LaTeX tables, JSON). `ab` is the main design;
 * repeat / sequence / mcp_overhead have their own, simpler, views. */
const METRIC_FORMAT = { total_ms: (v) => `${fmtNumber(v / 1000, 1)} s`, est_cost_usd: (v) => `$${fmtNumber(v, 5)}` }
const show = (key, v) => (v == null ? '—' : (METRIC_FORMAT[key] ? METRIC_FORMAT[key](v) : fmtNumber(v, 2)))

function SigBadge({ c }) {
  if (c.p_adjusted == null) return <span className="sig na" title="could not be tested">n/a</span>
  return (
    <span className={`sig ${c.significant ? 'yes' : 'no'}`} title={`Holm-adjusted p = ${fmtP(c.p_adjusted)}`}>
      {c.significant ? '✓ significant' : 'not significant'}{c.low_power ? ' · low n' : ''}
    </span>
  )
}

function AbView({ r }) {
  const keys = Object.keys(r.metric_labels)
  const armRows = r.arms.map((a) => ({ label: a.label, n: a.n, errors: a.errors, pass: a.n ? a.passed / a.n : null,
    ...Object.fromEntries(keys.map((k) => [k, a.metrics[k]?.mean])), ...Object.fromEntries(keys.map((k) => [`${k}_sd`, a.metrics[k]?.sd])) }))
  const comparisonRows = []
  keys.forEach((k) => {
    const comp = r.comparisons[k]
    if (!comp?.comparisons) return
    comp.comparisons.forEach((c) => comparisonRows.push({ id: `${k}-${c.level}`, metric: r.metric_labels[k], key: k, level: c.level, c }))
  })
  return (
    <>
      <div className="chart-grid three">
        {keys.filter((k) => r.comparisons[k]?.groups).map((k) => {
          const comp = r.comparisons[k]
          const data = r.arms.map((a) => ({ arm: a.label, value: comp.groups[a.label]?.mean ?? 0 }))
          return (
            <div key={k} className="metric-card">
              <BarCard title={r.metric_labels[k]} subtitle={`baseline: ${r.baseline}`} data={data} xKey="arm"
                       series={[{ key: 'value', label: r.metric_labels[k] }]} height={170} yFormatter={(v) => show(k, v)} />
              {comp.comparisons.map((c) => (
                <div key={c.level} className="comparison-line">
                  <b>{c.level}</b>{' '}
                  {c.pct_change != null ? `${c.pct_change > 0 ? '+' : ''}${fmtNumber(c.pct_change, 1)}%` : c.effect.diff != null ? `${fmtNumber(c.effect.diff * 100, 1)} pp` : ''}{' '}
                  <SigBadge c={c} />
                </div>
              ))}
            </div>
          )
        })}
        {keys.filter((k) => r.comparisons[k]?.skipped).map((k) => (
          <div key={k} className="metric-card"><div className="chart-title">{r.metric_labels[k]}</div><div className="hint">{r.comparisons[k].skipped}</div></div>
        ))}
      </div>

      <h4>Per condition (mean)</h4>
      <SortableTable rowKey="label" rows={armRows} columns={[
        { key: 'label', label: 'Condition' }, { key: 'n', label: 'Runs', align: 'right' }, { key: 'errors', label: 'Errors', align: 'right' },
        { key: 'pass', label: 'Pass rate', align: 'right', render: (a) => (a.pass == null ? '—' : `${Math.round(a.pass * 100)}%`) },
        ...keys.filter((k) => k !== 'pass').map((k) => ({ key: k, label: r.metric_labels[k], align: 'right',
          render: (a) => (a[k] == null ? '—' : `${show(k, a[k])} ± ${show(k, a[`${k}_sd`] ?? 0)}`) })),
      ]} />

      <h4>Statistical comparisons vs <code>{r.baseline}</code></h4>
      <SortableTable rowKey="id" rows={comparisonRows} columns={[
        { key: 'metric', label: 'Metric' }, { key: 'level', label: 'Condition' },
        { key: 'delta', label: 'Δ', align: 'right', sortValue: (x) => x.c.pct_change ?? x.c.effect.diff,
          render: (x) => (x.c.pct_change != null ? `${x.c.pct_change > 0 ? '+' : ''}${fmtNumber(x.c.pct_change, 1)}%` : x.c.effect.diff != null ? `${fmtNumber(x.c.effect.diff * 100, 1)} pp` : '—') },
        { key: 'ci', label: '95% CI of Δ', align: 'right', render: (x) => (x.c.effect.ci95 ? `[${show(x.key, x.c.effect.ci95[0])}, ${show(x.key, x.c.effect.ci95[1])}]` : '—') },
        { key: 'p', label: 'p', align: 'right', sortValue: (x) => x.c.p, render: (x) => fmtP(x.c.p) },
        { key: 'padj', label: 'p (Holm)', align: 'right', sortValue: (x) => x.c.p_adjusted, render: (x) => fmtP(x.c.p_adjusted) },
        { key: 'd', label: "Cohen's d", align: 'right', render: (x) => (x.c.effect.cohens_d != null ? `${fmtNumber(x.c.effect.cohens_d, 2)} (${x.c.effect.label})` : '—') },
        { key: 'sig', label: 'Result', sortValue: (x) => (x.c.significant ? 1 : 0), render: (x) => <SigBadge c={x.c} /> },
      ]} />
    </>
  )
}

function RepeatView({ r }) {
  const rows = Object.entries(r.comparisons).map(([k, t]) => ({ k, label: r.metric_labels[k], ...t }))
  return (
    <>
      <h4>First (cold) vs repeat (warm) run — paired t-test</h4>
      <SortableTable rowKey="k" rows={rows} columns={[
        { key: 'label', label: 'Metric' }, { key: 'mean_first', label: 'First', align: 'right', render: (t) => show(t.k, t.mean_first) },
        { key: 'mean_repeat', label: 'Repeat', align: 'right', render: (t) => show(t.k, t.mean_repeat) },
        { key: 'pct_change', label: 'Δ', align: 'right', render: (t) => (t.pct_change == null ? '—' : `${t.pct_change > 0 ? '+' : ''}${fmtNumber(t.pct_change, 1)}%`) },
        { key: 'n', label: 'Pairs', align: 'right' }, { key: 'p', label: 'p', align: 'right', render: (t) => fmtP(t.p) },
      ]} />
      <p className="hint">{r.memory_hit_in_repeat} of {r.pairs.length} repeat runs used a memory hit. {r.excluded.length} pair(s) excluded.</p>
      {r.excluded.map((e, i) => <div key={i} className="hint">excluded: “{e.request.slice(0, 60)}” — {e.reason}</div>)}
    </>
  )
}

function SequenceView({ r }) {
  const data = r.series.map((s) => ({ position: s.position, attempts: s.avg_attempts, tokens: s.tokens_total }))
  const rows = Object.entries(r.trends).map(([k, t]) => ({ k, label: r.metric_labels[k], ...t }))
  return (
    <>
      <div className="chart-grid">
        <LineCard title="Attempts per task by position" data={data} xKey="position" series={[{ key: 'attempts', label: 'attempts' }]} />
        <LineCard title="Tokens by position" data={data} xKey="position" series={[{ key: 'tokens', label: 'tokens', color: '#34d399' }]} />
      </div>
      <SortableTable rowKey="k" rows={rows} columns={[
        { key: 'label', label: 'Metric' }, { key: 'slope', label: 'Slope / position', align: 'right', render: (t) => fmtNumber(t.slope, 3) },
        { key: 'r', label: 'r', align: 'right', render: (t) => fmtNumber(t.r, 2) }, { key: 'p', label: 'p', align: 'right', render: (t) => fmtP(t.p) },
      ]} />
      <p className="hint">Memory hit in {Math.round((r.memory_hit_rate || 0) * 100)}% of the runs.</p>
    </>
  )
}

function McpView({ r }) {
  return (
    <>
      <BarCard title="Mean latency per write (ms)" data={[{ path: 'MCP', ms: r.mcp_ms.mean }, { path: 'direct', ms: r.direct_ms.mean }]} xKey="path"
               series={[{ key: 'ms', label: 'ms' }]} height={180} />
      <SortableTable rowKey="path" rows={[{ path: 'MCP write_file', ...r.mcp_ms }, { path: 'Direct write', ...r.direct_ms }]} columns={[
        { key: 'path', label: 'Path' }, { key: 'n', label: 'Calls', align: 'right' }, { key: 'mean', label: 'Mean (ms)', align: 'right', render: (x) => fmtNumber(x.mean, 3) },
        { key: 'sd', label: 'SD', align: 'right', render: (x) => fmtNumber(x.sd, 3) }, { key: 'median', label: 'Median', align: 'right', render: (x) => fmtNumber(x.median, 3) },
      ]} />
      <p className="hint">Overhead {fmtNumber(r.overhead_ms, 2)} ms/call ({fmtNumber(r.ratio, 0)}×), Welch p = {fmtP(r.tests.welch.p)}, Mann-Whitney p = {fmtP(r.tests.mann_whitney.p)}.</p>
    </>
  )
}

export default function ExperimentComparison({ experimentId, onBack, onOpenBatch }) {
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)
  const load = useCallback(async () => {
    try { setRes(await batchApi.experimentResults(experimentId)); setError(null) } catch (e) { setError(e.message) }
  }, [experimentId])
  useEffect(() => { load() }, [load])
  const running = res ? isActive(res.experiment.status) : true
  useEffect(() => {
    if (!running) return undefined
    const id = setInterval(load, 2500)
    return () => clearInterval(id)
  }, [running, load])

  if (!res) return <div className="panel">{error ? <div className="error-banner">{error}</div> : 'Loading…'}</div>
  const { experiment: e, result: r } = res
  const cfg = e.config || {}

  return (
    <div className="experiment-comparison">
      <div className="panel-head">
        <div>
          <button className="btn small" onClick={onBack}>← Back</button>{' '}
          <strong>{e.name}</strong> <StatusBadge status={e.status} /> <span className="hint">{e.design}{cfg.factor ? ` · ${cfg.factor}` : ''}</span>
        </div>
        <div className="metrics-actions">
          {running && <button className="btn" onClick={async () => { await batchApi.cancelExperiment(experimentId); load() }}>✕ Cancel</button>}
          <a className="btn" href={batchApi.experimentUrl(experimentId, 'csv')}>⬇ Raw CSV</a>
          <a className="btn" href={batchApi.experimentUrl(experimentId, 'latex')}>⬇ LaTeX</a>
          <a className="btn" href={batchApi.experimentUrl(experimentId, 'json')} target="_blank" rel="noreferrer">⬇ JSON</a>
        </div>
      </div>
      {(error || e.error) && <div className="error-banner">{error || e.error}</div>}

      <div className="progress" role="progressbar" aria-valuenow={Math.round(e.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-bar" style={{ width: `${e.progress * 100}%` }} />
        <span className="progress-label">{e.finished_runs}/{e.total_runs || '?'} runs</span>
      </div>
      <div className="arm-links">
        {e.arms.map((a) => (
          <button key={a.batch_id} className="btn small" onClick={() => onOpenBatch?.(a.batch_id)}>
            {a.level} <StatusBadge status={a.status} /> {a.finished}/{a.total}
          </button>
        ))}
      </div>

      {!r && <p className="hint">Results appear when every condition has finished.</p>}
      {r && (
        <>
          {r.verdict?.length > 0 && (
            <div className="verdict"><h4>Verdict</h4><ul>{r.verdict.map((v, i) => <li key={i}>{v}</li>)}</ul></div>
          )}
          {r.warnings?.length > 0 && (
            <div className="warnings"><h4>Read before citing</h4><ul>{r.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></div>
          )}
          {r.design === 'ab' && <AbView r={r} />}
          {r.design === 'repeat' && <RepeatView r={r} />}
          {r.design === 'sequence' && <SequenceView r={r} />}
          {r.design === 'mcp_overhead' && <McpView r={r} />}
        </>
      )}

      <details className="repro">
        <summary>Reproducibility: configuration and environment</summary>
        <pre>{JSON.stringify(cfg, null, 2)}</pre>
      </details>
    </div>
  )
}
