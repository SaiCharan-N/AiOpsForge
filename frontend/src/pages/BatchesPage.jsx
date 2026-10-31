import { useCallback, useEffect, useState } from 'react'
import StatusBadge from '../components/StatusBadge.jsx'
import { FRAMEWORKS } from '../constants/frameworks.js'
import { batchApi, isActive } from '../lib/batchApi.js'
import BatchResults from './BatchResults.jsx'
import ExperimentComparison from './ExperimentComparison.jsx'

/** The "Batches" tab: start many runs at once, or a controlled experiment, and
 * open the results of either. Lists refresh every few seconds while anything runs. */
const lines = (text) => text.split('\n').map((l) => l.trim()).filter(Boolean)

function usePoll(fn, active, ms = 3000) {
  useEffect(() => {
    fn()
    if (!active) return undefined
    const id = setInterval(fn, ms)
    return () => clearInterval(id)
  }, [fn, active, ms])
}

function Progress({ value }) {
  return <div className="progress mini"><div className="progress-bar" style={{ width: `${(value || 0) * 100}%` }} /></div>
}

function NewBatch({ onCreated }) {
  const [text, setText] = useState('')
  const [framework, setFramework] = useState('')
  const [mode, setMode] = useState('')
  const [attempts, setAttempts] = useState('')
  const [memory, setMemory] = useState('')
  const [conc, setConc] = useState('')
  const [name, setName] = useState('')
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const requests = lines(text)

  async function submit(e) {
    e.preventDefault()
    setBusy(true); setErr(null)
    const defaults = {}
    if (framework) defaults.framework = framework
    if (mode) defaults.communication_mode = mode
    if (attempts) defaults.max_qa_attempts = Number(attempts)
    if (memory) defaults.use_memory = memory === 'on'
    try {
      const r = await batchApi.createBatch({ name: name || undefined, requests, defaults, ...(conc ? { max_concurrency: Number(conc) } : {}) })
      setText('')
      onCreated(r.batch_id)
    } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  return (
    <form className="panel batch-form" onSubmit={submit}>
      <h3>New batch</h3>
      <p className="hint">One request per line. Every line becomes its own project; they run at the same time.</p>
      <textarea rows={5} value={text} onChange={(e) => setText(e.target.value)} placeholder={'a function that checks if a number is prime, with a test\nBuild a Flask app with a home page and an about page\n…'} />
      <div className="form-row">
        <input placeholder="Name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
        <select aria-label="framework" value={framework} onChange={(e) => setFramework(e.target.value)}>
          {FRAMEWORKS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
        </select>
        <select aria-label="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="">default mode</option><option value="blackboard">blackboard</option><option value="message_passing">message_passing</option>
        </select>
        <select aria-label="memory" value={memory} onChange={(e) => setMemory(e.target.value)}>
          <option value="">memory: default (on)</option><option value="on">memory on</option><option value="off">memory off</option>
        </select>
        <input type="number" min="1" max="10" placeholder="max attempts" aria-label="max attempts" value={attempts} onChange={(e) => setAttempts(e.target.value)} />
        <input type="number" min="1" max="20" placeholder="concurrency" aria-label="concurrency" value={conc} onChange={(e) => setConc(e.target.value)} />
      </div>
      {err && <div className="error-banner">{err}</div>}
      <button className="btn primary" type="submit" disabled={busy || requests.length === 0}>▶ Run {requests.length || ''} request{requests.length === 1 ? '' : 's'}</button>
    </form>
  )
}

function parseLevels(factor, text) {
  const parts = text.split(',').map((s) => s.trim()).filter(Boolean)
  if (factor === 'use_memory') return parts.map((p) => ['on', 'true', '1', 'yes'].includes(p.toLowerCase()))
  if (factor === 'max_qa_attempts') return parts.map(Number)
  return parts
}
const showLevels = (factor, levels) => (levels || []).map((l) => (factor === 'use_memory' ? (l ? 'on' : 'off') : l)).join(', ')

function NewExperiment({ presets, onCreated }) {
  const [preset, setPreset] = useState('communication_mode')
  const [form, setForm] = useState({ name: '', design: 'ab', factor: 'communication_mode', levels: 'blackboard, message_passing', requests: '', repetitions: 3, calls: 20 })
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const set = (patch) => setForm((f) => ({ ...f, ...patch }))

  useEffect(() => {
    const p = presets?.presets.find((x) => x.name === preset)
    if (!p) return
    set({ design: p.design, factor: p.factor || '', levels: showLevels(p.factor, p.levels), requests: (p.requests || []).join('\n'),
          repetitions: p.repetitions || 1, calls: p.calls || 20, name: '' })
  }, [preset, presets])  // eslint-disable-line react-hooks/exhaustive-deps

  const description = presets?.presets.find((x) => x.name === preset)?.description
  const isAb = form.design === 'ab'

  async function submit(e) {
    e.preventDefault()
    setBusy(true); setErr(null)
    const body = { preset: preset || undefined, name: form.name || undefined, design: form.design }
    if (form.design === 'mcp_overhead') body.calls = Number(form.calls)
    else {
      body.requests = lines(form.requests)
      if (isAb) { body.factor = form.factor; body.levels = parseLevels(form.factor, form.levels); body.repetitions = Number(form.repetitions) }
    }
    try { onCreated((await batchApi.createExperiment(body)).experiment_id) } catch (ex) { setErr(ex.message) } finally { setBusy(false) }
  }

  return (
    <form className="panel batch-form" onSubmit={submit}>
      <h3>New experiment</h3>
      <div className="form-row">
        <select aria-label="preset" value={preset} onChange={(e) => setPreset(e.target.value)}>
          {(presets?.presets || []).map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
        </select>
        <input placeholder="Name (optional)" value={form.name} onChange={(e) => set({ name: e.target.value })} />
      </div>
      {description && <p className="hint">{description}</p>}
      {isAb && (
        <div className="form-row">
          <select aria-label="factor" value={form.factor} onChange={(e) => set({ factor: e.target.value })}>
            {(presets?.factors || []).map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <input aria-label="levels" value={form.levels} onChange={(e) => set({ levels: e.target.value })} placeholder="levels, comma separated — the first is the baseline" />
          <input type="number" min="1" max="10" aria-label="repetitions" value={form.repetitions} onChange={(e) => set({ repetitions: e.target.value })} title="repetitions per request per level" />
        </div>
      )}
      {form.design === 'mcp_overhead'
        ? <div className="form-row"><input type="number" min="2" max="200" aria-label="calls" value={form.calls} onChange={(e) => set({ calls: e.target.value })} /> tool calls</div>
        : <textarea rows={5} aria-label="requests" value={form.requests} onChange={(e) => set({ requests: e.target.value })} placeholder="one request per line" />}
      <p className="hint">Memory is switched off in every condition unless memory is the factor, so one condition cannot learn from another.</p>
      {err && <div className="error-banner">{err}</div>}
      <button className="btn primary" type="submit" disabled={busy}>▶ Run experiment</button>
    </form>
  )
}

export default function BatchesPage({ onOpenProject }) {
  const [tab, setTab] = useState('batches')
  const [batches, setBatches] = useState([])
  const [experiments, setExperiments] = useState([])
  const [presets, setPresets] = useState(null)
  const [selected, setSelected] = useState(null)   // { kind: 'batch' | 'experiment', id }
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const [b, e] = await Promise.all([batchApi.batches(), batchApi.experiments()])
      setBatches(b); setExperiments(e); setError(null)
    } catch (ex) { setError(`Could not load: ${ex.message}`) }
  }, [])
  usePoll(refresh, batches.some((b) => isActive(b.status)) || experiments.some((e) => isActive(e.status)))
  useEffect(() => { batchApi.presets().then(setPresets).catch(() => {}) }, [])

  if (selected?.kind === 'batch') {
    return <div className="batches-page"><BatchResults batchId={selected.id} onBack={() => { setSelected(null); refresh() }} onOpenProject={onOpenProject} /></div>
  }
  if (selected?.kind === 'experiment') {
    return (
      <div className="batches-page">
        <ExperimentComparison experimentId={selected.id} onBack={() => { setSelected(null); refresh() }}
                              onOpenBatch={(id) => setSelected({ kind: 'batch', id })} />
      </div>
    )
  }

  return (
    <div className="batches-page">
      <div className="metrics-head">
        <div><h2>Batches &amp; experiments</h2>
          <p className="subtitle">Run many requests at once, or compare conditions with statistics.</p></div>
        <nav className="tabs sub">
          <button className={tab === 'batches' ? 'active' : ''} onClick={() => setTab('batches')}>Batches</button>
          <button className={tab === 'experiments' ? 'active' : ''} onClick={() => setTab('experiments')}>Experiments</button>
        </nav>
      </div>
      {error && <div className="error-banner">{error}</div>}

      {tab === 'batches' ? (
        <>
          <NewBatch onCreated={(id) => setSelected({ kind: 'batch', id })} />
          <section className="panel"><h3>Recent batches</h3>
            <table className="metrics-table"><thead><tr><th>Name</th><th>Status</th><th>Progress</th><th className="num">Runs</th><th className="num">Passed</th><th className="num">Errors</th><th>Created</th></tr></thead>
              <tbody>
                {batches.length === 0 && <tr><td colSpan={7} className="empty-cell">No batches yet</td></tr>}
                {batches.map((b) => (
                  <tr key={b.batch_id} onClick={() => setSelected({ kind: 'batch', id: b.batch_id })}>
                    <td>{b.name || b.batch_id.slice(0, 8)}{b.experiment_id ? ' ·exp' : ''}</td><td><StatusBadge status={b.status} /></td>
                    <td><Progress value={b.progress} /></td><td className="num">{b.total}</td><td className="num">{b.passed}</td><td className="num">{b.errors}</td>
                    <td>{String(b.created_at).slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody></table>
          </section>
        </>
      ) : (
        <>
          <NewExperiment presets={presets} onCreated={(id) => setSelected({ kind: 'experiment', id })} />
          <section className="panel"><h3>Experiments</h3>
            <table className="metrics-table"><thead><tr><th>Name</th><th>Design</th><th>Status</th><th>Created</th></tr></thead>
              <tbody>
                {experiments.length === 0 && <tr><td colSpan={4} className="empty-cell">No experiments yet</td></tr>}
                {experiments.map((x) => (
                  <tr key={x.experiment_id} onClick={() => setSelected({ kind: 'experiment', id: x.experiment_id })}>
                    <td>{x.name}</td><td>{x.design}</td><td><StatusBadge status={x.status} /></td><td>{String(x.created_at).slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody></table>
          </section>
        </>
      )}
    </div>
  )
}
