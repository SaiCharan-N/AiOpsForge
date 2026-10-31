import { useEffect, useMemo, useState } from 'react'
import CodeBlock from './CodeBlock.jsx'

/** Shown above the composer while the pipeline is parked waiting for the
 * user (event `pipeline_paused`). What it renders depends on the pause point:
 *   planner     editable task list  ("Does this plan make sense?")
 *   developer   the first generated file ("Is this the right approach?")
 *   qa_failure  the analyzer's diagnosis ("retry, add guidance, or skip?")
 *   manual      a summary of where the run is
 * and answers with { action, guidance, clarification, tasks } via onSubmit.
 */
function Countdown({ since, timeoutS }) {
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    if (!timeoutS) return undefined
    const id = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(id)
  }, [timeoutS])
  if (!timeoutS) return null
  const left = Math.max(0, Math.round(since + timeoutS - now))
  return <span className="pause-timer">auto-continues in {Math.floor(left / 60)}:{String(left % 60).padStart(2, '0')}</span>
}

export default function PausePanel({ pause, onSubmit, busy }) {
  const payload = pause.payload || {}
  const original = useMemo(() => (payload.tasks || []).map((t) => t.description), [payload.tasks])
  const [tasks, setTasks] = useState(original)
  const [guidance, setGuidance] = useState('')
  const [clarification, setClarification] = useState('')
  useEffect(() => { setTasks(original); setGuidance(''); setClarification('') }, [pause.since, original])

  const isPlan = pause.node === 'planner'
  const planEdited = isPlan && JSON.stringify(tasks.map((t) => t.trim()).filter(Boolean)) !== JSON.stringify(original)
  const hasText = guidance.trim() || clarification.trim()
  const canRevise = Boolean(hasText || planEdited)
  const has = (a) => pause.options.includes(a)

  function send(action) {
    const body = { action }
    if (action === 'revise') {
      if (guidance.trim()) body.guidance = guidance.trim()
      if (clarification.trim()) body.clarification = clarification.trim()
      if (planEdited) body.tasks = tasks.map((t) => t.trim()).filter(Boolean)
    }
    onSubmit(body)
  }

  const a = payload.analysis
  return (
    <section className="pause-panel" role="region" aria-label="Pipeline paused">
      <div className="pause-head">
        <span className="pause-badge">⏸ Paused</span>
        <strong>{pause.question}</strong>
        <Countdown since={pause.since} timeoutS={pause.timeout_s} />
      </div>

      {isPlan && (
        <ol className="pause-tasks">
          {tasks.map((t, i) => (
            <li key={i}>
              <textarea rows={2} value={t} disabled={busy}
                        onChange={(e) => setTasks(tasks.map((x, j) => (j === i ? e.target.value : x)))} />
              <button type="button" className="pause-x" title="Remove this task" disabled={busy || tasks.length <= 1}
                      onClick={() => setTasks(tasks.filter((_, j) => j !== i))}>✕</button>
            </li>
          ))}
          <li className="pause-add">
            <button type="button" disabled={busy || tasks.length >= 12} onClick={() => setTasks([...tasks, ''])}>＋ Add a task</button>
          </li>
        </ol>
      )}

      {pause.point === 'developer' && payload.code && (
        <CodeBlock code={payload.code} filename={payload.filename} />
      )}

      {pause.point === 'qa_failure' && a && (
        <div className="pause-analysis">
          <div><b>{a.error_type}</b>{a.error_category ? ` (${a.error_category})` : ''}{a.code_location ? ` · ${a.code_location}` : ''}
            {a.repeat_error && <span className="pause-repeat"> · same error as last attempt</span>}</div>
          {a.hint && <div className="pause-hint">💡 {a.hint}</div>}
          {a.snippet && <CodeBlock code={a.snippet} />}
        </div>
      )}

      {pause.point === 'manual' && (
        <div className="pause-analysis">
          Paused after <b>{pause.node.replace('_', ' ')}</b>
          {payload.task && <> · task {payload.task_order}: {payload.task}</>}
          {payload.error && <pre className="pause-error">{payload.error}</pre>}
        </div>
      )}

      <div className="pause-fields">
        <textarea rows={2} value={guidance} disabled={busy} onChange={(e) => setGuidance(e.target.value)}
                  placeholder={isPlan ? 'Guidance — e.g. "split it into data model + API, no UI"' : 'Guidance — e.g. "use dataclasses, no external libraries"'} />
        <input value={clarification} disabled={busy} onChange={(e) => setClarification(e.target.value)}
               placeholder='Clarification — e.g. "I meant a CLI tool, not a web app"' />
      </div>

      <div className="pause-actions">
        <button className="primary" disabled={busy} onClick={() => send('approve')}>▶ {pause.point === 'qa_failure' ? 'Retry as suggested' : 'Approve & continue'}</button>
        {has('revise') && (
          <button disabled={busy || !canRevise} onClick={() => send('revise')}
                  title={canRevise ? '' : 'Write some guidance or edit a task first'}>
            ✎ {isPlan ? 'Apply changes' : 'Send feedback'}
          </button>
        )}
        {has('skip_task') && <button disabled={busy} onClick={() => send('skip_task')}>⏭ Skip this task</button>}
        {has('abort') && (
          <button className="danger" disabled={busy}
                  onClick={() => { if (window.confirm('Cancel this run? Files written so far stay available to download.')) send('abort') }}>
            ✕ Cancel run
          </button>
        )}
      </div>
    </section>
  )
}
