import { frameworkLabel } from '../constants/frameworks.js'

// Pure, presentation-only helpers — no state, no side effects, easy to
// reason about (and to unit-test if this project adds a JS test runner
// later) independent of the WebSocket/React plumbing in usePipelineSocket.

/** Guesses a language tag for a code block from a filename. Kept as a
 * plain data attribute for now rather than pulling in a syntax-highlighting
 * library — see CodeBlock.jsx. */
export function languageFor(filename) {
  if (!filename) return ''
  const ext = filename.split('.').pop()
  return { py: 'python', html: 'html', css: 'css', txt: 'text', js: 'javascript', jsx: 'javascript', json: 'json', mjs: 'javascript' }[ext] || ext
}

/** Turns a raw backend event (see backend/app/pipeline/event_types.py) into
 * what a chat bubble should show: the narration text, an optional status
 * badge, and — for events that carry generated source (developer_done,
 * developer_file_written) — the code itself and its filename.
 */
export function describeEvent(event) {
  let badge = null
  let code = null
  let filename = null

  if (event.type === 'qa_result') {
    badge = event.data.passed
      ? { label: 'PASSED', tone: 'good' }
      : { label: 'FAILED', tone: 'bad' }
  } else if (event.type === 'escalate') {
    badge = { label: 'NEEDS REVIEW', tone: 'warn' }
  } else if (event.type === 'memory_hit') {
    badge = { label: '🧠 memory reused', tone: 'memory' }
  } else if (event.type === 'planner_done' && event.data.framework) {
    badge = { label: `🧱 ${frameworkLabel(event.data.framework)}`, tone: 'memory' }
  } else if (event.type === 'venv_ready') {
    badge = { label: event.data.framework && ['express', 'next'].includes(event.data.framework) ? '✅ deps ready' : '✅ venv ready', tone: 'good' }
  } else if (event.type === 'venv_failed') {
    badge = { label: event.data.framework && ['express', 'next'].includes(event.data.framework) ? '⚠ npm issue' : '⚠ venv issue', tone: 'warn' }
  } else if (event.type === 'failure_analysis_done') {
    // Failure Recovery: the analyzer's structured diagnosis. A repeat of the
    // previous attempt's error outranks the category badge — it's the
    // signal that the last fix didn't work.
    const a = event.data.analysis || {}
    badge = a.repeat_error
      ? { label: 'REPEAT ERROR', tone: 'bad' }
      : { label: `🩺 ${String(a.error_category || 'failure').toUpperCase()}`, tone: 'warn' }
  } else if (event.type === 'pause_requested') {
    badge = { label: '⏸ pause requested', tone: 'memory' }
  } else if (event.type === 'pipeline_paused') {
    badge = { label: '⏸ WAITING FOR YOU', tone: 'warn' }
  } else if (event.type === 'pipeline_resumed') {
    const a = event.data.action
    badge = event.data.source === 'timeout' ? { label: 'AUTO-CONTINUED', tone: 'warn' }
      : a === 'abort' ? { label: 'CANCELLED', tone: 'bad' }
      : a === 'skip_task' ? { label: 'TASK SKIPPED', tone: 'warn' }
      : a === 'revise' ? { label: '✎ FEEDBACK APPLIED', tone: 'memory' }
      : { label: '▶ APPROVED', tone: 'good' }
  } else if (event.type === 'plan_revised') {
    badge = { label: '✎ PLAN EDITED', tone: 'memory' }
  } else if (event.type === 'pipeline_done') {
    badge = event.data.aborted
      ? { label: 'CANCELLED', tone: 'bad' }
      : event.data.needs_human_review
        ? { label: 'NEEDS REVIEW', tone: 'warn' }
        : { label: 'ALL PASSED', tone: 'good' }
  }

  // Phase 7: these two event types carry actual source code — this is
  // what makes the chat feed show real generated code instead of only
  // narrating that something was written.
  if ((event.type === 'developer_done' || event.type === 'developer_file_written') && event.data.code) {
    code = event.data.code
    filename = event.data.filename
  }
  // Failure Recovery: show the line-numbered snippet the analyzer pinned the failure to.
  if (event.type === 'failure_analysis_done' && event.data.snippet) {
    code = event.data.snippet
    filename = event.data.filename
  }

  return { text: event.message, badge, code, filename }
}
