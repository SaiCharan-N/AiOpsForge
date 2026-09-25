// Pure, presentation-only helpers — no state, no side effects, easy to
// reason about (and to unit-test if this project adds a JS test runner
// later) independent of the WebSocket/React plumbing in usePipelineSocket.

/** Guesses a language tag for a code block from a filename. Kept as a
 * plain data attribute for now rather than pulling in a syntax-highlighting
 * library — see CodeBlock.jsx. */
export function languageFor(filename) {
  if (!filename) return ''
  const ext = filename.split('.').pop()
  return { py: 'python', html: 'html', css: 'css', txt: 'text', js: 'javascript' }[ext] || ext
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
  } else if (event.type === 'venv_ready') {
    badge = { label: '✅ venv ready', tone: 'good' }
  } else if (event.type === 'venv_failed') {
    badge = { label: '⚠ venv issue', tone: 'warn' }
  } else if (event.type === 'pipeline_done') {
    badge = event.data.needs_human_review
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

  return { text: event.message, badge, code, filename }
}
