// Mirrors backend/app/pipeline/event_types.py by hand — JS can't import a
// Python module, so if you add a new event type constant there, add the
// matching mapping here too.
//
// Maps an incoming WebSocket event's `type` to which flow-diagram node
// should be highlighted right now. `null` means "don't change it" (e.g.
// task_complete is a bookkeeping event, not a change of which agent is
// active); 'idle' means "clear the highlight" (the whole run finished).
export const EVENT_TO_NODE = {
  planner_start: 'planner',
  planner_done: 'planner',
  developer_start: 'developer',
  developer_retry: 'developer',
  memory_hit: 'developer',
  developer_file_written: 'developer', // one per file, for multi-file task handlers
  developer_done: 'developer',
  venv_setup: null,
  venv_ready: null,
  venv_failed: null,
  qa_start: 'qa',
  qa_result: 'qa',
  // Phase 12: previously a single 'retry_scheduled' event (reused by both
  // qa_node AND failure_analyzer_node, under different agents) mapped
  // straight to 'retry-edge' — a direct QA->Developer arrow that doesn't
  // match the actual graph shape since the Failure Analyzer node was
  // added (QA fail -> Failure Analyzer -> targeted repair -> Developer).
  // Two distinct event types now carry the two distinct hops.
  failure_analysis_start: 'failure_analyzer', // QA fail -> handing off to the analyzer
  failure_analysis_done: 'retry-edge',        // analyzer done -> handing back to Developer
  retry_scheduled: null, // legacy event type, no longer emitted by the backend
  escalate: 'escalate',
  task_complete: null,
  pipeline_start: 'planner',
  pipeline_done: 'idle',
}
