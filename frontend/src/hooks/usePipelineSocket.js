import { useCallback, useEffect, useRef, useState } from 'react'
import { EVENT_TO_NODE } from '../constants/eventTypes.js'
import { describeEvent } from '../lib/describeEvent.js'

import { BACKEND_URL } from '../lib/config.js'

const WS_BACKEND_URL = BACKEND_URL.replace(/^http/, 'ws')

/** Encapsulates everything needed to submit a request and watch it run
 * live: message history, which flow-diagram node is active, run status,
 * and the WebSocket plumbing itself. Kept independent of any rendering —
 * App.jsx only reads the returned state and calls submitRequest().
 */
export function usePipelineSocket() {
  const [messages, setMessages] = useState([])
  const [activeNode, setActiveNode] = useState(null)
  const [runStatus, setRunStatus] = useState('idle') // idle | running | done | needs_review
  const [error, setError] = useState(null)
  const [projectId, setProjectId] = useState(null)
  // User intervention: the pause the run is parked at (null when running), and
  // whether the user has asked for a pause that hasn't fired yet.
  const [pause, setPause] = useState(null)
  const [pendingPause, setPendingPause] = useState(false)
  const [feedbackBusy, setFeedbackBusy] = useState(false)
  const wsRef = useRef(null)
  const msgIdRef = useRef(0)

  const nextId = () => `m${msgIdRef.current++}`

  const pushMessage = useCallback((partial) => {
    setMessages((prev) => [...prev, { id: nextId(), ts: Date.now() / 1000, ...partial }])
  }, [])

  useEffect(() => () => wsRef.current?.close(), [])

  const connectToProject = useCallback((projectId) => {
    setProjectId(projectId)
    const ws = new WebSocket(`${WS_BACKEND_URL}/ws/projects/${projectId}`)
    wsRef.current = ws

    ws.onmessage = (raw) => {
      const event = JSON.parse(raw.data)
      const { text, badge, code, filename } = describeEvent(event)

      pushMessage({
        role: 'agent',
        agent: event.agent,
        text,
        badge,
        code,
        filename,
        ts: event.ts,
        download: event.type === 'pipeline_done' ? `${BACKEND_URL}/projects/${projectId}/download` : undefined,
      })

      if (event.type === 'pipeline_paused') { setPause(event.data); setPendingPause(false) }
      else if (event.type === 'pipeline_resumed') setPause(null)
      else if (event.type === 'pause_requested') setPendingPause(true)

      const nodeUpdate = EVENT_TO_NODE[event.type]
      if (nodeUpdate === 'idle') {
        setActiveNode(null)
        setPause(null)
        setPendingPause(false)
        setRunStatus(event.data.aborted ? 'cancelled' : event.data.needs_human_review ? 'needs_review' : 'done')
        ws.close()
      } else if (nodeUpdate !== null && nodeUpdate !== undefined) {
        setActiveNode(nodeUpdate)
      }
    }

    ws.onerror = () => {
      setError('Lost the live connection to the backend. The run may still be in progress server-side.')
    }
  }, [pushMessage])

  const submitRequest = useCallback(async (text, { interactive = false, framework = '' } = {}) => {
    if (!text.trim() || runStatus === 'running') return

    setError(null)
    setPause(null)
    setPendingPause(false)
    setRunStatus('running')
    setActiveNode('planner')
    pushMessage({ role: 'user', agent: 'user', text })

    try {
      const res = await fetch(`${BACKEND_URL}/request/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request: text, interactive, ...(framework ? { framework } : {}) }),
      })
      if (!res.ok) throw new Error(`backend returned ${res.status}`)
      const data = await res.json()
      connectToProject(data.project_id)
    } catch (err) {
      setError(`Could not submit request: ${err.message}`)
      setRunStatus('idle')
      setActiveNode(null)
    }
  }, [runStatus, pushMessage, connectToProject])

  // POST helper for the intervention endpoints; surfaces the backend's `detail` message.
  const post = useCallback(async (path, body) => {
    const res = await fetch(`${BACKEND_URL}/projects/${projectId}/${path}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    })
    if (!res.ok) {
      let detail = `backend returned ${res.status}`
      try { const j = await res.json(); if (j.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* not json */ }
      throw new Error(detail)
    }
    return res.json()
  }, [projectId])

  /** Ask the running pipeline to stop for the user at the next step. */
  const requestPause = useCallback(async () => {
    if (!projectId) return
    try { await post('intervention', { pause_after: 'next' }); setPendingPause(true); setError(null) }
    catch (err) { setError(`Could not request a pause: ${err.message}`) }
  }, [projectId, post])

  /** Answer the current pause: { action, guidance?, clarification?, tasks? }. */
  const sendFeedback = useCallback(async (feedback) => {
    setFeedbackBusy(true)
    try {
      await post('feedback', feedback)
      setError(null)
      return true
    } catch (err) {
      setError(`Could not send feedback: ${err.message}`)
      return false
    } finally {
      setFeedbackBusy(false)
    }
  }, [post])

  return { messages, activeNode, runStatus, error, projectId, submitRequest,
           pause, pendingPause, feedbackBusy, requestPause, sendFeedback }
}
