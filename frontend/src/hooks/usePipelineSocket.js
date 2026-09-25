import { useCallback, useEffect, useRef, useState } from 'react'
import { EVENT_TO_NODE } from '../constants/eventTypes.js'
import { describeEvent } from '../lib/describeEvent.js'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
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
  const wsRef = useRef(null)
  const msgIdRef = useRef(0)

  const nextId = () => `m${msgIdRef.current++}`

  const pushMessage = useCallback((partial) => {
    setMessages((prev) => [...prev, { id: nextId(), ts: Date.now() / 1000, ...partial }])
  }, [])

  useEffect(() => () => wsRef.current?.close(), [])

  const connectToProject = useCallback((projectId) => {
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

      const nodeUpdate = EVENT_TO_NODE[event.type]
      if (nodeUpdate === 'idle') {
        setActiveNode(null)
        setRunStatus(event.data.needs_human_review ? 'needs_review' : 'done')
        ws.close()
      } else if (nodeUpdate !== null && nodeUpdate !== undefined) {
        setActiveNode(nodeUpdate)
      }
    }

    ws.onerror = () => {
      setError('Lost the live connection to the backend. The run may still be in progress server-side.')
    }
  }, [pushMessage])

  const submitRequest = useCallback(async (text) => {
    if (!text.trim() || runStatus === 'running') return

    setError(null)
    setRunStatus('running')
    setActiveNode('planner')
    pushMessage({ role: 'user', agent: 'user', text })

    try {
      const res = await fetch(`${BACKEND_URL}/request/async`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request: text }),
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

  return { messages, activeNode, runStatus, error, submitRequest }
}
