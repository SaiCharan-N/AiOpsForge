import { useEffect, useRef, useState } from 'react'
import FlowDiagram from './FlowDiagram.jsx'
import Bubble from './components/Bubble.jsx'
import ChatComposer from './components/ChatComposer.jsx'
import { usePipelineSocket } from './hooks/usePipelineSocket.js'

/** Full-viewport split layout: chat on the left, live agent flow diagram on
 * the right, both stretching to fill the available space — replaces the
 * old single centered column. All WebSocket/state logic still lives in
 * usePipelineSocket; this file only arranges the panes.
 */
export default function App() {
  const [requestText, setRequestText] = useState('')
  const { messages, activeNode, runStatus, error, submitRequest } = usePipelineSocket()
  const feedRef = useRef(null)

  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  function handleSubmit(e) {
    e.preventDefault()
    const text = requestText.trim()
    if (!text) return
    submitRequest(text)
    setRequestText('')
  }

  return (
    <div className="app-split">
      <header className="topbar">
        <h1>AIOpsForge</h1>
        <p className="subtitle">Ask it to build something — watch Planner, Developer, and QA work it out live.</p>
      </header>

      <div className="split-body">
        <section className="split-left">
          <section className="chat-feed" ref={feedRef}>
            {messages.length === 0 && (
              <div className="chat-empty">
                Try something like: <em>"a function that checks if a string is a palindrome, with a test"</em>
                {' '}or <em>"build a website for a bakery with a homepage and contact page"</em>
              </div>
            )}
            {messages.map((m) => <Bubble key={m.id} message={m} />)}
          </section>

          {error && <div className="error-banner">{error}</div>}

          <ChatComposer
            value={requestText}
            onChange={setRequestText}
            onSubmit={handleSubmit}
            disabled={runStatus === 'running'}
          />
        </section>

        <section className="split-right">
          <FlowDiagram activeNode={activeNode} />
        </section>
      </div>
    </div>
  )
}
