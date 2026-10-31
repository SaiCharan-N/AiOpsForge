import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import FlowDiagram from './FlowDiagram.jsx'
import Bubble from './components/Bubble.jsx'
import ChatComposer from './components/ChatComposer.jsx'
import PausePanel from './components/PausePanel.jsx'
import { usePipelineSocket } from './hooks/usePipelineSocket.js'

// Recharts is ~half the bundle, so the dashboard is only downloaded the first time it's opened.
const MetricsDash = lazy(() => import('./pages/MetricsDash.jsx'))
const BatchesPage = lazy(() => import('./pages/BatchesPage.jsx'))

/** Full-viewport split layout: chat on the left, live agent flow diagram on
 * the right, both stretching to fill the available space — replaces the
 * old single centered column. All WebSocket/state logic still lives in
 * usePipelineSocket; this file only arranges the panes.
 */
export default function App() {
  const [requestText, setRequestText] = useState('')
  const [view, setView] = useState('chat') // 'chat' | 'metrics' | 'batches'
  const [metricsOpened, setMetricsOpened] = useState(false)
  const [batchesOpened, setBatchesOpened] = useState(false)
  const [focusProject, setFocusProject] = useState(null) // a project opened from a batch
  const [interactive, setInteractive] = useState(false)
  const [framework, setFramework] = useState('')
  const {
    messages, activeNode, runStatus, error, projectId, submitRequest,
    pause, pendingPause, feedbackBusy, requestPause, sendFeedback,
  } = usePipelineSocket()
  const feedRef = useRef(null)

  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  function handleSubmit(e) {
    e.preventDefault()
    const text = requestText.trim()
    if (!text) return
    submitRequest(text, { interactive, framework })
    setRequestText('')
  }

  return (
    <div className="app-split">
      <header className="topbar">
        <div>
          <h1>AIOpsForge</h1>
          <p className="subtitle">Ask it to build something — watch Planner, Developer, and QA work it out live.</p>
        </div>
        <nav className="tabs">
          <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>
            Chat{runStatus === 'running' ? ' •' : ''}
          </button>
          <button className={view === 'metrics' ? 'active' : ''}
                  onClick={() => { setView('metrics'); setMetricsOpened(true) }}>
            Metrics
          </button>
          <button className={view === 'batches' ? 'active' : ''}
                  onClick={() => { setView('batches'); setBatchesOpened(true) }}>
            Batches
          </button>
        </nav>
      </header>

      {metricsOpened && (
        <div className="metrics-view" hidden={view !== 'metrics'}>
          <Suspense fallback={<div className="chat-empty">Loading dashboard…</div>}>
            <MetricsDash focusProjectId={focusProject || (runStatus === 'done' || runStatus === 'needs_review' ? projectId : null)} />
          </Suspense>
        </div>
      )}

      {batchesOpened && (
        <div className="metrics-view" hidden={view !== 'batches'}>
          <Suspense fallback={<div className="chat-empty">Loading…</div>}>
            <BatchesPage onOpenProject={(id) => { setFocusProject(id); setMetricsOpened(true); setView('metrics') }} />
          </Suspense>
        </div>
      )}

      {/* The chat pane stays mounted (just hidden) while the Metrics tab is open, so a live run keeps streaming. */}
      <div className="split-body" hidden={view !== 'chat'}>
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

          {pause && <PausePanel pause={pause} busy={feedbackBusy} onSubmit={sendFeedback} />}

          <ChatComposer
            value={requestText}
            onChange={setRequestText}
            onSubmit={handleSubmit}
            disabled={runStatus === 'running'}
            interactive={interactive}
            onInteractiveChange={setInteractive}
            framework={framework}
            onFrameworkChange={setFramework}
            running={runStatus === 'running'}
            paused={Boolean(pause)}
            pendingPause={pendingPause}
            onPause={requestPause}
          />
        </section>

        <section className="split-right">
          <FlowDiagram activeNode={activeNode} paused={Boolean(pause)} />
        </section>
      </div>
    </div>
  )
}
