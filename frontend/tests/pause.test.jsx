// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, cleanup, act } from '@testing-library/react'
import App from '../src/App.jsx'

globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }
Element.prototype.scrollTo = () => {}

let sockets = []
class FakeWS {
  constructor(url) { this.url = url; this.closed = false; sockets.push(this) }
  close() { this.closed = true }
  emit(type, data = {}, message = '', agent = 'system') {
    act(() => { this.onmessage({ data: JSON.stringify({ type, agent, message, data, ts: Date.now() / 1000 }) }) })
  }
}
globalThis.WebSocket = FakeWS

const PLAN = [{ order: 1, description: 'Write add(a, b) with a test', type: 'code' }, { order: 2, description: 'Write sub(a, b) with a test', type: 'code' }]
const pausedEvent = (point, node, extra = {}) => ({
  point, node, question: point === 'planner' ? 'Here is the plan. Does it make sense?' : 'Is this the right approach?',
  options: point === 'qa_failure' ? ['approve', 'revise', 'skip_task', 'abort'] : ['approve', 'revise', 'abort'],
  payload: point === 'planner' ? { tasks: PLAN } : point === 'developer' ? { filename: 'add.py', code: 'def add(a, b):\n    return a + b' }
    : { analysis: { error_type: 'TypeError', error_category: 'type', code_location: 'add.py:3', hint: 'total is None', snippet: '>   3 | x' } },
  timeout_s: 900, since: Date.now() / 1000, ...extra,
})

let calls
beforeEach(() => {
  cleanup(); sockets = []; calls = []
  globalThis.window.confirm = vi.fn(() => true)
  globalThis.fetch = vi.fn(async (url, opts = {}) => {
    calls.push({ url: String(url), body: opts.body ? JSON.parse(opts.body) : null })
    if (String(url).endsWith('/request/async')) return { ok: true, json: async () => ({ project_id: 'p123', status: 'building' }) }
    if (String(url).endsWith('/feedback') && globalThis.__fail) return { ok: false, status: 409, json: async () => ({ detail: 'The run is not waiting for feedback right now' }) }
    return { ok: true, json: async () => ({ accepted: true }) }
  })
  globalThis.__fail = false
})

async function startRun({ interactive = false } = {}) {
  render(<App />)
  if (interactive) fireEvent.click(screen.getByLabelText(/Pause for my review/))
  fireEvent.change(screen.getByPlaceholderText('What should it build?'), { target: { value: 'build add and sub' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(sockets.length).toBe(1))
  return sockets[0]
}

describe('user intervention UI', () => {
  it('sends interactive flag only when toggled; plain Send is unchanged', async () => {
    await startRun({ interactive: false })
    expect(calls.find((c) => c.url.endsWith('/request/async')).body).toEqual({ request: 'build add and sub', interactive: false })
    cleanup(); calls.length = 0; sockets = []
    await startRun({ interactive: true })
    expect(calls.find((c) => c.url.endsWith('/request/async')).body.interactive).toBe(true)
  })

  it('plan pause: panel appears, tasks are editable, edit/remove/add -> revise payload; approve payload', async () => {
    const ws = await startRun({ interactive: true })
    expect(screen.queryByRole('region', { name: 'Pipeline paused' })).toBeNull()
    ws.emit('pipeline_paused', pausedEvent('planner', 'planner'), 'Here is the plan. Does it make sense?')
    await screen.findByRole('region', { name: 'Pipeline paused' })
    expect(screen.getByText(/Paused — waiting for your input/)).toBeTruthy()      // flow diagram note

    const apply = screen.getByRole('button', { name: /Apply changes/ })
    expect(apply.disabled).toBe(true)                                              // nothing changed yet
    const boxes = screen.getAllByRole('textbox').filter((t) => t.value.startsWith('Write'))
    expect(boxes.length).toBe(2)
    fireEvent.change(boxes[1], { target: { value: 'Write mul(a, b) with a test' } })
    fireEvent.click(screen.getByText('＋ Add a task'))
    const all = screen.getAllByRole('textbox').filter((t) => t.tagName === 'TEXTAREA' && (t.value.startsWith('Write') || t.value === ''))
    fireEvent.change(all[2], { target: { value: 'Write div(a, b) with a test' } })
    fireEvent.click(screen.getAllByTitle('Remove this task')[0])                   // drop task 1
    expect(apply.disabled).toBe(false)
    fireEvent.change(screen.getByPlaceholderText(/Clarification/), { target: { value: 'ints only' } })
    fireEvent.click(apply)
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/projects/p123/feedback'))).toBe(true))
    const fb = calls.find((c) => c.url.endsWith('/feedback')).body
    expect(fb).toEqual({ action: 'revise', clarification: 'ints only', tasks: ['Write mul(a, b) with a test', 'Write div(a, b) with a test'] })

    ws.emit('pipeline_resumed', { point: 'planner', action: 'revise' }, 'Feedback received — applying it.')
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Pipeline paused' })).toBeNull())
    expect(screen.getByText('✎ FEEDBACK APPLIED')).toBeTruthy()
  })

  it('approve sends only the action; guidance-only revise omits tasks', async () => {
    const ws = await startRun({ interactive: true })
    ws.emit('pipeline_paused', pausedEvent('developer', 'developer'))
    await screen.findByText('add.py')                                              // first-file preview
    fireEvent.click(screen.getByRole('button', { name: /Approve & continue/ }))
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/feedback'))).toBe(true))
    expect(calls.find((c) => c.url.endsWith('/feedback')).body).toEqual({ action: 'approve' })

    calls.length = 0
    ws.emit('pipeline_resumed', { action: 'approve' })
    ws.emit('pipeline_paused', pausedEvent('developer', 'developer', { since: Date.now() / 1000 + 1 }))
    fireEvent.change(await screen.findByPlaceholderText(/Guidance/), { target: { value: 'add docstrings' } })
    fireEvent.click(screen.getByRole('button', { name: /Send feedback/ }))
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/feedback'))).toBe(true))
    expect(calls.find((c) => c.url.endsWith('/feedback')).body).toEqual({ action: 'revise', guidance: 'add docstrings' })
  })

  it('qa_failure pause shows the diagnosis and offers skip; abort asks for confirmation', async () => {
    const ws = await startRun({ interactive: true })
    ws.emit('pipeline_paused', pausedEvent('qa_failure', 'failure_analyzer'))
    await screen.findByText(/total is None/)
    expect(screen.getByText('TypeError')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Retry as suggested/ })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Skip this task/ }))
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/feedback'))).toBe(true))
    expect(calls.find((c) => c.url.endsWith('/feedback')).body).toEqual({ action: 'skip_task' })

    calls.length = 0
    globalThis.window.confirm = vi.fn(() => false)
    fireEvent.click(screen.getByRole('button', { name: /Cancel run/ }))
    expect(calls.length).toBe(0)                                                    // declined the confirm -> nothing sent
    globalThis.window.confirm = vi.fn(() => true)
    fireEvent.click(screen.getByRole('button', { name: /Cancel run/ }))
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/feedback'))).toBe(true))
    expect(calls.find((c) => c.url.endsWith('/feedback')).body).toEqual({ action: 'abort' })
  })

  it('a rejected answer keeps the panel and shows the backend message', async () => {
    const ws = await startRun({ interactive: true })
    ws.emit('pipeline_paused', pausedEvent('planner', 'planner'))
    await screen.findByRole('region', { name: 'Pipeline paused' })
    globalThis.__fail = true
    fireEvent.click(screen.getByRole('button', { name: /Approve & continue/ }))
    await screen.findByText(/not waiting for feedback right now/)
    expect(screen.getByRole('region', { name: 'Pipeline paused' })).toBeTruthy()
  })

  it('Pause button asks for a pause on a running (non-interactive) run, then reflects pending state', async () => {
    const ws = await startRun({ interactive: false })
    fireEvent.click(await screen.findByRole('button', { name: '⏸ Pause' }))
    await waitFor(() => expect(calls.some((c) => c.url.endsWith('/projects/p123/intervention'))).toBe(true))
    expect(calls.find((c) => c.url.endsWith('/intervention')).body).toEqual({ pause_after: 'next' })
    await screen.findByRole('button', { name: /Pausing at next step/ })
    ws.emit('pipeline_paused', pausedEvent('qa_failure', 'failure_analyzer', { point: 'manual' }))
    await screen.findByRole('region', { name: 'Pipeline paused' })
    expect(screen.queryByRole('button', { name: /Pausing at next step/ })).toBeNull()   // pause button hidden while paused
  })

  it('abort ends the run as cancelled (panel gone, composer usable again)', async () => {
    const ws = await startRun({ interactive: true })
    ws.emit('pipeline_paused', pausedEvent('planner', 'planner'))
    await screen.findByRole('region', { name: 'Pipeline paused' })
    ws.emit('pipeline_resumed', { action: 'abort' }, 'Run cancelled at your request.')
    ws.emit('pipeline_done', { done: false, needs_human_review: false, aborted: true }, 'Run cancelled.')
    await waitFor(() => expect(screen.queryByRole('region', { name: 'Pipeline paused' })).toBeNull())
    expect(screen.getAllByText('CANCELLED').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy()               // not stuck on "Building…"
    expect(ws.closed).toBe(true)
  })
})
