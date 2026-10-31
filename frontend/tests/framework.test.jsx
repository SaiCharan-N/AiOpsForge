// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, cleanup, act } from '@testing-library/react'
import App from '../src/App.jsx'

globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }
Element.prototype.scrollTo = () => {}
let sockets = []
class FakeWS {
  constructor() { sockets.push(this) }
  close() {}
  emit(type, data = {}, message = '', agent = 'planner') { act(() => { this.onmessage({ data: JSON.stringify({ type, agent, message, data, ts: Date.now() / 1000 }) }) }) }
}
globalThis.WebSocket = FakeWS

let bodies
beforeEach(() => {
  cleanup(); sockets = []; bodies = []
  globalThis.fetch = vi.fn(async (url, opts = {}) => {
    bodies.push(JSON.parse(opts.body || 'null'))
    return { ok: true, json: async () => ({ project_id: 'p1', status: 'building' }) }
  })
})

async function send(text) {
  fireEvent.change(screen.getByPlaceholderText('What should it build?'), { target: { value: text } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(sockets.length).toBe(1))
}

describe('framework picker', () => {
  it('lists every supported framework and defaults to auto-detect (nothing sent)', async () => {
    render(<App />)
    const select = screen.getByTitle(/Which web framework/)
    expect([...select.options].map((o) => o.value)).toEqual(['', 'fastapi', 'flask', 'django', 'express', 'next'])
    expect(select.value).toBe('')
    await send('build a bakery website')
    expect(bodies[0]).toEqual({ request: 'build a bakery website', interactive: false })   // no `framework` key at all
  })

  it('sends the chosen framework, and the control locks while a run is active', async () => {
    render(<App />)
    fireEvent.change(screen.getByTitle(/Which web framework/), { target: { value: 'django' } })
    await send('build a bakery website')
    expect(bodies[0]).toEqual({ request: 'build a bakery website', interactive: false, framework: 'django' })
    expect(screen.getByTitle(/Which web framework/).disabled).toBe(true)
  })

  it('shows the resolved framework as a badge on the plan, and a deps badge for npm installs', async () => {
    render(<App />)
    await send('Make an Express server for a bakery')
    const ws = sockets[0]
    ws.emit('planner_done', { tasks: [], framework: 'express' }, 'Planned 1 task(s):\n1. Make an Express server')
    await screen.findByText('🧱 Express (Node.js)')
    ws.emit('venv_ready', { ok: true, framework: 'express' }, 'Dependencies installed.', 'system')
    await screen.findByText('✅ deps ready')
    ws.emit('venv_ready', { ok: true, framework: 'flask' }, 'Virtualenv ready.', 'system')
    await screen.findByText('✅ venv ready')
  })
})
