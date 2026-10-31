// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react'
import MetricsDash from '../src/pages/MetricsDash.jsx'

globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} }

const project = (id, over = {}) => ({
  project_id: id, name: `proj-${id}`, request: 'req', status: 'done', communication_mode: 'blackboard',
  model: 'qwen', task_types: 'code', tasks: 2, tasks_passed: 2, pass_rate: 1, attempts: 3, avg_attempts: 1.5,
  escalations: 0, repeated_errors: 0, tokens_in: 100, tokens_out: 50, tokens_total: 150, est_cost_usd: 0.000125,
  llm_calls: 3, total_ms: 12000, llm_ms: 9000, created_at: '2026-01-01', ...over,
})
const detail = (id) => ({
  project_id: id, communication_mode: 'blackboard', model: 'qwen',
  tokens: { tokens_in: 100, tokens_out: 50, total: 150, llm_calls: 3, by_agent: {
    planner: { calls: 1, tokens_in: 30, tokens_out: 10, est_cost: 0.00003 },
    developer: { calls: 2, tokens_in: 70, tokens_out: 40, est_cost: 0.0001 } } },
  cost: { est_cost_usd: 0.000125 },
  latency: { total_ms: 12000, node_total_ms: 11000, llm_ms: 9000, tool_ms: 1500, db_ms: 200, embed_ms: 100, other_ms: 200,
             network_ms: 300, model_load_ms: 2000, ollama_compute_ms: 8000, by_node: {} },
  tools: { write_file: { calls: 2, total_ms: 400, avg_ms: 200, errors: 0 } },
  nodes: [{ node: 'planner', task_order: null, attempt: 1, start_ms: 0, duration_ms: 3000, llm_ms: 2800, tool_ms: 0, db_ms: 100, embed_ms: 0, other_ms: 100, ok: true },
          { node: 'developer', task_order: 1, attempt: 1, start_ms: 3000, duration_ms: 9000, llm_ms: 6200, tool_ms: 1500, db_ms: 100, embed_ms: 100, other_ms: 1100, ok: true }],
  tasks: [{ order: 1, type: 'code', passed: true, attempts: 1, escalated: false, repeated_errors: 0 }],
  success: { pass_rate: 1, avg_attempts: 1 },
})
const summary = { overall: { tasks: 4, pass_rate: 0.75, avg_attempts: 1.5, escalation_rate: 0.25, timeout_rate: 0, repeat_error_rate: 0.5 },
  by_task_type: { code: { tasks: 4, pass_rate: 0.75 } }, attempt_histogram: { '1': 2, '2': 2 },
  failure_recovery: { avg_attempts_with_similar_fix: 2, avg_attempts_without_similar_fix: 2.5 } }
const models = { models: { qwen: { llm_calls: 6, ms_per_output_token: 12.5, tokens_per_s_generation: 80, cold_loads: 1, task_pass_rate: 0.75, task_avg_attempts: 1.5, slower_than_fastest: false },
                            slowmodel: { llm_calls: 3, ms_per_output_token: 40, tokens_per_s_generation: 25, cold_loads: 0, task_pass_rate: 0.5, task_avg_attempts: 2, slower_than_fastest: true, slowdown_vs_fastest: 3.2 } } }

function mockFetch() {
  globalThis.fetch = vi.fn(async (url) => {
    const u = String(url)
    const json = (b) => ({ ok: true, json: async () => b })
    if (u.includes('/metrics/projects')) return json({ projects: [project('aaaaaa1'), project('bbbbbb2', { tokens_total: 300, status: 'needs_human_review', communication_mode: 'message_passing' })] })
    if (u.includes('/metrics/summary')) return json(summary)
    if (u.includes('/metrics/models')) return json(models)
    if (u.includes('/metrics/compare')) return json({ projects: [detail('aaaaaa1'), detail('bbbbbb2')], deltas: [{ project_id: 'bbbbbb2', tokens_total: { baseline: 150, value: 300, pct_change: 100 }, est_cost_usd: { baseline: 1, value: 2, pct_change: 100 } }] })
    if (u.includes('/metrics/regression')) return json({ model: 'slowmodel', regression: true, recent_ms_per_token: 40, baseline_ms_per_token: 20, pct_change: 100 })
    if (u.includes('/projects/') && u.endsWith('/metrics')) return json(detail(decodeURIComponent(u.split('/projects/')[1].split('/')[0])))
    return { ok: false, status: 404, json: async () => ({ detail: 'nope' }) }
  })
}

describe('MetricsDash', () => {
  beforeEach(() => { cleanup(); mockFetch() })

  it('renders KPIs, project table, models; sorts; filters', async () => {
    render(<MetricsDash />)
    await waitFor(() => expect(screen.getByText('proj-aaaaaa1')).toBeTruthy())
    expect(screen.getAllByText('450').length).toBeGreaterThan(0)          // total tokens KPI = 150 + 300
    expect(screen.getAllByText('75%').length).toBeGreaterThan(1)                          // task pass rate KPI
    expect(screen.getByText(/3\.2× slower/)).toBeTruthy()                   // slow-model flag
    fireEvent.click(screen.getAllByRole('columnheader').find((h) => h.textContent.startsWith('Tokens')))                            // sort desc by tokens
    const rows = screen.getAllByRole('row').map((r) => r.textContent)
    expect(rows.findIndex((t) => t.includes('bbbbbb2'))).toBeLessThan(rows.findIndex((t) => t.includes('aaaaaa1')))
    fireEvent.change(screen.getAllByRole('combobox')[1], { target: { value: 'message_passing' } })
    await waitFor(() => expect(screen.queryByText('proj-aaaaaa1')).toBeNull())
    expect(screen.getByText('proj-bbbbbb2')).toBeTruthy()
  })

  it('drill-down shows breakdown + waterfall + tools; compare shows % change; regression flagged', async () => {
    render(<MetricsDash />)
    await waitFor(() => expect(screen.getByText('proj-aaaaaa1')).toBeTruthy())
    fireEvent.click(screen.getByText('proj-aaaaaa1'))
    await waitFor(() => expect(screen.getByText(/Run detail/)).toBeTruthy())
    expect(screen.getByText('Per-node timeline')).toBeTruthy()
    expect(screen.getByText('write_file')).toBeTruthy()
    expect(screen.getByText(/developer · t1/)).toBeTruthy()

    const boxes = screen.getAllByRole('checkbox')
    fireEvent.click(boxes[0]); fireEvent.click(boxes[1])
    await waitFor(() => expect(screen.getByText('Comparison')).toBeTruthy())
    expect(screen.getAllByText('+100%').length).toBeGreaterThan(0)

    fireEvent.click(screen.getAllByText('Check regression')[1])
    await waitFor(() => expect(screen.getByText(/REGRESSION/)).toBeTruthy())
  })

  it('shows a readable error when the backend is unreachable', async () => {
    globalThis.fetch = vi.fn(async () => { throw new Error('Failed to fetch') })
    render(<MetricsDash />)
    await waitFor(() => expect(screen.getByText(/Could not load metrics: Failed to fetch/)).toBeTruthy())
  })

  it('empty state does not crash', async () => {
    globalThis.fetch = vi.fn(async (u) => ({ ok: true, json: async () => (String(u).includes('projects') ? { projects: [] } : String(u).includes('models') ? { models: {} } : { overall: { tasks: 0 }, by_task_type: {}, attempt_histogram: {}, failure_recovery: {} }) }))
    render(<MetricsDash />)
    await waitFor(() => expect(screen.getByText(/No runs with metrics yet/)).toBeTruthy())
  })
})
