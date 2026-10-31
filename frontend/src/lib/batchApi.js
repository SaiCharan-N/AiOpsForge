import { BACKEND_URL } from './config.js'

async function request(path, options) {
  const res = await fetch(`${BACKEND_URL}${path}`, options)
  if (!res.ok) {
    let detail = ''
    try { const j = await res.json(); detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* not json */ }
    throw new Error(detail || `backend returned ${res.status}`)
  }
  return res.json()
}
const post = (path, body) => request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) })

export const batchApi = {
  createBatch: (body) => post('/batch', body),
  batches: () => request('/batch?limit=50').then((d) => d.batches),
  batch: (id) => request(`/batch/${id}`),
  batchResults: (id) => request(`/batch/${id}/results`),
  cancelBatch: (id) => post(`/batch/${id}/cancel`),
  presets: () => request('/experiments/presets'),
  createExperiment: (body) => post('/experiments', body),
  experiments: () => request('/experiments?limit=50').then((d) => d.experiments),
  experiment: (id) => request(`/experiments/${id}`),
  experimentResults: (id) => request(`/experiments/${id}/results`),
  cancelExperiment: (id) => post(`/experiments/${id}/cancel`),
  batchCsvUrl: (id) => `${BACKEND_URL}/batch/${id}/results?format=csv`,
  experimentUrl: (id, format) => `${BACKEND_URL}/experiments/${id}/results?format=${format}`,
}

export const fmtP = (p) => (p == null ? '—' : p < 0.001 ? '<0.001' : p.toFixed(3))
export const fmtNumber = (x, d = 2) => (x == null ? '—' : Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : Number(x).toFixed(d).replace(/\.?0+$/, ''))
export const isActive = (s) => s === 'running' || s === 'pending'
