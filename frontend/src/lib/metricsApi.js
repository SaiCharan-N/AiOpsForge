import { BACKEND_URL } from './config.js'

async function getJson(path) {
  const res = await fetch(`${BACKEND_URL}${path}`)
  if (!res.ok) {
    let detail = ''
    try { detail = (await res.json()).detail } catch { /* not json */ }
    throw new Error(detail || `backend returned ${res.status}`)
  }
  return res.json()
}

export const metricsApi = {
  projects: (limit = 100) => getJson(`/metrics/projects?limit=${limit}`).then((d) => d.projects),
  project: (id) => getJson(`/projects/${encodeURIComponent(id)}/metrics`),
  compare: (ids) => getJson(`/metrics/compare?ids=${ids.map(encodeURIComponent).join(',')}`),
  summary: () => getJson('/metrics/summary'),
  models: () => getJson('/metrics/models'),
  regression: (model, window = 5, threshold = 0.2) =>
    getJson(`/metrics/regression?model=${encodeURIComponent(model)}&window=${window}&threshold=${threshold}`),
  exportUrl: (format) => `${BACKEND_URL}/metrics/export?format=${format}`,
}

// ---- formatting helpers shared by the dashboard components ----
export const fmtNum = (n) => (n == null ? '—' : Number(n).toLocaleString())
export const fmtCost = (c) => (c == null ? '—' : c < 0.01 ? `$${c.toFixed(5)}` : `$${c.toFixed(3)}`)
export const fmtPct = (p) => (p == null ? '—' : `${Math.round(p * 100)}%`)
export function fmtMs(ms) {
  if (ms == null) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`
}
