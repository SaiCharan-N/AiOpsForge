import { useMemo, useState } from 'react'
import SortableTable from './SortableTable.jsx'

/** SortableTable plus a filter bar: free-text search over the request, and
 * dropdowns / a cost range over whatever `filters` describe. Rows are filtered
 * here; sorting is the table's own (click a header).
 *
 *   filters: [{ key: 'status', label: 'status' }, ...]  -> a <select> of the distinct values of row[key]
 *   costKey: row field holding the cost, enables the min/max cost inputs
 */
export default function ResultsTable({ rows, columns, rowKey, filters = [], textKey = 'request', costKey, ...tableProps }) {
  const [text, setText] = useState('')
  const [selected, setSelected] = useState({})
  const [minCost, setMinCost] = useState('')
  const [maxCost, setMaxCost] = useState('')

  const options = useMemo(() => Object.fromEntries(filters.map((f) => [
    f.key, [...new Set(rows.map((r) => r[f.key]).filter((v) => v != null && v !== ''))].sort(),
  ])), [rows, filters])

  const visible = useMemo(() => rows.filter((r) => {
    if (text && !String(r[textKey] ?? '').toLowerCase().includes(text.toLowerCase())) return false
    for (const f of filters) if (selected[f.key] && String(r[f.key]) !== selected[f.key]) return false
    if (costKey) {
      const c = r[costKey]
      if (minCost !== '' && (c == null || c < Number(minCost))) return false
      if (maxCost !== '' && (c == null || c > Number(maxCost))) return false
    }
    return true
  }), [rows, text, selected, minCost, maxCost, filters, textKey, costKey])

  return (
    <div className="results-table">
      <div className="filter-bar">
        <input placeholder="Search request…" value={text} onChange={(e) => setText(e.target.value)} />
        {filters.map((f) => (
          <select key={f.key} aria-label={`filter ${f.label}`} value={selected[f.key] || ''}
                  onChange={(e) => setSelected({ ...selected, [f.key]: e.target.value })}>
            <option value="">any {f.label}</option>
            {options[f.key].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        ))}
        {costKey && (
          <span className="cost-range">
            cost $<input type="number" step="any" min="0" placeholder="min" aria-label="min cost" value={minCost} onChange={(e) => setMinCost(e.target.value)} />
            –<input type="number" step="any" min="0" placeholder="max" aria-label="max cost" value={maxCost} onChange={(e) => setMaxCost(e.target.value)} />
          </span>
        )}
        <span className="filter-count">{visible.length} of {rows.length}</span>
      </div>
      <SortableTable columns={columns} rows={visible} rowKey={rowKey} {...tableProps} />
    </div>
  )
}
