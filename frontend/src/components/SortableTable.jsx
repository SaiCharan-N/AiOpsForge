import { useMemo, useState } from 'react'

/** Small sortable table. columns: [{ key, label, render?(row), align?, sortValue?(row) }].
 * Clicking a header sorts; `renderPrefix` lets a page add a leading cell
 * (e.g. a compare checkbox) without the table knowing about it.
 */
export default function SortableTable({
  columns, rows, rowKey, onRowClick, selectedKey, renderPrefix, initialSort, emptyText = 'Nothing to show yet',
}) {
  const [sort, setSort] = useState(initialSort || null) // { key, dir }

  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find((c) => c.key === sort.key)
    const val = (r) => (col?.sortValue ? col.sortValue(r) : r[sort.key])
    return [...rows].sort((a, b) => {
      const x = val(a), y = val(b)
      if (x == null && y == null) return 0
      if (x == null) return 1
      if (y == null) return -1
      const cmp = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))
      return sort.dir === 'asc' ? cmp : -cmp
    })
  }, [rows, sort, columns])

  function toggle(key) {
    setSort((s) => (s?.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' }))
  }

  return (
    <div className="table-wrap">
      <table className="metrics-table">
        <thead>
          <tr>
            {renderPrefix && <th />}
            {columns.map((c) => (
              <th key={c.key} className={c.align === 'right' ? 'num' : ''} onClick={() => toggle(c.key)}>
                {c.label}{sort?.key === c.key ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr><td colSpan={columns.length + (renderPrefix ? 1 : 0)} className="empty-cell">{emptyText}</td></tr>
          )}
          {sorted.map((r) => (
            <tr key={r[rowKey]} className={r[rowKey] === selectedKey ? 'selected' : ''}
                onClick={() => onRowClick?.(r)}>
              {renderPrefix && <td onClick={(e) => e.stopPropagation()}>{renderPrefix(r)}</td>}
              {columns.map((c) => (
                <td key={c.key} className={c.align === 'right' ? 'num' : ''}>
                  {c.render ? c.render(r) : (r[c.key] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
