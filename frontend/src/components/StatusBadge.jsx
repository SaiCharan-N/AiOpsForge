/** Small coloured pill for run / batch / experiment states. */
const LABELS = { needs_review: 'needs review' }
export default function StatusBadge({ status }) {
  return <span className={`status-badge s-${status}`}>{LABELS[status] || status}</span>
}
