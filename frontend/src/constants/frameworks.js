// Mirrors backend/app/scaffolds (GET /frameworks is the source of truth).
// `value: ''` = auto-detect from the request text.
export const FRAMEWORKS = [
  { value: '', label: 'Auto-detect framework' },
  { value: 'fastapi', label: 'FastAPI' },
  { value: 'flask', label: 'Flask' },
  { value: 'django', label: 'Django' },
  { value: 'express', label: 'Express (Node.js)' },
  { value: 'next', label: 'Next.js' },
]

export const frameworkLabel = (id) => FRAMEWORKS.find((f) => f.value === id)?.label || id
