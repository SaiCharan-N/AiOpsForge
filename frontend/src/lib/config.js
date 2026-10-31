// Single source of truth for where the backend lives (set VITE_BACKEND_URL at build time).
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
