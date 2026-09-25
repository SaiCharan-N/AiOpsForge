// Display metadata for each agent's chat bubble. Add a new agent (e.g. a
// future dedicated "Scaffolder" bubble distinct from "Developer") here —
// nothing else needs to change to render it correctly.
export const AGENT_META = {
  user:      { label: 'You',       color: '#a5b4fc', avatar: '🧑' },
  planner:   { label: 'Planner',   color: '#38bdf8', avatar: '🗂️' },
  developer: { label: 'Developer', color: '#34d399', avatar: '🧑‍💻' },
  qa:        { label: 'QA',        color: '#f59e0b', avatar: '🔍' },
  system:    { label: 'System',    color: '#9ca3af', avatar: '⚙️' },
}
