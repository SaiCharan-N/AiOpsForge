import { AGENT_META } from '../constants/agents.js'
import CodeBlock from './CodeBlock.jsx'

export default function Bubble({ message }) {
  const meta = AGENT_META[message.agent] || AGENT_META.system
  return (
    <div className={`bubble bubble-${message.agent}`}>
      <div className="bubble-avatar" style={{ background: meta.color }}>
        {meta.avatar}
      </div>
      <div className="bubble-body">
        <div className="bubble-meta">
          <span className="bubble-agent" style={{ color: meta.color }}>{meta.label}</span>
          <span className="bubble-time">
            {new Date(message.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        </div>
        <div className="bubble-text">{message.text}</div>
        {message.code && <CodeBlock code={message.code} filename={message.filename} />}
        {message.badge && (
          <span className={`bubble-badge bubble-badge-${message.badge.tone}`}>{message.badge.label}</span>
        )}
        {message.download && (
          <a className="download-link" href={message.download}>⬇ Download project</a>
        )}
      </div>
    </div>
  )
}
