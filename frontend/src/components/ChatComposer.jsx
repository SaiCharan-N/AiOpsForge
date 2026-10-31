import { FRAMEWORKS } from '../constants/frameworks.js'

/** The request box. Besides sending, it carries the two intervention controls
 * that belong with a request: the "pause for my review" switch (read when the
 * run STARTS) and, while a run is going, a Pause button (ask it to stop at the
 * next step). Both are optional — a plain Send behaves exactly as before.
 */
export default function ChatComposer({
  value, onChange, onSubmit, disabled,
  interactive = false, onInteractiveChange,
  framework = '', onFrameworkChange,
  running = false, paused = false, pendingPause = false, onPause,
}) {
  return (
    <form className="chat-composer" onSubmit={onSubmit}>
      <div className="composer-main">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="What should it build?"
          rows={2}
          disabled={disabled}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSubmit(e)
            }
          }}
        />
        <div className="composer-options">
          <label className={`toggle${disabled ? ' disabled' : ''}`}
                 title="Stops after the plan, after the first file, and after a failed test so you can steer it">
            <input type="checkbox" checked={interactive} disabled={disabled}
                   onChange={(e) => onInteractiveChange?.(e.target.checked)} />
            Pause for my review
          </label>
          <select className="framework-select" value={framework} disabled={disabled}
                  onChange={(e) => onFrameworkChange?.(e.target.value)}
                  title="Which web framework to build with (a request that names one is detected automatically)">
            {FRAMEWORKS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
          </select>
          {running && !paused && (
            <button type="button" className="pause-now" disabled={pendingPause} onClick={onPause}>
              {pendingPause ? '⏸ Pausing at next step…' : '⏸ Pause'}
            </button>
          )}
        </div>
      </div>
      <button type="submit" disabled={disabled || !value.trim()}>
        {disabled ? (paused ? 'Waiting…' : 'Building…') : 'Send'}
      </button>
    </form>
  )
}
