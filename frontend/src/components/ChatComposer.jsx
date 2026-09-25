export default function ChatComposer({ value, onChange, onSubmit, disabled }) {
  return (
    <form className="chat-composer" onSubmit={onSubmit}>
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
      <button type="submit" disabled={disabled || !value.trim()}>
        {disabled ? 'Building…' : 'Send'}
      </button>
    </form>
  )
}
