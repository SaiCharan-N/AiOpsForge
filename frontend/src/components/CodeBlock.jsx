import { languageFor } from '../lib/describeEvent.js'

export default function CodeBlock({ code, filename }) {
  return (
    <div className="code-block">
      {filename && <div className="code-block-filename">{filename}</div>}
      <pre data-lang={languageFor(filename)}><code>{code}</code></pre>
    </div>
  )
}
