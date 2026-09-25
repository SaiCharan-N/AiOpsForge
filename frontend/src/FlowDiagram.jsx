import { useEffect, useState } from 'react'

/* Phase 5: a small, dependency-free SVG rendering of the pipeline topology
 * from backend/app/graph.py — Planner -> Developer -> QA, with the retry
 * loop back to Developer and the escalate branch on exhausted retries.
 * `activeNode` is one of 'planner' | 'developer' | 'qa' | 'escalate' |
 * 'retry-edge' | null, driven by the latest WebSocket event — see
 * EVENT_TO_NODE in App.jsx.
 *
 * Deliberately hand-rolled rather than a charting/flow library: the graph
 * is fixed and small (5 nodes), so a real dependency would be overkill for
 * what's ultimately four boxes and some arrows.
 */

const NODES = {
  planner:   { x: 40,  y: 70,  label: 'Planner',   sub: 'breaks request into tasks' },
  developer: { x: 260, y: 70,  label: 'Developer',  sub: 'writes code via MCP' },
  qa:        { x: 480, y: 70,  label: 'QA',         sub: 'runs pytest via MCP' },
  escalate:  { x: 480, y: 210, label: 'Escalate',   sub: 'needs human review' },
}

const NODE_W = 160
const NODE_H = 64

function Node({ id, active, pulse }) {
  const n = NODES[id]
  return (
    <g transform={`translate(${n.x}, ${n.y})`}>
      <rect
        width={NODE_W}
        height={NODE_H}
        rx={12}
        className={`flow-node${active ? ' flow-node-active' : ''}${pulse ? ' flow-node-pulse' : ''}`}
      />
      <text x={NODE_W / 2} y={26} textAnchor="middle" className="flow-node-label">
        {n.label}
      </text>
      <text x={NODE_W / 2} y={45} textAnchor="middle" className="flow-node-sub">
        {n.sub}
      </text>
    </g>
  )
}

function Arrow({ from, to, active, dashed, label, curveUp }) {
  const a = NODES[from]
  const b = NODES[to]
  const startX = a.x + NODE_W
  const startY = a.y + NODE_H / 2
  const endX = b.x
  const endY = b.y + NODE_H / 2
  const midY = curveUp ? Math.min(startY, endY) - 55 : (startY + endY) / 2
  const path = `M ${startX} ${startY} Q ${(startX + endX) / 2} ${midY} ${endX} ${endY}`
  return (
    <g>
      <path
        d={path}
        className={`flow-arrow${active ? ' flow-arrow-active' : ''}${dashed ? ' flow-arrow-dashed' : ''}`}
        markerEnd="url(#arrowhead)"
      />
      {label && (
        <text
          x={(startX + endX) / 2}
          y={midY - 6}
          textAnchor="middle"
          className={`flow-arrow-label${active ? ' flow-arrow-label-active' : ''}`}
        >
          {label}
        </text>
      )}
    </g>
  )
}

export default function FlowDiagram({ activeNode }) {
  // Brief pulse animation whenever activeNode changes, so even a fast
  // event (e.g. a QA pass) is visually noticeable rather than an instant
  // color-swap someone might miss while narrating a demo.
  const [pulseKey, setPulseKey] = useState(0)
  useEffect(() => {
    setPulseKey((k) => k + 1)
  }, [activeNode])

  const retryActive = activeNode === 'retry-edge'

  return (
    <div className="flow-diagram-wrap">
      <svg viewBox="0 0 680 300" className="flow-diagram">
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" className="flow-arrowhead" />
          </marker>
        </defs>

        <Arrow from="planner" to="developer" active={activeNode === 'developer'} />
        <Arrow from="developer" to="qa" active={activeNode === 'qa'} />
        <Arrow
          from="qa" to="developer" active={retryActive}
          dashed curveUp label="retry (≤3)"
        />
        <Arrow
          from="qa" to="escalate" active={activeNode === 'escalate'}
          dashed label="retries exhausted"
        />

        <Node id="planner" active={activeNode === 'planner'} pulse={activeNode === 'planner'} key={`p-${pulseKey}`} />
        <Node id="developer" active={activeNode === 'developer' || retryActive} pulse={activeNode === 'developer'} key={`d-${pulseKey}`} />
        <Node id="qa" active={activeNode === 'qa' || retryActive} pulse={activeNode === 'qa'} key={`q-${pulseKey}`} />
        <Node id="escalate" active={activeNode === 'escalate'} pulse={activeNode === 'escalate'} key={`e-${pulseKey}`} />
      </svg>
    </div>
  )
}
