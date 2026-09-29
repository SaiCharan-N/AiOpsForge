import { useEffect, useState } from 'react'

/* Phase 12: SVG rendering of the pipeline topology from
 * backend/app/pipeline/graph.py — Planner -> Developer -> QA, with QA
 * failures routed through a Failure Analyzer node before a targeted repair
 * goes back to Developer, and the escalate branch on exhausted retries.
 *
 * `activeNode` is one of 'planner' | 'developer' | 'qa' | 'failure_analyzer'
 * | 'escalate' | 'retry-edge' | null, driven by the latest WebSocket event —
 * see EVENT_TO_NODE in constants/eventTypes.js.
 *
 * Previously this diagram only had 4 nodes and drew QA's retry straight
 * back to Developer — that was the pre-Phase-11 graph shape. Once
 * failure_analyzer_node was added (QA fail -> Failure Analyzer -> targeted
 * repair -> Developer), this diagram was never updated to match, so it was
 * silently showing an incorrect flow: no Failure Analyzer node existed at
 * all, and the 'developer_retry' event it reused during analysis lit up
 * the Developer node instead. Fixed by adding the node and splitting the
 * retry edge into its two real hops (QA -> Failure Analyzer -> Developer).
 *
 * Deliberately hand-rolled rather than a charting/flow library: the graph
 * is fixed and small (5 nodes), so a real dependency would be overkill for
 * what's ultimately five boxes and some arrows.
 */

const NODES = {
  planner:          { x: 40,  y: 70,  label: 'Planner',          sub: 'breaks request into tasks' },
  developer:        { x: 260, y: 70,  label: 'Developer',         sub: 'writes code via MCP' },
  qa:               { x: 480, y: 70,  label: 'QA',                sub: 'deterministic checks + tests' },
  failure_analyzer: { x: 260, y: 220, label: 'Failure Analyzer',  sub: 'diagnoses root cause' },
  escalate:         { x: 480, y: 220, label: 'Escalate',          sub: 'needs human review' },
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

/** Horizontal-flow arrow: right edge of `from` to left edge of `to`, with an
 * optional upward bulge. Used for planner->developer->qa and qa->escalate
 * (same shape those already used before this fix). */
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

/** Vertical-flow arrow: bottom-center of `from` to top-center of `to` (or
 * top-to-bottom when `reverse` is set, for the analyzer->developer hop
 * that goes back UP). Used for the two new QA<->Failure Analyzer edges,
 * which sit in different rows rather than side by side. */
function VerticalArrow({ from, to, active, dashed, label, reverse, xOffset = 0 }) {
  const a = NODES[from]
  const b = NODES[to]
  const startX = a.x + NODE_W / 2 + xOffset
  const endX = b.x + NODE_W / 2 + xOffset
  const startY = reverse ? a.y : a.y + NODE_H
  const endY = reverse ? b.y + NODE_H : b.y
  const midX = (startX + endX) / 2
  const midY = (startY + endY) / 2
  const path = `M ${startX} ${startY} Q ${midX + (reverse ? -50 : 50)} ${midY} ${endX} ${endY}`
  return (
    <g>
      <path
        d={path}
        className={`flow-arrow${active ? ' flow-arrow-active' : ''}${dashed ? ' flow-arrow-dashed' : ''}`}
        markerEnd="url(#arrowhead)"
      />
      {label && (
        <text
          x={midX + (reverse ? -58 : 58)}
          y={midY}
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

  const analyzing = activeNode === 'failure_analyzer'
  const handoffToDeveloper = activeNode === 'retry-edge'
  // Developer only lights when actively developing, not during analysis or handoff.
  const developerActive = activeNode === 'developer'

  return (
    <div className="flow-diagram-wrap">
      <svg viewBox="0 0 680 320" className="flow-diagram">
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" className="flow-arrowhead" />
          </marker>
        </defs>

        <Arrow from="planner" to="developer" active={activeNode === 'developer'} />
        <Arrow from="developer" to="qa" active={activeNode === 'qa'} />
        <Arrow
          from="qa" to="escalate" active={activeNode === 'escalate'}
          dashed label="retries exhausted"
        />

        <VerticalArrow
          from="qa" to="failure_analyzer" active={analyzing}
          dashed label="on failure" xOffset={-30}
        />
        <VerticalArrow
          from="failure_analyzer" to="developer" active={handoffToDeveloper}
          dashed reverse label="repair (≤3)" xOffset={30}
        />

        <Node id="planner" active={activeNode === 'planner'} pulse={activeNode === 'planner'} key={`p-${pulseKey}`} />
        <Node id="developer" active={developerActive} pulse={activeNode === 'developer'} key={`d-${pulseKey}`} />
        <Node id="qa" active={activeNode === 'qa'} pulse={activeNode === 'qa'} key={`q-${pulseKey}`} />
        <Node id="failure_analyzer" active={analyzing} pulse={analyzing} key={`fa-${pulseKey}`} />
        <Node id="escalate" active={activeNode === 'escalate'} pulse={activeNode === 'escalate'} key={`e-${pulseKey}`} />
      </svg>
    </div>
  )
}
