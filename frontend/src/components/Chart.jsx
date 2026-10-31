import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

/** Thin, dark-themed wrappers around recharts so pages describe WHAT to plot
 * (data + series) and never touch chart styling. Reused by the metrics
 * dashboard now and by the batch/experiment views later.
 */
const AXIS = '#9ca3af'
const GRID = '#262b36'
export const PALETTE = ['#6366f1', '#34d399', '#f59e0b', '#38bdf8', '#fb7185', '#a78bfa', '#9ca3af']
const tooltipStyle = {
  contentStyle: { background: '#171a21', border: '1px solid #262b36', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#e5e7eb' },
  itemStyle: { color: '#e5e7eb' },
}

function Card({ title, subtitle, height = 240, children, empty }) {
  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>
      {subtitle && <div className="chart-sub">{subtitle}</div>}
      <div style={{ height }}>
        {empty ? <div className="chart-empty">No data yet</div> : children}
      </div>
    </div>
  )
}

/** series: [{ key, label, color }]; stacked bars share a stackId. */
export function BarCard({ title, subtitle, data, xKey, series, stacked = false, height, yFormatter }) {
  return (
    <Card title={title} subtitle={subtitle} height={height} empty={!data?.length}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={xKey} stroke={AXIS} fontSize={11} interval={0} tickLine={false} />
          <YAxis stroke={AXIS} fontSize={11} tickFormatter={yFormatter} tickLine={false} width={48} />
          <Tooltip {...tooltipStyle} cursor={{ fill: 'rgba(99,102,241,0.08)' }} />
          {series.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {series.map((s, i) => (
            <Bar key={s.key} dataKey={s.key} name={s.label || s.key} fill={s.color || PALETTE[i % PALETTE.length]}
                 stackId={stacked ? 'a' : undefined} radius={stacked ? 0 : [3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </Card>
  )
}

export function LineCard({ title, subtitle, data, xKey, series, height, yFormatter }) {
  return (
    <Card title={title} subtitle={subtitle} height={height} empty={!data?.length}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey={xKey} stroke={AXIS} fontSize={11} tickLine={false} />
          <YAxis stroke={AXIS} fontSize={11} tickFormatter={yFormatter} tickLine={false} width={48} />
          <Tooltip {...tooltipStyle} />
          {series.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
          {series.map((s, i) => (
            <Line key={s.key} type="monotone" dataKey={s.key} name={s.label || s.key} dot={{ r: 3 }}
                  stroke={s.color || PALETTE[i % PALETTE.length]} strokeWidth={2} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

/** data: [{ name, value }] */
export function PieCard({ title, subtitle, data, height, valueFormatter }) {
  const clean = (data || []).filter((d) => d.value > 0)
  return (
    <Card title={title} subtitle={subtitle} height={height} empty={!clean.length}>
      <ResponsiveContainer>
        <PieChart>
          <Pie data={clean} dataKey="value" nameKey="name" innerRadius="48%" outerRadius="78%" paddingAngle={2} stroke="none">
            {clean.map((d, i) => <Cell key={d.name} fill={d.color || PALETTE[i % PALETTE.length]} />)}
          </Pie>
          <Tooltip {...tooltipStyle} formatter={valueFormatter} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </Card>
  )
}
