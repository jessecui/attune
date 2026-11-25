import { axisColor, capital, type Axis } from '../api'

// Leans range from under one to several dozen, so the vertical scale is compressed (log).
const squash = (v: number) => Math.sign(v) * Math.log10(1 + Math.abs(v))

/** How one axis moved question by question: a small chart, one per axis, sharing one vertical scale. */
export function Trajectory({ axis, limit }: { axis: Axis; limit: number }) {
  // Ratings of the writing (1 to 5, 3 in the middle) when there are any; else the model's style choices.
  const rated = !!axis.written?.length, points = rated ? axis.written! : axis.lean, mid = rated ? 3 : 0
  const W = 300, H = 120, L = 6, R = 6, T = 16, B = 22, n = points.length
  const x = (i: number) => L + ((W - L - R) * i) / Math.max(1, n - 1)
  const y = (v: number) => T + (H - T - B) * (1 - (rated ? (v - 1) / 4 : (squash(v) + limit) / (2 * limit)))
  const side = (v: number) => (v >= mid ? axis.first : axis.second)
  return (
    <figure>
      <figcaption className="flex items-center gap-2 font-medium text-gray-900"><span className="h-2 w-2 rounded-full" style={{ background: axisColor(axis.axis) }} aria-hidden />{capital(axis.axis)}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1 w-full" role="img"
        aria-label={`${axis.axis} by question: ends leaning ${side(points[n - 1])}`}>
        <line x1={L} x2={W - R} y1={y(mid)} y2={y(mid)} stroke="var(--color-gray-300)" strokeWidth={1} />
        <text x={L} y={9} className="fill-gray-500" fontSize={8}>↑ {axis.first}</text>
        <text x={L} y={H - 11} className="fill-gray-500" fontSize={8}>↓ {axis.second}</text>
        <text x={L} y={H - 1} className="fill-gray-500" fontSize={8}>start</text>
        <text x={W - R} y={H - 1} textAnchor="end" className="fill-gray-500" fontSize={8}>question {n - 1}</text>
        <polyline points={points.map((v, i) => `${x(i)},${y(v)}`).join(' ')} fill="none" stroke={axisColor(axis.axis)} strokeWidth={2} strokeLinejoin="round" />
        {points.map((v, i) => (
          <g key={i}>
            <circle cx={x(i)} cy={y(v)} r={3} fill={axisColor(axis.axis)} stroke="#fff" strokeWidth={1.5} />
            <circle cx={x(i)} cy={y(v)} r={12} fill="transparent">
              <title>{`${i ? `After question ${i}` : 'Standard model'}: ${rated ? `writing rated ${v.toFixed(1)} of 5 (5 = ${axis.first})` : v === 0 ? 'even' : `leans ${side(v)}`}`}</title>
            </circle>
          </g>
        ))}
      </svg>
    </figure>
  )
}
export const sharedLimit = (axes: Axis[]) => Math.max(1, ...axes.flatMap((a) => a.lean.map((v) => Math.abs(squash(v)))))
