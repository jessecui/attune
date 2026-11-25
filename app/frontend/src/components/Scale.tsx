import { AXIS_NOTES, axisColor, capital, read, type Axis } from '../api'

/** One axis of the profile as a two-ended scale. The shaded middle is the zone where the
 *  model did not commit either way; the marker is where it ended the session. */
export function Scale({ axis }: { axis: Axis }) {
  const r = read(axis), note = AXIS_NOTES[axis.axis]
  const result = r.strength === 'none' ? 'No clear preference' : r.strength === 'leans' ? `Leans ${r.side}` : capital(r.side)
  // Keep the marker inside the track so "over 95%" does not sit on the very edge.
  const left = Math.min(97, Math.max(3, r.position * 100))
  const share = r.share > 0.95 ? 'over 95%' : `${Math.floor(r.share * 100)}%`
  return (
    <div className="py-5 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="flex items-center gap-2 font-medium"><span className="h-2.5 w-2.5 rounded-full" style={{ background: axisColor(axis.axis) }} aria-hidden />{note?.title ?? capital(axis.axis)}</h3>
        <p className={`font-medium ${r.strength === 'none' ? 'text-muted' : 'text-ink'}`}>{result}</p>
      </div>
      <div className="relative mt-4 h-2 rounded-full bg-rule" role="img"
        aria-label={`${note?.title}: ${result}. The model ended ${share} toward ${r.side}.`}>
        <div className="absolute inset-y-0 left-1/4 w-1/2 bg-rule-strong/60" />
        <div className="absolute top-1/2 h-4 w-px -translate-y-1/2 bg-muted" style={{ left: '50%' }} />
        <div className="absolute top-1/2 h-[18px] w-[18px] -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-sheet shadow"
          style={{ left: `${left}%`, background: r.strength === 'none' ? 'var(--color-muted)' : axisColor(axis.axis) }} />
      </div>
      <div className="mt-2 flex justify-between font-mono text-xs text-muted">
        <span>{capital(axis.second)}</span><span>{capital(axis.first)}</span>
      </div>
      <p className="mt-2 text-sm text-ink-2">{note?.about} <span className="text-muted">The model ended {share} toward {r.side}.</span></p>
    </div>
  )
}
