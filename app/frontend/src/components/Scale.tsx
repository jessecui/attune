import { AXIS_NOTES, axisColor, capital, read, shown, type Axis } from '../api'

/** One axis of the profile as a two-ended scale. The result is shown on the scale itself: the end
 *  the model landed on is in bold, so no separate label sits above the opposite end. */
export function Scale({ axis }: { axis: Axis }) {
  const r = read(axis), note = AXIS_NOTES[axis.axis]
  const result = r.strength === 'none' ? 'No clear preference' : r.toward ? `Moved toward ${shown(r.other)}` : r.strength === 'leans' ? `Leans ${shown(r.side)}` : capital(shown(r.side))
  // Keep the marker inside the track so an extreme result does not sit on the very edge.
  const left = Math.min(97, Math.max(3, r.position * 100))
  const end = (name: string) => {
    // Bold marks what the training taught. A side the standard model already sat on, further out, is not that.
    const chosen = r.strength !== 'none' && !r.toward && r.side === name
    return <span className={chosen ? 'font-semibold text-gray-900' : ''}>{chosen && r.strength === 'leans' ? `Leans ${shown(name)}` : capital(shown(name))}</span>
  }
  return (
    <div className="px-4 py-3.5">
      <h3 className="flex items-center gap-2 font-medium text-gray-900"><span className="h-2 w-2 rounded-full" style={{ background: axisColor(axis.axis) }} aria-hidden />{note?.title ?? capital(axis.axis)}</h3>
      <div className="relative mt-3 h-1.5 rounded-full bg-gray-100" role="img" aria-label={`${note?.title}: ${result}. ${r.detail}`}>
        <div className={`absolute inset-y-0 bg-gray-200 ${axis.written?.length ? 'left-[37.5%] w-1/4' : 'left-1/4 w-1/2'}`} />
        <div className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-gray-400" style={{ left: '50%' }} />
        {/* where the standard model's writing was rated, for comparison */}
        {axis.written?.length ? <div className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-gray-400 bg-surface"
          style={{ left: `${Math.min(97, Math.max(3, ((axis.written[0] - 1) / 4) * 100))}%` }} /> : null}
        <div className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface"
          style={{ left: `${left}%`, background: r.strength === 'none' ? 'var(--color-gray-400)' : axisColor(axis.axis) }} />
      </div>
      <div className="hint mt-1.5 flex justify-between">{end(axis.second)}{end(axis.first)}</div>
      <p className="hint mt-1">{r.strength === 'none' ? 'No clear preference. ' : ''}{r.detail}</p>
    </div>
  )
}
