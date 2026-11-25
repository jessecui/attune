import { AXIS_NOTES, axisColor, capital, read, type Axis } from '../api'

/** One axis of the profile as a two-ended scale. The darker middle is the zone where the
 *  model did not commit either way; the marker is where it ended the session. */
export function Scale({ axis }: { axis: Axis }) {
  const r = read(axis), note = AXIS_NOTES[axis.axis]
  const result = r.strength === 'none' ? 'No clear preference' : r.strength === 'leans' ? `Leans ${r.side}` : capital(r.side)
  // Keep the marker inside the track so "more than 95%" does not sit on the very edge.
  const left = Math.min(97, Math.max(3, r.position * 100))
  return (
    <div className="px-4 py-3.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <h3 className="flex items-center gap-2 font-medium text-gray-900"><span className="h-2 w-2 rounded-full" style={{ background: axisColor(axis.axis) }} aria-hidden />{note?.title ?? capital(axis.axis)}</h3>
        <p className={r.strength === 'none' ? 'text-gray-500' : 'font-medium text-gray-900'}>{result}</p>
      </div>
      <div className="relative mt-3 h-1.5 rounded-full bg-gray-100" role="img"
        aria-label={`${note?.title}: ${result}. ${r.detail}`}>
        <div className={`absolute inset-y-0 bg-gray-200 ${axis.written?.length ? 'left-[37.5%] w-1/4' : 'left-1/4 w-1/2'}`} />
        <div className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-gray-400" style={{ left: '50%' }} />
        <div className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white"
          style={{ left: `${left}%`, background: r.strength === 'none' ? 'var(--color-gray-400)' : axisColor(axis.axis) }} />
      </div>
      <div className="hint mt-1.5 flex justify-between"><span>{capital(axis.second)}</span><span>{capital(axis.first)}</span></div>
      <p className="hint mt-1">{r.detail}</p>
    </div>
  )
}
