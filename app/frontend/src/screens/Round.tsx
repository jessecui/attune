import { useState } from 'react'
import { api, type Round, type Summary } from '../api'
import { Spinner } from '../components/Spinner'

export function RoundScreen({ round, onNext, onExpired }:
  { round: Round; onNext: (next: Round | Summary) => void; onExpired: (e: unknown) => boolean }) {
  const [best, setBest] = useState<number | null>(null), [worst, setWorst] = useState<number | null>(null)
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const last = round.round === round.rounds

  // An answer cannot be both; choosing one clears the other mark from that card.
  const mark = (kind: 'best' | 'worst', i: number) => {
    if (kind === 'best') { setBest(best === i ? null : i); if (worst === i) setWorst(null) }
    else { setWorst(worst === i ? null : i); if (best === i) setBest(null) }
  }
  const submit = async () => {
    if (best === null || worst === null) return
    setBusy(true); setError('')
    try { onNext(await api.feedback(round.session, best, worst)) }
    catch (e) { if (!onExpired(e)) { setError((e as Error).message); setBusy(false) } }
  }
  const ready = best !== null && worst !== null
  const message = error ? error
    : busy ? (last ? 'Training on your last pick and building your results, about half a minute' : 'Training on your pick and fetching the next question')
    : best === null && worst === null ? 'Mark one answer as best and one as worst'
    : best === null ? 'Now mark the best' : worst === null ? 'Now mark the worst' : 'Both marked'
  // The pill's colour says where things stand: waiting on a pick, ready, working, or failed.
  const tone = error ? 'bg-red-50 text-red-800' : busy ? 'bg-teal-50 text-teal-900' : ready ? 'bg-teal-50 text-teal-900' : 'bg-gray-100 text-gray-700'
  const dot = error ? 'bg-red-600' : ready ? 'bg-teal-600' : 'bg-gray-400'

  return (
    <div className="mx-auto max-w-5xl">
      <div className="panel p-5">
        <p className="hint">
          Question {round.round} of {round.rounds}
          {round.role && !round.builtin && <> · Answered by <span className="font-medium text-gray-700">{round.role}</span></>}
        </p>
        <h1 className="mt-1 text-lg font-semibold text-gray-900">{round.question}</h1>
        <p className="mt-1 text-gray-600">Pick the answer you like best and the one you like worst.</p>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {round.candidates.map((text, i) => {
          const state = best === i ? 'best' : worst === i ? 'worst' : null
          return (
            <article key={i} className={`panel flex flex-col p-4 transition-opacity ${busy ? 'opacity-60' : ''} ${state === 'best' ? 'border-teal-600 ring-1 ring-teal-600' : state === 'worst' ? 'border-gray-500 ring-1 ring-gray-500' : ''}`}>
              <div className="flex items-center justify-between">
                <span className="hint font-medium">{'ABCD'[i]}</span>
                {state && <span className={`text-xs font-medium ${state === 'best' ? 'text-teal-800' : 'text-gray-600'}`}>{state === 'best' ? 'Your best' : 'Your worst'}</span>}
              </div>
              <p className="mt-2 flex-1 whitespace-pre-line text-[15px] leading-relaxed text-gray-900">{text}</p>
              <div className="mt-4 flex gap-2">
                <button aria-pressed={best === i} disabled={busy} onClick={() => mark('best', i)}
                  className={`btn flex-1 py-1.5 ${best === i ? 'bg-teal-700 text-white' : 'btn-quiet'}`}>Best</button>
                <button aria-pressed={worst === i} disabled={busy} onClick={() => mark('worst', i)}
                  className={`btn flex-1 py-1.5 ${worst === i ? 'bg-slate-600 text-white' : 'btn-quiet'}`}>Worst</button>
              </div>
            </article>
          )
        })}
      </div>

      {/* The action bar: what has been picked, what is needed next, and the way forward. */}
      <div className="panel mt-4 flex flex-wrap items-center gap-3 px-4 py-3">
        <div className="flex flex-wrap items-center gap-2" role="status" aria-live="polite">
          <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 font-medium ${tone}`}>
            {busy ? <Spinner className="h-3.5 w-3.5" /> : <span className={`h-1.5 w-1.5 rounded-full ${dot}`} aria-hidden />}
            {message}
          </span>
          {!busy && !error && (
            <span className="hint">Best: <b className="font-medium text-gray-700">{best === null ? 'none' : 'ABCD'[best]}</b> · Worst: <b className="font-medium text-gray-700">{worst === null ? 'none' : 'ABCD'[worst]}</b></span>
          )}
        </div>
        <button className="btn btn-primary ml-auto" disabled={busy || best === null || worst === null} onClick={submit}>
          {last ? 'See results' : 'Next question'}
        </button>
      </div>
    </div>
  )
}
