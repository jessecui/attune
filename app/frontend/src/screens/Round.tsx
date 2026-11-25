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
  const hint = best === null && worst === null ? 'Mark one answer as best and one as worst.'
    : best === null ? 'Now mark the best.' : worst === null ? 'Now mark the worst.' : 'Both marked.'

  return (
    <div className="mx-auto max-w-5xl">
      <div className="panel p-5">
        <p className="hint">Question {round.round} of {round.rounds}</p>
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
                  className={`btn flex-1 py-1.5 ${worst === i ? 'bg-gray-700 text-white' : 'btn-quiet'}`}>Worst</button>
              </div>
            </article>
          )
        })}
      </div>

      <div className={`mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border px-4 py-3 ${busy ? 'border-teal-200 bg-teal-50' : 'border-transparent'}`}>
        <div className="flex items-center gap-2.5 text-gray-600" role="status" aria-live="polite">
          {error ? <span className="text-red-700">{error}</span>
            : busy ? <><Spinner className="h-4 w-4 text-teal-700" /><span className="text-teal-900">{last ? 'Training on your last pick and building your results. About half a minute.' : 'Training on your pick and fetching the next question.'}</span></>
            : hint}
        </div>
        <button className="btn btn-primary" disabled={busy || best === null || worst === null} onClick={submit}>
          {last ? 'See results' : 'Next question'}
        </button>
      </div>
    </div>
  )
}
