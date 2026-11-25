import { useState } from 'react'
import { api, type Round, type Summary } from '../api'

export function RoundScreen({ round, onNext, onExpired }:
  { round: Round; onNext: (next: Round | Summary) => void; onExpired: (e: unknown) => boolean }) {
  const [best, setBest] = useState<number | null>(null), [least, setLeast] = useState<number | null>(null)
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const last = round.round === round.rounds

  // An answer cannot be both; choosing one clears the other mark from that card.
  const mark = (kind: 'best' | 'least', i: number) => {
    if (kind === 'best') { setBest(best === i ? null : i); if (least === i) setLeast(null) }
    else { setLeast(least === i ? null : i); if (best === i) setBest(null) }
  }
  const submit = async () => {
    if (best === null || least === null) return
    setBusy(true); setError('')
    try { onNext(await api.feedback(round.session, best, least)) }
    catch (e) { if (!onExpired(e)) { setError((e as Error).message); setBusy(false) } }
  }
  const hint = best === null && least === null ? 'Mark one answer as best and one as least.'
    : best === null ? 'Now mark the one you like best.' : least === null ? 'Now mark the one you like least.' : 'Both marked.'

  return (
    <div>
      <p className="eyebrow">Question {round.round} of {round.rounds}</p>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{round.question}</h1>
      <p className="mt-2 text-ink-2">Four answers with the same advice, said in different ways. Which suits you?</p>

      <div className="mt-7 grid gap-4 md:grid-cols-2">
        {round.candidates.map((text, i) => {
          const state = best === i ? 'best' : least === i ? 'least' : null
          return (
            <article key={i} className={`sheet flex flex-col p-5 transition-colors ${state === 'best' ? 'border-accent ring-1 ring-accent' : state === 'least' ? 'border-least ring-1 ring-least' : ''}`}>
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-muted">{'ABCD'[i]}</span>
                {state && <span className={`eyebrow ${state === 'best' ? '!text-accent-ink' : '!text-least'}`}>{state === 'best' ? 'Your best' : 'Your least'}</span>}
              </div>
              <p className="mt-3 flex-1 whitespace-pre-line text-[1.02rem] leading-relaxed">{text}</p>
              <div className="mt-5 flex gap-2">
                <button aria-pressed={best === i} disabled={busy} onClick={() => mark('best', i)}
                  className={`btn flex-1 ${best === i ? 'bg-accent text-white' : 'btn-quiet'}`}>Best</button>
                <button aria-pressed={least === i} disabled={busy} onClick={() => mark('least', i)}
                  className={`btn flex-1 ${least === i ? 'bg-least text-white' : 'btn-quiet'}`}>Least</button>
              </div>
            </article>
          )
        })}
      </div>

      <div className="sticky bottom-0 -mx-5 mt-8 border-t border-rule bg-page/95 px-5 py-4 backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-ink-2" role="status" aria-live="polite">
            {error ? <span className="text-danger">{error}</span> : busy ? (last ? 'Applying your last choice and preparing your profile. This can take half a minute.' : 'Learning from your choice.') : hint}
          </p>
          <button className="btn btn-primary px-6" disabled={busy || best === null || least === null} onClick={submit}>
            {busy ? 'One moment' : last ? 'See my profile' : 'Next question'}
          </button>
        </div>
      </div>
    </div>
  )
}
