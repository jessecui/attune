import { useState } from 'react'
import { AXIS_NOTES } from '../api'

const SCALES: [string, string, string][] = [
  ['warmth', 'Neutral', 'Warm'], ['directiveness', 'Collaborative', 'Directive'],
  ['formality', 'Formal', 'Casual'], ['depth', 'Just the advice', 'Explains why'],
]

export function Welcome({ choices, error, onStart }: { choices: number[]; error?: string; onStart: (rounds: number) => void }) {
  const [rounds, setRounds] = useState(choices[0])
  return (
    <div className="grid gap-10 lg:grid-cols-[1.15fr_1fr] lg:gap-14">
      <section>
        <p className="eyebrow">Assessment · about {rounds} minutes</p>
        <h1 className="mt-3 text-4xl font-semibold leading-tight tracking-tight">Find out how you like to be spoken to.</h1>
        <p className="mt-4 text-lg text-ink-2">
          You will read four short answers to an everyday question, then mark the one you like best and the one you like least.
          With every choice, a language model adjusts itself to you.
        </p>
        <ol className="mt-8 space-y-4">
          {[['Read and choose', `${rounds} questions, four answers each. There are no right answers.`],
            ['See your profile', 'Where you sit on four scales of communication style, and how sure the result is.'],
            ['Take your model with you', 'Download the model that was tuned to your choices and run it yourself.']].map(([t, d], i) => (
            <li key={t} className="flex gap-4">
              <span className="font-mono text-sm text-accent mt-0.5 w-5 shrink-0">{String(i + 1).padStart(2, '0')}</span>
              <span><span className="font-medium">{t}.</span> <span className="text-ink-2">{d}</span></span>
            </li>
          ))}
        </ol>

        <fieldset className="mt-9">
          <legend className="text-sm font-medium">Session length</legend>
          <div className="mt-2 inline-flex rounded-md border border-rule-strong bg-sheet p-1">
            {choices.map((n, i) => (
              <label key={n} className={`cursor-pointer rounded px-4 py-2 text-sm ${rounds === n ? 'bg-accent-soft text-accent-ink font-medium' : 'text-ink-2'}`}>
                <input type="radio" name="rounds" className="sr-only" checked={rounds === n} onChange={() => setRounds(n)} />
                {i === 0 ? 'Standard' : 'Extended'} <span className="font-mono text-xs ml-1">{n} questions</span>
              </label>
            ))}
          </div>
        </fieldset>
        {error && <p role="alert" className="mt-5 rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">{error} Try again.</p>}
        <div className="mt-6 flex flex-wrap items-center gap-4">
          <button className="btn btn-primary px-6 py-3 text-base" onClick={() => onStart(rounds)}>Begin the assessment</button>
          <span className="text-sm text-muted">Starting takes a minute or two.</span>
        </div>
      </section>

      <aside className="sheet self-start p-6">
        <p className="eyebrow">What it measures</p>
        <ul className="mt-4 divide-y divide-rule">
          {SCALES.map(([key, low, high]) => (
            <li key={key} className="py-4 first:pt-0 last:pb-0">
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-medium">{AXIS_NOTES[key].title}</span>
                <span className="font-mono text-xs text-muted">{low} ↔ {high}</span>
              </div>
              <p className="mt-1 text-sm text-ink-2">{AXIS_NOTES[key].about}</p>
            </li>
          ))}
        </ul>
        <p className="mt-5 border-t border-rule pt-4 text-sm text-muted">
          The scales are informed by research on interpersonal communication. Your result describes the choices you make in this
          session. It is not a diagnosis, and it may differ next time.
        </p>
      </aside>
    </div>
  )
}
