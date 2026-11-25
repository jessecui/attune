import { useState } from 'react'
import { AXIS_NOTES, axisColor } from '../api'

const SCALES: [string, string, string][] = [
  ['warmth', 'Neutral', 'Warm'], ['directiveness', 'Collaborative', 'Directive'],
  ['formality', 'Formal', 'Casual'], ['depth', 'Just the advice', 'Explains why'],
]

export function Welcome({ choices, error, onStart }: { choices: number[]; error?: string; onStart: (rounds: number) => void }) {
  const [rounds, setRounds] = useState(choices[0])
  const steps = [`Pick the best and the least of four answers, ${rounds} times.`, 'The model trains on each pick, live on a GPU.', 'Download your model and run it yourself.']
  return (
    <div className="grid gap-14 pt-6 lg:grid-cols-[1.25fr_1fr] lg:gap-20">
      <section>
        <p className="eyebrow">Live fine-tuning · about {rounds} minutes</p>
        <h1 className="mt-4 text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">Train your own <span className="mark">personal AI model</span>.</h1>
        <p className="mt-6 max-w-xl text-lg text-ink-2">
          Most assistants are written for an average reader. This one learns how you like to be answered, and the weights are yours to keep.
        </p>

        <ol className="mt-10 space-y-4">
          {steps.map((text, i) => (
            <li key={i} className="flex items-center gap-4">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-accent-soft font-mono text-xs font-medium text-accent-ink">{i + 1}</span>
              <span>{text}</span>
            </li>
          ))}
        </ol>

        <fieldset className="mt-12">
          <legend className="text-sm font-medium">Session length</legend>
          <div className="mt-2 inline-flex rounded-lg border border-rule-strong bg-sheet p-1">
            {choices.map((n, i) => (
              <label key={n} className={`cursor-pointer rounded-md px-4 py-2 text-sm ${rounds === n ? 'bg-accent-soft text-accent-ink font-medium' : 'text-ink-2'}`}>
                <input type="radio" name="rounds" className="sr-only" checked={rounds === n} onChange={() => setRounds(n)} />
                {i === 0 ? 'Standard' : 'Extended'} <span className="font-mono text-xs ml-1">{n} questions</span>
              </label>
            ))}
          </div>
        </fieldset>
        {error && <p role="alert" className="mt-5 rounded-md border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger">{error} Try again.</p>}
        <div className="mt-8 flex flex-wrap items-center gap-4">
          <button className="btn btn-primary px-6 py-3 text-base" onClick={() => onStart(rounds)}>Start training</button>
          <span className="text-sm text-muted">The GPU takes a minute or two to start.</span>
        </div>
      </section>

      <aside className="sheet self-start p-8">
        <p className="eyebrow">What the model learns</p>
        <ul className="mt-6 space-y-6">
          {SCALES.map(([key, low, high]) => (
            <li key={key}>
              <span className="flex items-center gap-2 font-medium"><span className="h-2.5 w-2.5 rounded-full" style={{ background: axisColor(key) }} aria-hidden />{AXIS_NOTES[key].title}</span>
              <span className="mt-1 block pl-[1.125rem] font-mono text-xs text-muted">{low} ↔ {high}</span>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  )
}
