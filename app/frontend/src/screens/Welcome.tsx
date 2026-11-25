import { useState } from 'react'
import { AXIS_NOTES } from '../api'

const SCALES: [string, string, string][] = [
  ['warmth', 'Neutral', 'Warm'], ['directiveness', 'Collaborative', 'Directive'],
  ['formality', 'Formal', 'Casual'], ['depth', 'Just the advice', 'Explains why'],
]

const WHY: [string, string][] = [
  ['It answers in your style', 'Assistants are tuned for an average reader. This one learns whether you want warmth, instructions, formality and reasons, and writes that way without being told each time.'],
  ['It is yours', 'You download the weights. They run on your own machine with an open base model: no account, no subscription, and nothing you type leaves your computer.'],
  ['You see how it works', 'Training on human feedback is how labs shape their assistants. Here it runs live on your picks, and you get the round-by-round record of what changed.'],
]

export function Welcome({ choices, error, onStart }: { choices: number[]; error?: string; onStart: (rounds: number) => void }) {
  const [rounds, setRounds] = useState(choices[0])
  return (
    <>
    <div className="grid gap-10 lg:grid-cols-[1.15fr_1fr] lg:gap-14">
      <section>
        <p className="eyebrow">Live fine-tuning · about {rounds} minutes</p>
        <h1 className="mt-3 text-4xl font-semibold leading-tight tracking-tight">Train your own AI model.</h1>
        <p className="mt-4 text-lg text-ink-2">
          Read four answers to a question and pick the best and the worst. Each pick fine-tunes a language model
          to your style. After {rounds} questions, the model is yours to download.
        </p>
        <ol className="mt-8 space-y-4">
          {[['You pick', `${rounds} questions, four answers each. Mark the best and the least.`],
            ['The model trains', 'Every pick updates an open 7-billion-parameter model on a dedicated GPU, between questions.'],
            ['You download it', 'The tuned model, a profile of the style it learned, and a script to run it.']].map(([t, d], i) => (
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
          <button className="btn btn-primary px-6 py-3 text-base" onClick={() => onStart(rounds)}>Start training</button>
          <span className="text-sm text-muted">The GPU takes a minute or two to start.</span>
        </div>
      </section>

      <aside className="sheet self-start p-6">
        <p className="eyebrow">What the model learns</p>
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
          Four scales of communication style, drawn from research on interpersonal communication. The result reflects
          this session's picks and can differ next time.
        </p>
      </aside>
    </div>

    <section className="mt-12 border-t border-rule pt-8" aria-labelledby="why">
      <h2 id="why" className="eyebrow">Why train your own</h2>
      <div className="mt-5 grid gap-8 md:grid-cols-3">
        {WHY.map(([title, text]) => (
          <div key={title}>
            <h3 className="font-medium">{title}</h3>
            <p className="mt-1.5 text-sm text-ink-2">{text}</p>
          </div>
        ))}
      </div>
    </section>
    </>
  )
}
