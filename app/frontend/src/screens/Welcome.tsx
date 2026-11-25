import { useState } from 'react'
import { AXIS_NOTES, axisColor } from '../api'

const SCALES: [string, string, string][] = [
  ['warmth', 'Neutral', 'Warm'], ['directiveness', 'Collaborative', 'Directive'],
  ['formality', 'Formal', 'Casual'], ['depth', 'Just the advice', 'Explains why'],
]
// Measured: the GPU starts in under a minute, and each question takes as long as you take to read it.
const minutes = (rounds: number) => (rounds <= 5 ? 'a few minutes' : `about ${rounds} minutes`)

export function Welcome({ choices, error, onStart }: { choices: number[]; error?: string; onStart: (rounds: number) => void }) {
  const [rounds, setRounds] = useState(choices[0])
  const steps = [`Pick the best and worst of four answers, ${rounds} times.`, 'The model trains on each pick, live on a GPU.', 'Run your model on your own machine.']
  return (
    <div className="grid items-start gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
      <aside className="panel order-2 lg:order-1">
        <h2 className="panel-title border-b border-gray-200 px-4 py-3">What the model learns about you</h2>
        <ul className="divide-y divide-gray-100">
          {SCALES.map(([key, low, high]) => (
            <li key={key} className="px-4 py-3">
              <span className="flex items-center gap-2 font-medium text-gray-900"><span className="h-2 w-2 rounded-full" style={{ background: axisColor(key) }} aria-hidden />{AXIS_NOTES[key].title}</span>
              <span className="hint mt-0.5 block pl-4">{low} ↔ {high}</span>
            </li>
          ))}
        </ul>
      </aside>

      <section className="panel order-1 p-5 lg:order-2">
        <h1 className="text-lg font-semibold text-gray-900">Train your own personal AI model</h1>
        <p className="mt-2 max-w-2xl text-gray-600">
          Most AI models are tuned for the average person. This app trains one that knows how you like to be spoken to,
          and you can download it at the end.
        </p>
        <ol className="mt-4 space-y-2">
          {steps.map((text, i) => (
            <li key={i} className="flex items-center gap-3">
              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-gray-100 text-xs font-medium text-gray-600">{i + 1}</span>
              <span>{text}</span>
            </li>
          ))}
        </ol>

        <div className="mt-5 border-t border-gray-100 pt-4">
          <fieldset>
            <legend className="hint">Session length</legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {choices.map((n, i) => (
                <label key={n} className={`cursor-pointer rounded-md border px-3 py-1.5 ${rounds === n ? 'border-teal-600 bg-teal-50 text-teal-900' : 'border-gray-200 text-gray-700 hover:border-gray-300'}`}>
                  <input type="radio" name="rounds" className="sr-only" checked={rounds === n} onChange={() => setRounds(n)} />
                  {i === 0 ? 'Standard' : 'Extended'} <span className="text-gray-500">· {n} questions</span>
                </label>
              ))}
            </div>
          </fieldset>
          {error && <p role="alert" className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-800">{error} Try again.</p>}
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button className="btn btn-primary" onClick={() => onStart(rounds)}>Start training</button>
            <span className="hint">Takes {minutes(rounds)}, including about a minute for the GPU to start.</span>
          </div>
        </div>
      </section>
    </div>
  )
}
