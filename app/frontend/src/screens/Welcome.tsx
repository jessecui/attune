import { useState, type FormEvent } from 'react'
import { api, AXIS_NOTES, axisColor, type Draft } from '../api'
import { Spinner } from '../components/Spinner'

const SCALES: [string, string, string][] = [
  ['warmth', 'Neutral', 'Warm'], ['directiveness', 'Collaborative', 'Directive'],
  ['formality', 'Formal', 'Casual'], ['depth', 'Just the advice', 'Explains why'],
]
const EXAMPLES = ['My doctor, explaining test results', 'My tutor, helping with exam revision', 'My manager, giving me feedback', 'A financial adviser, on saving']
// Measured: the GPU starts in under a minute, and each question takes as long as you take to read it.
const minutes = (rounds: number) => (rounds <= 5 ? 'a few minutes' : `about ${rounds} minutes`)

/** The start screen: say who the model should speak as, and the app drafts questions for that setting. */
export function Welcome({ choices, error, onDraft, onStart, onExpired }:
  { choices: number[]; error?: string; onDraft: (rounds: number, draft: Draft) => void; onStart: (rounds: number) => void; onExpired: (e: unknown) => boolean }) {
  const [rounds, setRounds] = useState(choices[0]), [setting, setSetting] = useState('')
  const [busy, setBusy] = useState(false), [problem, setProblem] = useState(error ?? '')

  const prepare = async (e?: FormEvent) => {
    e?.preventDefault()
    if (setting.trim().length < 3) return
    setBusy(true); setProblem('')
    try {
      const draft = await api.prepare(setting, rounds)
      if (draft.suitable) onDraft(rounds, draft)
      else setProblem(draft.reason || 'That setting does not suit this exercise. Try another.')
    } catch (err) { if (!onExpired(err)) setProblem((err as Error).message) } finally { setBusy(false) }
  }

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
          and you can download it at the end. It takes {rounds} questions and {minutes(rounds)}.
        </p>
        <ol className="mt-4 space-y-2">
          {[
            ['Choose a setting', `Say who the model should be, such as your doctor or your manager. The app writes ${rounds} questions you might ask them.`],
            ['Pick the best and worst', 'Each question gets four answers in different styles. The model trains on every pick.'],
            ['See what changed', 'Compare your model with the standard one, then download it.'],
          ].map(([title, text], i) => (
            <li key={title} className="flex gap-3">
              <span className="mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full bg-gray-100 text-xs font-medium text-gray-600">{i + 1}</span>
              <span><span className="font-medium text-gray-900">{title}.</span> <span className="text-gray-600">{text}</span></span>
            </li>
          ))}
        </ol>

        <form onSubmit={prepare} className="mt-5 border-t border-gray-100 pt-4">
          <label htmlFor="setting" className="font-medium text-gray-900">To start: who should the model be when it speaks to you, and about what?</label>
          <div className="mt-2 flex gap-2">
            <input id="setting" value={setting} onChange={(e) => setSetting(e.target.value)} maxLength={200} disabled={busy}
              placeholder="e.g. My doctor, explaining test results" autoComplete="off"
              className="min-w-0 flex-1 rounded-md border border-gray-300 px-3 py-2 focus:border-teal-600" />
            <button className="btn btn-primary shrink-0" disabled={busy || setting.trim().length < 3}>
              {busy ? <><Spinner />Writing questions</> : 'Write my questions'}
            </button>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="hint">Examples</span>
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" disabled={busy} onClick={() => setSetting(ex)}
                className="rounded-md border border-gray-200 bg-white px-2.5 py-1 text-xs text-gray-700 hover:border-gray-300 hover:bg-gray-50">{ex}</button>
            ))}
          </div>
          {problem && <p role="alert" className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-800">{problem}</p>}

          {choices.length > 1 && (
          <fieldset className="mt-5">
            <legend className="hint">Session length</legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {choices.map((n, i) => (
                <label key={n} className={`cursor-pointer rounded-md border px-3 py-1.5 ${rounds === n ? 'border-teal-600 bg-teal-50 text-teal-900' : 'border-gray-200 text-gray-700 hover:border-gray-300'}`}>
                  <input type="radio" name="rounds" className="sr-only" checked={rounds === n} onChange={() => setRounds(n)} disabled={busy} />
                  {i === 0 ? 'Standard' : 'Extended'} <span className="text-gray-500">· {n} questions</span>
                </label>
              ))}
            </div>
          </fieldset>
          )}
          <p className="hint mt-4">
            Not sure what to type?{' '}
            <button type="button" className="underline underline-offset-2 hover:text-gray-900" disabled={busy} onClick={() => onStart(rounds)}>
              Use general everyday questions instead
            </button>
          </p>
        </form>
      </section>
    </div>
  )
}
