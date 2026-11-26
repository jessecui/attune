import { useState, type FormEvent } from 'react'
import { api, AXIS_NOTES, axisColor, type Draft } from '../api'
import { Spinner } from '../components/Spinner'

const SCALES = ['warmth', 'directiveness', 'formality', 'depth']
const EXAMPLES = ['My doctor, explaining test results', 'My tutor, helping with exam revision', 'My manager, giving me feedback', 'A financial adviser, on saving']

/** The start screen: one question to the visitor, with the rest kept to a line each. */
export function Welcome({ choices, error, onDraft, onStart, onExpired }:
  { choices: number[]; error?: string; onDraft: (rounds: number, draft: Draft) => void; onStart: (rounds: number) => void; onExpired: (e: unknown) => boolean }) {
  const rounds = choices[0]
  const [setting, setSetting] = useState(''), [busy, setBusy] = useState(false), [problem, setProblem] = useState(error ?? '')
  const steps = [['Choose a setting', `We write ${rounds} questions for it.`], ['Pick the best and worst', 'The model trains on each pick.'], ['Keep your model', 'See what changed, then download it.']]

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
    <div className="mx-auto max-w-2xl pt-8 sm:pt-14">
      <h1 className="text-center text-3xl font-semibold tracking-tight text-gray-900">Train your own personal AI model</h1>
      <p className="mx-auto mt-3 max-w-lg text-center text-base text-gray-600">
        Most AI models are tuned for the average person. Train one that knows how you like to be spoken to.
      </p>

      <form onSubmit={prepare} className="panel mt-8 p-5">
        <label htmlFor="setting" className="font-medium text-gray-900">Who should the model be when it speaks to you?</label>
        <div className="mt-2 flex gap-2">
          <input id="setting" value={setting} onChange={(e) => setSetting(e.target.value)} maxLength={200} disabled={busy}
            placeholder="e.g. My doctor, explaining test results" autoComplete="off"
            className="min-w-0 flex-1 rounded-md border border-gray-300 px-3 py-2.5 text-base focus:border-teal-600" />
          <button className="btn btn-primary shrink-0 px-4" disabled={busy || setting.trim().length < 3}>
            {busy ? <><Spinner />Writing questions</> : 'Start'}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" disabled={busy} onClick={() => setSetting(ex)}
              className="rounded-md border border-gray-200 bg-white px-2.5 py-1 text-xs text-gray-700 hover:border-gray-300 hover:bg-gray-50">{ex}</button>
          ))}
          <button type="button" disabled={busy} onClick={() => onStart(rounds)}
            className="rounded-md px-2.5 py-1 text-xs text-gray-500 underline underline-offset-2 hover:text-gray-900">or general questions</button>
        </div>
        {problem && <p role="alert" className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-red-800">{problem}</p>}
      </form>

      <ol className="mt-10 grid gap-6 sm:grid-cols-3">
        {steps.map(([title, text], i) => (
          <li key={title}>
            <span className="grid h-6 w-6 place-items-center rounded-full bg-teal-50 text-xs font-semibold text-teal-800">{i + 1}</span>
            <p className="mt-2 font-medium text-gray-900">{title}</p>
            <p className="mt-0.5 text-gray-600">{text}</p>
          </li>
        ))}
      </ol>

      <div className="mt-10 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 border-t border-gray-200 pt-5 text-gray-600">
        <span className="hint">It learns your taste on</span>
        {SCALES.map((key) => (
          <span key={key} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: axisColor(key) }} aria-hidden />{AXIS_NOTES[key].title}</span>
        ))}
        <span className="hint">· about five minutes</span>
      </div>
    </div>
  )
}
