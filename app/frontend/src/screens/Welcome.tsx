import { useState, type FormEvent } from 'react'
import { api, AXIS_NOTES, axisColor, type Draft } from '../api'
import { Spinner } from '../components/Spinner'

const SCALES = ['warmth', 'directiveness', 'formality', 'depth']
const EXAMPLES = ['My doctor explaining my test results', 'My tutor helping me study for a test', 'My manager giving me feedback', 'A financial adviser helping me save']

/** The start screen: one question to the visitor, with the rest kept to a line each. */
export function Welcome({ choices, error, onDraft, onStart, onExpired }:
  { choices: number[]; error?: string; onDraft: (rounds: number, draft: Draft) => void; onStart: (rounds: number) => void; onExpired: (e: unknown) => boolean }) {
  const rounds = choices[0]
  const [setting, setSetting] = useState(''), [busy, setBusy] = useState(false), [problem, setProblem] = useState(error ?? '')

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
    <div className="mx-auto max-w-2xl pt-10 text-center sm:pt-20">
      <h1 className="text-4xl font-semibold tracking-tight text-gray-900">Train your own personal AI model</h1>
      <p className="mx-auto mt-4 max-w-lg text-lg leading-relaxed text-gray-500">
        Most AI models are tuned for the average person. Train one that knows how you like to be spoken to.
        It takes a few minutes, and you can download your model at the end.
      </p>

      <form onSubmit={prepare} className="mt-10">
        <label htmlFor="setting" className="font-medium text-gray-900">Who should the model be when it speaks to you?</label>
        {/* One field with its button inside it */}
        <div className="relative mt-3">
          <input id="setting" value={setting} onChange={(e) => setSetting(e.target.value)} maxLength={200} disabled={busy}
            placeholder="My doctor explaining my test results" autoComplete="off"
            className="w-full rounded-xl border border-gray-200 bg-white py-4 pl-5 pr-32 text-base shadow-sm transition placeholder:text-gray-400 focus:border-teal-600 focus:outline-none focus:ring-4 focus:ring-teal-600/10" />
          <button className="btn btn-primary absolute inset-y-2 right-2 rounded-lg px-5" disabled={busy || setting.trim().length < 3}>
            {busy ? <><Spinner />Writing</> : 'Start'}
          </button>
        </div>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" disabled={busy} onClick={() => setSetting(ex)}
              className="cursor-pointer rounded-full border border-gray-200 bg-white px-3 py-1 text-xs text-gray-600 transition hover:border-teal-600 hover:bg-teal-50 hover:text-teal-900 disabled:cursor-not-allowed">{ex}</button>
          ))}
        </div>
        {problem && <p role="alert" className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-left text-red-800">{problem}</p>}
      </form>

      <div className="mt-10 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-gray-500">
        {SCALES.map((key) => (
          <span key={key} className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: axisColor(key) }} aria-hidden />{AXIS_NOTES[key].title}</span>
        ))}
      </div>
    </div>
  )
}
