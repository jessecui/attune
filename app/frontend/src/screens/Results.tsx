import { useEffect, useState } from 'react'
import { api, AS_ADJECTIVE, read, type ModelStatus, type Summary } from '../api'
import { Scale } from '../components/Scale'
import { Trajectory, sharedLimit } from '../components/Trajectory'

/** "warm, collaborative and formal" */
const list = (words: string[]) => words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`

export function Results({ summary, mock, onAgain }: { summary: Summary; mock: boolean; onAgain: () => void }) {
  const readings = summary.axes.map((a) => ({ axis: a.axis, ...read(a) }))
  const found = readings.filter((r) => r.strength !== 'none'), open = readings.filter((r) => r.strength === 'none')
  const [shown, setShown] = useState(0)
  const example = summary.examples[shown]
  const limit = sharedLimit(summary.axes)

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-6">
        <ModelCard session={summary.session} mock={mock} instructions={summary.style_instructions} />
        <button className="btn btn-quiet w-full" onClick={onAgain}>Train another</button>
      </div>

      <div className="space-y-6">
        <section className="panel p-5">
          <p className="hint">Result · {summary.rounds} questions</p>
          <h1 className="mt-1 text-lg font-semibold text-gray-900">Your personal AI model is ready</h1>
          <p className="mt-2 text-gray-600">
            {found.length > 0
              ? <>It learned that you prefer answers that are <strong className="font-semibold text-gray-900">{list(found.map((r) => (r.strength === 'leans' ? 'somewhat ' : '') + (AS_ADJECTIVE[r.side] ?? r.side)))}</strong>.</>
              : <>Your picks did not point clearly one way on any scale.</>}
            {open.length > 0 && found.length > 0 && <> On {list(open.map((r) => r.axis))}, your picks showed no clear preference.</>}
          </p>
        </section>

        <section className="panel" aria-labelledby="scales">
          <h2 id="scales" className="panel-title border-b border-gray-200 px-4 py-3">What it learned</h2>
          <div className="divide-y divide-gray-100">{summary.axes.map((a) => <Scale key={a.axis} axis={a} />)}</div>
          <p className="hint border-t border-gray-100 px-4 py-3">In the darker middle, the model did not commit either way. Results can differ between sessions.</p>
        </section>

        {example && (
          <section className="panel" aria-labelledby="difference">
            <div className="flex items-center justify-between gap-3 border-b border-gray-200 px-4 py-3">
              <h2 id="difference" className="panel-title">Standard model against yours, on new questions</h2>
              <div className="flex gap-1" role="tablist" aria-label="Example question">
                {summary.examples.map((_, i) => (
                  <button key={i} role="tab" aria-selected={shown === i} onClick={() => setShown(i)}
                    className={`rounded px-2 py-0.5 text-xs ${shown === i ? 'bg-teal-50 font-medium text-teal-900' : 'text-gray-500 hover:text-gray-900'}`}>{i + 1}</button>
                ))}
              </div>
            </div>
            <div className="p-4">
              <p className="font-medium text-gray-900">{example.question}</p>
              <div className="mt-3 grid gap-4 md:grid-cols-2">
                <div><p className="hint">Standard model</p><p className="mt-1 whitespace-pre-line text-[15px] leading-relaxed text-gray-600">{example.before}</p></div>
                <div className="md:border-l md:border-gray-100 md:pl-4"><p className="text-xs font-medium text-teal-800">Your model</p><p className="mt-1 whitespace-pre-line text-[15px] leading-relaxed text-gray-900">{example.after}</p></div>
              </div>
            </div>
          </section>
        )}

        <details className="panel">
          <summary className="panel-title cursor-pointer px-4 py-3">Training, question by question</summary>
          <div className="border-t border-gray-200 p-4">
            <p className="hint max-w-2xl">
              Each point is the model after one more of your picks. Above the line it leans to the style named at the top, below it
              to the one at the bottom. The vertical scale is compressed: each step from the line is roughly ten times stronger.
            </p>
            <div className="mt-4 grid gap-5 sm:grid-cols-2">{summary.axes.map((a) => <Trajectory key={a.axis} axis={a} limit={limit} />)}</div>
          </div>
        </details>
      </div>
    </div>
  )
}

/** The adapter tuned in this session, packaged a few seconds after the last question. */
function ModelCard({ session, mock, instructions }: { session: string; mock: boolean; instructions: string[] }) {
  const [status, setStatus] = useState<ModelStatus | null>(null), [gaveUp, setGaveUp] = useState(false)
  useEffect(() => {
    if (mock) return
    let stop = false, tries = 0
    const check = async () => {
      try {
        const s = await api.modelStatus(session)
        if (stop) return
        setStatus(s)
        if (s.ready || s.configured === false) return
      } catch { /* keep trying */ }
      if (++tries >= 30) return setGaveUp(true)   // about two minutes
      setTimeout(check, 4000)
    }
    check()
    return () => { stop = true }
  }, [session, mock])

  const megabytes = status?.bytes ? `${Math.round(status.bytes / 1e6)} MB` : null
  return (
    <section className="panel" aria-labelledby="model">
      <h2 id="model" className="panel-title border-b border-gray-200 px-4 py-3">Your model</h2>
      <div className="p-4">
        <p className="text-gray-600">A LoRA adapter for Qwen2.5-7B, trained on your picks. Comes with your profile and a script to run it.</p>
        <div className="mt-4">
          {mock ? <p className="hint">Downloads are switched off in demo mode.</p>
            : status?.configured === false ? <p className="hint">Downloads are not set up on this server.</p>
            : status?.ready ? <a className="btn btn-primary w-full" href={api.modelUrl(session)} download>Download{megabytes && ` (${megabytes})`}</a>
            : gaveUp ? <p className="text-red-700" role="alert">The model could not be packaged this time. Your results are unaffected.</p>
            : <button className="btn btn-primary w-full" disabled>Packaging the model</button>}
          <p className="hint mt-2">Available for a day. Needs a GPU or a recent Mac to run.</p>
        </div>
      </div>
      {instructions.length > 0 && (
        <div className="border-t border-gray-100 p-4">
          <p className="hint">The instructions it settled on</p>
          <ul className="mt-2 space-y-2 text-gray-700">{instructions.map((c) => <li key={c}>{c}</li>)}</ul>
        </div>
      )}
    </section>
  )
}
