import { useEffect, useState } from 'react'
import { api, AS_ADJECTIVE, AS_CLAUSE, read, type ModelStatus, type Summary } from '../api'
import { Scale } from '../components/Scale'
import { Spinner } from '../components/Spinner'

/** "warm, collaborative and formal" */
const list = (words: string[]) => words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`

/** "are warm, collaborative and formal, and give the advice without the reasons" */
function describe(found: { side: string; strength: string }[]) {
  const adjectives = found.filter((r) => AS_ADJECTIVE[r.side]).map((r) => (r.strength === 'leans' ? 'somewhat ' : '') + AS_ADJECTIVE[r.side])
  const clauses = found.filter((r) => AS_CLAUSE[r.side]).map((r) => (r.strength === 'leans' ? 'mostly ' : '') + AS_CLAUSE[r.side])
  return [adjectives.length ? `are ${list(adjectives)}` : '', ...clauses].filter(Boolean).join(', and ')
}

export function Results({ summary, mock, onAgain }: { summary: Summary; mock: boolean; onAgain: () => void }) {
  const readings = summary.axes.map((a) => ({ axis: a.axis, ...read(a) }))
  // Only scales the training moved count as learned; the rest either show no preference or match the standard model.
  const found = readings.filter((r) => r.strength !== 'none' && r.moved), open = readings.filter((r) => r.strength === 'none')
  const same = readings.filter((r) => r.strength !== 'none' && !r.moved)
  // "when working with a workplace manager": the setting the visitor chose, if they chose one
  const setting = summary.role && !summary.builtin ? ` when working with ${summary.role}` : ''
  const [shown, setShown] = useState(0)
  const example = summary.examples[shown]
  const rated = summary.axes.every((a) => a.written?.length)

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-6">
        <ModelCard session={summary.session} mock={mock} />
        <button className="btn btn-quiet w-full" onClick={onAgain}>Train another</button>
      </div>

      <div className="space-y-6">
        <section className="panel p-5">
          <p className="hint">Result · {summary.rounds} questions</p>
          <h1 className="mt-1 text-lg font-semibold text-gray-900">Your personal AI model is ready</h1>
          <p className="mt-2 text-gray-600">
            {found.length > 0
              ? <>It learned that{setting}, you prefer answers that <strong className="font-semibold text-gray-900">{describe(found)}</strong>.</>
              : <>Your picks did not move the model clearly on any scale.</>}
            {open.length > 0 && found.length > 0 && <> On {list(open.map((r) => r.axis))}, your picks showed no clear preference.</>}
            {same.length > 0 && <> On {list(same.map((r) => r.axis))}, it writes much as the standard model already does.</>}
          </p>
        </section>

        <section className="panel" aria-labelledby="scales">
          <h2 id="scales" className="panel-title border-b border-gray-200 px-4 py-3">What it learned</h2>
          <div className="divide-y divide-gray-100">{summary.axes.map((a) => <Scale key={a.axis} axis={a} />)}</div>
          <p className="hint border-t border-gray-100 px-4 py-3">
            {rated ? <>Rated by a second AI model from answers to six questions your model was not trained on. <Marker hollow /> standard model, <Marker /> your model. The darker middle is no clear preference.</>
              : 'The writing could not be rated this time, so these show the styles your model chooses. The darker middle is no clear preference.'}
            {' '}Results can differ between sessions.
          </p>
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
      </div>
    </div>
  )
}

/** The key to the two markers on each scale. */
const Marker = ({ hollow }: { hollow?: boolean }) => (
  <span className={`mx-0.5 inline-block h-2.5 w-2.5 rounded-full align-[-1px] ${hollow ? 'border-2 border-gray-400 bg-white' : 'bg-gray-600'}`} aria-hidden />
)

/** The adapter tuned in this session, packaged a few seconds after the last question. */
function ModelCard({ session, mock }: { session: string; mock: boolean }) {
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
        <p className="text-gray-600">Your own version of an open AI model with 7 billion parameters, trained on your picks. The download includes your results and everything needed to run it on your own computer.</p>
        <div className="mt-4">
          {mock ? <p className="hint">Downloads are switched off in demo mode.</p>
            : status?.configured === false ? <p className="hint">Downloads are not set up on this server.</p>
            : status?.ready ? <a className="btn btn-primary w-full" href={api.modelUrl(session)} download>Download{megabytes && ` (${megabytes})`}</a>
            : gaveUp ? <p className="text-red-700" role="alert">The model could not be packaged this time. Your results are unaffected.</p>
            : <button className="btn btn-primary w-full" disabled><Spinner />Packaging the model</button>}
          <p className="hint mt-2">Available for a day. Running it needs a recent Mac or a computer with a graphics card.</p>
        </div>
      </div>
    </section>
  )
}
