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
    <div>
      <p className="eyebrow">Result · {summary.rounds} questions</p>
      <h1 className="mt-3 text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">Your personal AI model <span className="mark">is ready</span>.</h1>
      <p className="mt-6 max-w-3xl text-lg text-ink-2">
        {found.length > 0
          ? <>It learned that you prefer answers that are <strong className="font-semibold text-ink">{list(found.map((r) => (r.strength === 'leans' ? 'somewhat ' : '') + (AS_ADJECTIVE[r.side] ?? r.side)))}</strong>.</>
          : <>Your picks did not point clearly one way on any scale.</>}
        {open.length > 0 && found.length > 0 && <> On {list(open.map((r) => r.axis))}, your picks showed no clear preference.</>}
      </p>

      <div className="mt-12 grid gap-8 lg:grid-cols-[1fr_1.2fr]">
        <ModelCard session={summary.session} mock={mock} instructions={summary.style_instructions} />
        <section className="sheet p-8" aria-labelledby="scales">
          <h2 id="scales" className="eyebrow">What it learned</h2>
          <div className="mt-7 divide-y divide-rule">{summary.axes.map((a) => <Scale key={a.axis} axis={a} />)}</div>
          <p className="mt-7 border-t border-rule pt-5 text-sm text-muted">
            In the shaded middle, the model did not commit either way.
          </p>
        </section>
      </div>

      {example && (
        <section className="mt-8 sheet p-8" aria-labelledby="difference">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2 id="difference" className="eyebrow">Standard model against yours, on new questions</h2>
            <div className="flex gap-1" role="tablist" aria-label="Example question">
              {summary.examples.map((_, i) => (
                <button key={i} role="tab" aria-selected={shown === i} onClick={() => setShown(i)}
                  className={`rounded px-2.5 py-1 font-mono text-xs ${shown === i ? 'bg-accent-soft text-accent-ink' : 'text-muted hover:text-ink'}`}>{i + 1}</button>
              ))}
            </div>
          </div>
          <p className="mt-6 text-lg font-medium">{example.question}</p>
          <div className="mt-5 grid gap-8 md:grid-cols-2">
            <div><p className="eyebrow">Standard model</p><p className="mt-2 whitespace-pre-line text-ink-2">{example.before}</p></div>
            <div className="md:border-l md:border-rule md:pl-8"><p className="eyebrow !text-accent-ink">Your model</p><p className="mt-2 whitespace-pre-line">{example.after}</p></div>
          </div>
        </section>
      )}

      <details className="mt-8 sheet p-8">
        <summary className="cursor-pointer font-medium">Training, question by question</summary>
        <p className="mt-3 max-w-3xl text-sm text-ink-2">
          Each point is the model after one more of your picks. Above the line it leans to the style named at the top, below it
          to the one at the bottom. The vertical scale is compressed: each step from the line is roughly ten times stronger.
        </p>
        <div className="mt-5 grid gap-6 sm:grid-cols-2">{summary.axes.map((a) => <Trajectory key={a.axis} axis={a} limit={limit} />)}</div>
      </details>

      <div className="mt-10 flex flex-wrap items-center gap-4">
        <button className="btn btn-quiet" onClick={onAgain}>Train another</button>
        <span className="text-sm text-muted">Results can differ between sessions.</span>
      </div>
    </div>
  )
}

/** The payoff: the adapter tuned in this session, packaged a few seconds after the last round. */
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
    <section className="sheet model-card flex flex-col p-8" aria-labelledby="model">
      <h2 id="model" className="eyebrow !text-white/60">Your model</h2>
      <p className="mt-4 text-xl font-semibold tracking-tight">Your personal model</p>
      <p className="mt-2 text-white/75">
        A LoRA adapter for Qwen2.5-7B, trained on your picks. Comes with your profile and a script to run it.
      </p>
      {instructions.length > 0 && (
        <div className="mt-8">
          <p className="text-sm font-medium">The instructions it settled on</p>
          <ul className="mt-3 space-y-2.5 text-sm text-white/75">
            {instructions.map((c) => <li key={c} className="flex gap-2"><span className="text-teal-300" aria-hidden>—</span><span>{c}</span></li>)}
          </ul>
        </div>
      )}
      <div className="mt-auto pt-10">
        {mock ? <p className="text-sm text-white/60">Downloads are switched off in demo mode.</p>
          : status?.configured === false ? <p className="text-sm text-white/60">Downloads are not set up on this server.</p>
          : status?.ready ? <a className="btn w-full bg-white text-ink hover:bg-teal-50" href={api.modelUrl(session)} download>Download your model{megabytes && ` (${megabytes})`}</a>
          : gaveUp ? <p className="text-sm text-orange-200" role="alert">The model could not be packaged this time. Your profile above is unaffected.</p>
          : <button className="btn w-full bg-white/15 text-white" disabled>Packaging the model</button>}
        <p className="mt-3 text-sm text-white/60">Available for a day. Needs a GPU or a recent Mac to run.</p>
      </div>
    </section>
  )
}
