import { useEffect, useState } from 'react'

/** Shown while the GPU starts and the first answers are written. The wait is real, so the
 *  page says what is happening and how long it has been, without inventing a progress figure. */
export function Starting() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => { const t = setInterval(() => setSeconds((s) => s + 1), 1000); return () => clearInterval(t) }, [])
  return (
    <div className="mx-auto max-w-xl pt-10" role="status" aria-live="polite">
      <p className="eyebrow">Preparing your session</p>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Setting up a model just for you</h1>
      <p className="mt-3 text-ink-2">
        A GPU is being reserved and a fresh copy of the model loaded, then it writes your first four answers.
        This usually takes one to two minutes, and only happens once per session.
      </p>
      <div className="mt-8 h-1 overflow-hidden rounded-full bg-rule"><div className="indeterminate h-full w-1/4 rounded-full bg-accent" /></div>
      <p className="mt-3 font-mono text-sm text-muted">{seconds} s</p>
    </div>
  )
}
