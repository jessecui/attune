import { useEffect, useState } from 'react'

/** Shown while the GPU starts and the first answers are written. The wait is real, so the
 *  page says what is happening and how long it has been, without inventing a progress figure. */
export function Starting() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => { const t = setInterval(() => setSeconds((s) => s + 1), 1000); return () => clearInterval(t) }, [])
  return (
    <div className="panel mx-auto max-w-xl p-5" role="status" aria-live="polite">
      <h1 className="panel-title">Starting your GPU</h1>
      <p className="mt-2 text-gray-600">
        A GPU is being reserved and the model loaded. It then writes the first four answers. This takes about a minute, once per session.
      </p>
      <div className="mt-4 h-1 overflow-hidden rounded-full bg-gray-200"><div className="indeterminate h-full w-1/4 rounded-full bg-teal-700" /></div>
      <p className="hint mt-2">{seconds} s</p>
    </div>
  )
}
