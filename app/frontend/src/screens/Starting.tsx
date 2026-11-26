import { useEffect, useState } from 'react'
import { Spinner } from '../components/Spinner'

/** Shown while the GPU starts and the first answers are written. The wait is real, so the page
 *  lists what happens and how long it has been, without claiming to know which step it is on. */
export function Starting() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => { const t = setInterval(() => setSeconds((s) => s + 1), 1000); return () => clearInterval(t) }, [])
  return (
    <div className="panel mx-auto max-w-xl" role="status" aria-live="polite">
      <div className="flex items-center gap-3 border-b border-gray-200 px-4 py-3">
        <Spinner className="h-4 w-4 text-teal-700" />
        <h1 className="panel-title">Starting your session</h1>
        <span className="hint ml-auto tabular-nums">{seconds} s</span>
      </div>
      <div className="p-4">
        <p className="text-gray-600">This takes about a minute, once per session. In that time:</p>
        <ul className="mt-3 space-y-2">
          {['A GPU is reserved for you', 'The model is loaded onto it', 'The standard model answers a few test questions, for comparison later', 'It writes your first four answers'].map((step) => (
            <li key={step} className="flex items-center gap-2.5 text-gray-700"><span className="h-1.5 w-1.5 rounded-full bg-gray-300" aria-hidden />{step}</li>
          ))}
        </ul>
        <div className="mt-4 h-1 overflow-hidden rounded-full bg-gray-100"><div className="indeterminate h-full w-1/4 rounded-full bg-teal-700" /></div>
      </div>
    </div>
  )
}
