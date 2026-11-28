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
        <p className="text-gray-600">This takes up to a minute. In that time:</p>
        <ul className="mt-3 space-y-2">
          {['A GPU is reserved for you', 'The standard model is loaded onto it', 'The model answers a few test questions, for comparison later', 'It writes your first four answers'].map((step) => (
            <li key={step} className="flex items-center gap-2.5 text-gray-700"><span className="h-1.5 w-1.5 rounded-full bg-gray-300" aria-hidden />{step}</li>
          ))}
        </ul>
        {/* A normal start is under a minute; past that, no GPU was free and the service is waiting for one. */}
        {seconds >= 60 && <p className="mt-4 rounded-md bg-amber-50 px-3 py-2 text-amber-900">GPUs are busy right now. Still trying; this can take a few minutes.</p>}
        <p className="mt-4 text-gray-600">You will have 20 minutes once the first question appears.</p>
        <div className="mt-4 h-1 overflow-hidden rounded-full bg-gray-100"><div className="indeterminate h-full w-1/4 rounded-full bg-teal-700" /></div>
      </div>
    </div>
  )
}
