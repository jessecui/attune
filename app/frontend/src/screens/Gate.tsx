import { useState, type FormEvent } from 'react'
import { Logo } from '../components/Logo'
import { api } from '../api'

/** The password prompt, shown as a dialog over the start screen. */
export function Gate({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState(''), [error, setError] = useState(''), [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setError('')
    try { await api.login(password); onDone() } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="fixed inset-0 z-10 grid place-items-center bg-black/50 p-4">
      <form onSubmit={submit} role="dialog" aria-modal="true" aria-labelledby="gate-title" className="w-full max-w-md rounded-lg bg-surface p-6 shadow-xl">
        <div className="flex items-center gap-2.5">
          <Logo />
          <h1 id="gate-title" className="text-lg font-semibold text-gray-900">Attune</h1>
        </div>
        <p className="mt-3 text-gray-500">Train your own custom AI model.</p>
        <label htmlFor="password" className="mt-5 block font-medium leading-snug text-gray-900">Each session runs on its own GPU, so access is by password.</label>
        <input id="password" type="password" autoFocus autoComplete="current-password" placeholder="Password" value={password}
          onChange={(e) => setPassword(e.target.value)} aria-invalid={!!error} aria-describedby={error ? 'gate-error' : undefined}
          className="mt-4 w-full rounded-md border border-gray-300 bg-surface px-3 py-2.5 focus:border-teal-600 focus:outline-none focus:ring-4 focus:ring-teal-600/10" />
        {error && <p id="gate-error" role="alert" className="mt-2 text-red-700">{error}</p>}
        <button className="btn btn-primary mt-4 w-full" disabled={busy || !password}>{busy ? 'Checking' : 'Continue'}</button>
      </form>
    </div>
  )
}
