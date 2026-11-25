import { useState, type FormEvent } from 'react'
import { api } from '../api'

export function Gate({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState(''), [error, setError] = useState(''), [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setError('')
    try { await api.login(password); onDone() } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }
  return (
    <div className="mx-auto max-w-sm pt-10">
      <p className="eyebrow">Invitation only</p>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight">Sign in to begin</h1>
      <p className="mt-2 text-ink-2">Each session reserves a GPU for one person, so access is by password.</p>
      <form onSubmit={submit} className="sheet mt-6 p-5">
        <label htmlFor="password" className="block text-sm font-medium">Password</label>
        <input id="password" type="password" autoFocus autoComplete="current-password" value={password}
          onChange={(e) => setPassword(e.target.value)} aria-invalid={!!error} aria-describedby={error ? 'gate-error' : undefined}
          className="mt-1.5 w-full rounded-md border border-rule-strong bg-sheet px-3 py-2 focus:border-accent" />
        {error && <p id="gate-error" role="alert" className="mt-2 text-sm text-danger">{error}</p>}
        <button className="btn btn-primary mt-4 w-full" disabled={busy || !password}>{busy ? 'Checking' : 'Continue'}</button>
      </form>
    </div>
  )
}
