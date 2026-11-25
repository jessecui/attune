import type { ReactNode } from 'react'

/** The frame around every screen: a quiet header, the page, and the standing note. */
export function Shell({ children, mock, progress, onSignOut }:
  { children: ReactNode; mock: boolean; progress?: { at: number; of: number }; onSignOut?: () => void }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-rule bg-sheet">
        <div className="mx-auto max-w-5xl px-5 h-14 flex items-center gap-4">
          <div className="flex items-baseline gap-3">
            <span className="text-lg font-semibold tracking-tight">Attune</span>
            <span className="eyebrow hidden sm:inline">Live model fine-tuning</span>
          </div>
          <div className="ml-auto flex items-center gap-4">
            {mock && <span className="eyebrow rounded border border-rule-strong px-2 py-0.5">Demo data</span>}
            {progress && <span className="font-mono text-sm text-ink-2">Question {progress.at} of {progress.of}</span>}
            {onSignOut && <button className="text-sm text-muted hover:text-ink underline-offset-4 hover:underline" onClick={onSignOut}>Sign out</button>}
          </div>
        </div>
        {progress && (
          <div className="mx-auto max-w-5xl px-5 pb-3 flex gap-1.5" aria-hidden>
            {Array.from({ length: progress.of }, (_, i) => (
              <div key={i} className={`h-1 flex-1 rounded-full ${i < progress.at - 1 ? 'bg-accent' : i === progress.at - 1 ? 'bg-accent/50' : 'bg-rule'}`} />
            ))}
          </div>
        )}
      </header>
      <main className="relative flex-1">
        <div className="dots pointer-events-none absolute inset-x-0 top-0 h-72" aria-hidden />
        <div className="relative mx-auto w-full max-w-5xl px-5 py-14">{children}</div>
      </main>
      <footer className="border-t border-rule">
        <p className="mx-auto max-w-5xl px-5 py-5 text-sm text-muted">
          The answers you read demonstrate a style. They are not medical, legal or financial advice.
        </p>
      </footer>
    </div>
  )
}
