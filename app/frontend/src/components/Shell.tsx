import type { ReactNode } from 'react'
import { Logo } from './Logo'

/** The frame around every screen: a compact header, the page, and the standing note. */
export function Shell({ children, mock, progress, onSignOut }:
  { children: ReactNode; mock: boolean; progress?: { at: number; of: number }; onSignOut?: () => void }) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 sm:px-6">
          <Logo />
          <span className="text-base font-semibold text-gray-900">Attune</span>
          <span className="hidden h-5 w-px bg-gray-200 sm:block" aria-hidden />
          <p className="hidden truncate text-gray-500 sm:block">Train your own personal AI model</p>
          <div className="ml-auto flex items-center gap-4">
            {mock && <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-600">Demo data</span>}
            {progress && <span className="text-gray-600">Question {progress.at} of {progress.of}</span>}
            {onSignOut && <button className="text-gray-500 hover:text-gray-900" onClick={onSignOut}>Sign out</button>}
          </div>
        </div>
        {progress && (
          <div className="mx-auto flex max-w-6xl gap-1 px-4 pb-2.5 sm:px-6" aria-hidden>
            {Array.from({ length: progress.of }, (_, i) => (
              <div key={i} className={`h-1 flex-1 rounded-full ${i < progress.at - 1 ? 'bg-teal-700' : i === progress.at - 1 ? 'bg-teal-700/40' : 'bg-gray-200'}`} />
            ))}
          </div>
        )}
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6">{children}</main>
      <footer className="mx-auto w-full max-w-6xl px-4 pb-6 sm:px-6">
        <p className="hint px-1">The answers you read demonstrate a style. They are not medical, legal or financial advice.</p>
      </footer>
    </div>
  )
}
