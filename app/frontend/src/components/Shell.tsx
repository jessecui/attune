import { useState, type ReactNode } from 'react'
import { isDark, setDark } from '../theme'
import { Logo } from './Logo'

/** The frame around every screen: a compact header, the page, and the standing note. */
export function Shell({ children, mock, progress, minutesLeft, onHome }:
  { children: ReactNode; mock: boolean; progress?: { at: number; of: number };
    /** what is left of the session's time, rounded up to a minute; shown while a session is open */
    minutesLeft?: number; onHome: () => void }) {
  const [dark, setDarkState] = useState(isDark)
  const toggle = () => { setDark(!dark); setDarkState(!dark) }
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-gray-200 bg-surface">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 sm:px-6">
          <button onClick={onHome} aria-label="Attune, back to the start" className="flex cursor-pointer items-center gap-3 rounded-md">
            <Logo />
            <span className="text-base font-semibold text-gray-900">Attune</span>
          </button>
          <span className="hidden h-5 w-px bg-gray-200 sm:block" aria-hidden />
          <p className="hidden truncate text-gray-500 sm:block">Train an AI model tuned to your communication style</p>
          <div className="ml-auto flex items-center gap-4">
            {mock && <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-600">Demo data</span>}
            {progress && <span className="text-gray-600">Question {progress.at} of {progress.of}</span>}
            {minutesLeft !== undefined && (
              <span className={`tabular-nums ${minutesLeft <= 2 ? 'font-medium text-red-700' : 'text-gray-500'}`}>
                {minutesLeft > 0 ? `${minutesLeft} min left` : 'Time is up'}
              </span>
            )}
            <button onClick={toggle} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'} title={dark ? 'Light mode' : 'Dark mode'}
              className="grid h-8 w-8 cursor-pointer place-items-center rounded-md text-gray-500 hover:bg-gray-100 hover:text-gray-900">
              <svg viewBox="0 0 24 24" className="h-[18px] w-[18px]" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                {dark
                  ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>
                  : <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z" />}
              </svg>
            </button>
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
      <footer className="border-t border-gray-200">
        <p className="hint mx-auto max-w-6xl px-4 py-4 text-center sm:px-6">
          Answers are written by an AI model to show a style of speaking. They can be wrong, and they are not medical, legal or financial advice.
        </p>
      </footer>
    </div>
  )
}
