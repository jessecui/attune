import { useEffect, useRef, useState } from 'react'
import { api, type Draft, type Me, type Round, type Summary } from './api'
import { Shell } from './components/Shell'
import { Gate } from './screens/Gate'
import { Welcome } from './screens/Welcome'
import { Review } from './screens/Review'
import { Starting } from './screens/Starting'
import { RoundScreen } from './screens/Round'
import { Results } from './screens/Results'

type Stage =
  | { name: 'loading' } | { name: 'gate' } | { name: 'welcome'; error?: string } | { name: 'review'; rounds: number; draft: Draft }
  | { name: 'starting'; rounds: number } | { name: 'round'; round: Round } | { name: 'results'; summary: Summary }

export default function App() {
  const [me, setMe] = useState<Me | null>(null)
  const [stage, setStage] = useState<Stage>({ name: 'loading' })

  const refresh = () => api.me().then((m) => { setMe(m); setStage(m.signed_in ? { name: 'welcome' } : { name: 'gate' }) })
    .catch(() => setStage({ name: 'gate' }))
  useEffect(() => { refresh() }, [])

  // With a draft, the session runs on the visitor's setting; without one, on the built-in questions.
  // Each start gets a number. If the visitor goes home while the GPU is starting, the number moves
  // on and the late reply is ignored, so they are not pulled into a session they walked away from.
  const attempt = useRef(0)
  const begin = async (rounds: number, draft?: Draft) => {
    const mine = ++attempt.current
    setStage({ name: 'starting', rounds })
    try {
      const round = await api.start(rounds, draft)
      if (attempt.current === mine) { clock(round); setStage({ name: 'round', round }) }
    } catch (e) { if (attempt.current === mine && !expired(e)) setStage({ name: 'welcome', error: (e as Error).message }) }
  }
  // The logo goes home. Part-way through a session that means giving it up, so ask first.
  const goHome = () => {
    if (stage.name === 'gate' || stage.name === 'loading' || stage.name === 'welcome') return
    if (stage.name === 'round' && !window.confirm('Leave this session? The model trained so far will be lost.')) return
    attempt.current++
    setStage({ name: 'welcome' })
  }
  // A session lasts twenty minutes from its first question. Each reply says how long is left; the
  // header counts it down, and when it runs out the questions end (the results, once shown, stay).
  const [endsAt, setEndsAt] = useState<number | null>(null), [now, setNow] = useState(Date.now())
  const clock = (reply: { seconds_left?: number }) => { if (reply.seconds_left !== undefined) setEndsAt(Date.now() + reply.seconds_left * 1000) }
  const open = stage.name === 'round' || stage.name === 'results'
  useEffect(() => { if (!open) setEndsAt(null) }, [open])
  useEffect(() => {
    if (endsAt === null) return
    setNow(Date.now())
    const t = setInterval(() => setNow(Date.now()), 5000)
    return () => clearInterval(t)
  }, [endsAt])
  const left = endsAt === null || !open ? undefined : Math.max(0, endsAt - now)
  // The GPU stops after five minutes without a request, and the session is in its memory. While a
  // session is open and has time left, a request every two minutes keeps it running.
  const live = left !== undefined && left > 0
  useEffect(() => {
    if (!live) return
    const t = setInterval(() => { api.alive().catch(() => {}) }, 120000)
    return () => clearInterval(t)
  }, [live])
  useEffect(() => {
    if (left === 0 && stage.name === 'round') {
      attempt.current++
      setStage({ name: 'welcome', error: 'Your 20 minutes were up, so that session has ended. Start another whenever you like.' })
    }
  }, [left, stage.name])
  // A lapsed sign-in sends the visitor back to the password screen.
  const expired = (e: unknown) => (e as { status?: number }).status === 401 && (setStage({ name: 'gate' }), true)

  const progress = stage.name === 'round' ? { at: stage.round.round, of: stage.round.rounds } : undefined
  return (
    <Shell mock={!!me?.mock} progress={progress} minutesLeft={left === undefined ? undefined : Math.ceil(left / 60000)} onHome={goHome}>
      {/* The password prompt sits over the start screen, which is inert until sign-in. */}
      {stage.name === 'gate' && <><div inert aria-hidden><Welcome choices={me?.round_choices ?? [7]} onStart={() => {}} onDraft={() => {}} onExpired={() => false} /></div><Gate onDone={refresh} /></>}
      {stage.name === 'welcome' && <Welcome key={stage.error ?? ''} choices={me?.round_choices ?? [7]} error={stage.error} onStart={(r) => begin(r)}
        onDraft={(rounds, draft) => setStage({ name: 'review', rounds, draft })} onExpired={expired} />}
      {stage.name === 'review' && <Review draft={stage.draft} onBack={() => setStage({ name: 'welcome' })} onStart={(d) => begin(stage.rounds, d)} />}
      {stage.name === 'starting' && <Starting />}
      {stage.name === 'round' && (
        <RoundScreen key={stage.round.round} round={stage.round} onExpired={expired}
          onNext={(next) => { clock(next); setStage((now) =>
            // A reply for a session the visitor has since left (gone home, or started another) is dropped.
            now.name !== 'round' || now.round.session !== next.session ? now
              : 'done' in next ? { name: 'results', summary: next } : { name: 'round', round: next }) }} />
      )}
      {stage.name === 'results' && <Results summary={stage.summary} mock={!!me?.mock} over={left === 0} onAgain={() => setStage({ name: 'welcome' })} onExpired={expired} />}
    </Shell>
  )
}
