import { useEffect, useState } from 'react'
import { api, type Me, type Round, type Summary } from './api'
import { Shell } from './components/Shell'
import { Gate } from './screens/Gate'
import { Welcome } from './screens/Welcome'
import { Starting } from './screens/Starting'
import { RoundScreen } from './screens/Round'
import { Results } from './screens/Results'

type Stage =
  | { name: 'loading' } | { name: 'gate' } | { name: 'welcome'; error?: string }
  | { name: 'starting'; rounds: number } | { name: 'round'; round: Round } | { name: 'results'; summary: Summary }

export default function App() {
  const [me, setMe] = useState<Me | null>(null)
  const [stage, setStage] = useState<Stage>({ name: 'loading' })

  const refresh = () => api.me().then((m) => { setMe(m); setStage(m.signed_in ? { name: 'welcome' } : { name: 'gate' }) })
    .catch(() => setStage({ name: 'gate' }))
  useEffect(() => { refresh() }, [])

  const begin = async (rounds: number) => {
    setStage({ name: 'starting', rounds })
    try { setStage({ name: 'round', round: await api.start(rounds) }) }
    catch (e) { expired(e) || setStage({ name: 'welcome', error: (e as Error).message }) }
  }
  // A lapsed sign-in sends the visitor back to the password screen.
  const expired = (e: unknown) => (e as { status?: number }).status === 401 && (setStage({ name: 'gate' }), true)
  const signOut = () => api.logout().finally(() => setStage({ name: 'gate' }))

  const progress = stage.name === 'round' ? { at: stage.round.round, of: stage.round.rounds } : undefined
  return (
    <Shell mock={!!me?.mock} progress={progress} onSignOut={stage.name !== 'gate' && stage.name !== 'loading' ? signOut : undefined}>
      {stage.name === 'gate' && <Gate onDone={refresh} />}
      {stage.name === 'welcome' && <Welcome choices={me?.round_choices ?? [5, 10]} error={stage.error} onStart={begin} />}
      {stage.name === 'starting' && <Starting />}
      {stage.name === 'round' && (
        <RoundScreen key={stage.round.round} round={stage.round} onExpired={expired}
          onNext={(next) => setStage('done' in next ? { name: 'results', summary: next } : { name: 'round', round: next })} />
      )}
      {stage.name === 'results' && <Results summary={stage.summary} mock={!!me?.mock} onAgain={() => setStage({ name: 'welcome' })} />}
    </Shell>
  )
}
