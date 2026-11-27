// Everything the page asks of the backend. The backend holds the GPU service's address and key.

export type Round = { session: string; round: number; rounds: number; question: string; candidates: string[]; waited_seconds?: number; role?: string; topic?: string; title?: string; builtin?: boolean;
  /** what is left of the session's fifteen minutes */
  seconds_left?: number }
// A visitor's setting, drafted by the server: who speaks, the session's questions, and the questions the model is rated on.
// points is null for a question the visitor wrote or edited; the server writes its fact sheet at the start.
export type DraftItem = { question: string; points: string[] | null }
/** topic: what the visitor wants to talk about, as it reads after "about"; empty when they named only the role.
 *  title: the whole setting in one line, from the role's side ("A manager giving you feedback on your job performance") */
export type Draft = { suitable: boolean; reason: string; role: string; topic?: string; title?: string; items: DraftItem[]; measure: DraftItem[] }
// written: the judge's 1-to-5 rating of the writing (5 = first style): [standard model, your model], when available.
export type Axis = { axis: string; first: string; second: string; written?: number[] | null }
export type Example = { question: string; before: string; after: string }
export type Summary = { done: true; session: string; rounds: number; axes: Axis[]; examples: Example[]; role?: string; topic?: string; builtin?: boolean; seconds_left?: number }
export type Me = { signed_in: boolean; mock: boolean; round_choices: number[] }
export type ModelStatus = { ready: boolean; bytes?: number | null; mock?: boolean; configured?: boolean }

async function request<T>(path: string, body?: unknown, method = body === undefined ? 'GET' : 'POST'): Promise<T> {
  const r = await fetch(path, {
    method, credentials: 'same-origin',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) {
    const detail = await r.json().then((d) => d.detail, () => null)
    throw Object.assign(new Error(typeof detail === 'string' ? detail : `Request failed (${r.status})`), { status: r.status })
  }
  return r.json()
}

export const api = {
  me: () => request<Me>('/api/me'),
  login: (answer: string) => request<{ signed_in: boolean }>('/api/login', { answer }),
  prepare: (setting: string, rounds: number) => request<Draft>('/api/session/prepare', { setting, rounds }),
  start: (rounds: number, draft?: Draft) =>
    request<Round>('/api/session/start', draft ? { rounds, draft: { role: draft.role, topic: draft.topic ?? '', title: draft.title ?? '', items: draft.items, measure: draft.measure } } : { rounds }),
  feedback: (session: string, round: number, best: number, worst: number) =>
    request<Round | Summary>('/api/session/feedback', { session, round, best, worst }),
  alive: () => request<{ ok: boolean }>('/api/session/alive', {}),
  tryPrepare: (session: string, question: string) => request<{ prepared: boolean }>(`/api/session/${session}/try/prepare`, { question }),
  tryAnswer: (session: string, question: string) => request<Example>(`/api/session/${session}/try/answer`, { question }),
  modelStatus: (session: string) => request<ModelStatus>(`/api/session/${session}/model/status`),
  modelUrl: (session: string) => `/api/session/${session}/model`,
}

// ---- reading a result -------------------------------------------------------------------

export type Reading = { side: string; other: string; strength: 'none' | 'leans' | 'clear'; share: number; position: number; detail: string;
  /** false when the writing ended within half a point of the standard model's: too small a change to call learned */
  moved: boolean;
  /** the training moved the writing toward the other side, but not past the middle: the standard model sat further
   *  out on this side. The side it ended on is then where it started, not something it learned. */
  toward: boolean }

/** What the judge's ratings say about one scale: within half a point of the middle is no clear
 *  preference, up to a full point is a lean, beyond that is stated plainly. */
export function read(a: Axis): Reading {
  if (a.written?.length) {
    // From the writing: within half a point of 3 is no clear preference, a full point or more is clear.
    const rating = a.written[a.written.length - 1], off = Math.abs(rating - 3)
    const side = rating >= 3 ? a.first : a.second, other = rating >= 3 ? a.second : a.first
    const change = rating - a.written[0], moved = Math.abs(change) >= 0.5
    const toward = moved && (rating >= 3 ? change < 0 : change > 0)
    const standard = (rating >= 3 ? a.written[0] : 6 - a.written[0]).toFixed(1)   // the standard model's rating for the same side
    return { side, other, strength: off < 0.5 ? 'none' : off < 1 ? 'leans' : 'clear', share: off / 2, position: (rating - 1) / 4, moved, toward,
             // the standard model's rating on the same side, so "close" can be checked against the number
             detail: `Its writing rated ${(3 + off).toFixed(1)} out of 5 for ${shown(side)}`
               + (toward ? `, moved toward ${shown(other)} from the standard model's ${standard}.` : moved ? '.' : standard === (3 + off).toFixed(1) ? ', the same as the standard model.' : `, close to the standard model's ${standard}.`) }
  }
  // The judge was unavailable, so there is no rating to read.
  return { side: a.first, other: a.second, strength: 'none', share: 0, position: 0.5, moved: false, toward: false, detail: 'Not rated this time.' }
}

export const AXIS_NOTES: Record<string, { title: string; about: string }> = {
  warmth: { title: 'Empathy', about: 'Whether an answer first acknowledges how you might feel, or goes straight to the point.' },
  directiveness: { title: 'Directiveness', about: 'Whether you are told what to do, or shown what works and left to decide.' },
  formality: { title: 'Formality', about: 'A relaxed, conversational register, or a formal one.' },
  depth: { title: 'Depth', about: 'Whether each piece of advice comes with its reason.' },
}
export const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
// The service calls the empathy scale's first style "warm"; the page shows it under the scale's own name.
export const shown = (style: string) => (style === 'warm' ? 'empathetic' : style)

// How each style reads in a sentence ("answers that are empathetic, collaborative, and formal").
// Depth is about reasons, not length, so it is phrased as what the answers do, not as "concise".
export const AS_ADJECTIVE: Record<string, string> = {
  warm: 'empathetic', neutral: 'matter-of-fact', directive: 'directive', collaborative: 'collaborative', casual: 'casual', formal: 'formal',
}
export const AS_CLAUSE: Record<string, string> = { 'explains why': 'explain the reasons', 'just the advice': 'give the advice without the reasons' }
// A move toward a style that stopped short of it ("became more empathetic than the standard model").
export const AS_MORE: Record<string, string> = {
  warm: 'more empathetic', neutral: 'more matter-of-fact', directive: 'more directive', collaborative: 'more collaborative', casual: 'more casual',
  formal: 'more formal', 'explains why': 'more inclined to explain the reasons', 'just the advice': 'more inclined to give just the advice',
}

// Each scale keeps one colour wherever it appears.
export const axisColor = (axis: string) => `var(--color-${axis}, var(--color-accent))`
