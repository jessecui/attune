// Everything the page asks of the backend. The backend holds the GPU service's address and key.

export type Round = { session: string; round: number; rounds: number; question: string; candidates: string[]; waited_seconds?: number }
// lean: the model's log-odds of choosing the first style, after each round.
// written: the judge's 1-to-5 rating of the model's writing after each round (5 = first style), when available.
export type Axis = { axis: string; first: string; second: string; lean: number[]; written?: number[] | null }
export type Example = { question: string; before: string; after: string }
export type Summary = { done: true; session: string; rounds: number; axes: Axis[]; examples: Example[]; style_instructions: string[] }
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
  login: (password: string) => request<{ signed_in: boolean }>('/api/login', { password }),
  logout: () => request('/api/logout', {}),
  start: (rounds: number) => request<Round>('/api/session/start', { rounds }),
  feedback: (session: string, best: number, worst: number) =>
    request<Round | Summary>('/api/session/feedback', { session, best, worst }),
  modelStatus: (session: string) => request<ModelStatus>(`/api/session/${session}/model/status`),
  modelUrl: (session: string) => `/api/session/${session}/model`,
}

// ---- reading a result -------------------------------------------------------------------

// A lean is the model's log-odds for an axis's first style over its second.
export const probability = (lean: number) => 1 / (1 + Math.exp(-lean))

export type Reading = { side: string; other: string; strength: 'none' | 'leans' | 'clear'; share: number; position: number; detail: string;
  /** false when the writing ended within half a point of the standard model's: too small a change to call learned */
  moved: boolean }

/** What the profile says about one axis: under 75% either way is no clear preference,
 *  75% to 95% is a lean, above that is stated plainly. */
export function read(a: Axis): Reading {
  if (a.written?.length) {
    // From the writing: within half a point of 3 is no clear preference, a full point or more is clear.
    const rating = a.written[a.written.length - 1], off = Math.abs(rating - 3)
    const side = rating >= 3 ? a.first : a.second, other = rating >= 3 ? a.second : a.first
    const moved = Math.abs(rating - a.written[0]) >= 0.5
    return { side, other, strength: off < 0.5 ? 'none' : off < 1 ? 'leans' : 'clear', share: off / 2, position: (rating - 1) / 4, moved,
             // the standard model's rating on the same side, so "close" can be checked against the number
             detail: `Its writing rated ${(3 + off).toFixed(1)} out of 5 for ${side}`
               + (moved ? '.' : `, close to the standard model's ${(rating >= 3 ? a.written[0] : 6 - a.written[0]).toFixed(1)}.`) }
  }
  const last = a.lean[a.lean.length - 1], prev = a.lean[a.lean.length - 2] ?? last
  const p = probability(last), share = Math.max(p, 1 - p)
  const side = last >= 0 ? a.first : a.second, other = last >= 0 ? a.second : a.first
  const settled = Math.sign(last) === Math.sign(prev) || prev === 0
  const strength = share < 0.75 || !settled ? 'none' : share < 0.95 ? 'leans' : 'clear'
  return { side, other, strength, share, position: p, moved: true,
           detail: `Ended ${share > 0.95 ? 'more than 95%' : Math.floor(share * 100) + '%'} ${side} by its style choices.` }
}

export const AXIS_NOTES: Record<string, { title: string; about: string }> = {
  warmth: { title: 'Warmth', about: 'Whether an answer first acknowledges how you might feel, or goes straight to the point.' },
  directiveness: { title: 'Directiveness', about: 'Whether you are told what to do, or shown what works and left to decide.' },
  formality: { title: 'Formality', about: 'A relaxed, conversational register, or a formal one.' },
  depth: { title: 'Depth', about: 'Whether each piece of advice comes with its reason.' },
}
export const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

// How each style reads in a sentence ("answers that are warm, collaborative and formal").
// Depth is about reasons, not length, so it is phrased as what the answers do, not as "concise".
export const AS_ADJECTIVE: Record<string, string> = {
  warm: 'warm', neutral: 'matter-of-fact', directive: 'directive', collaborative: 'collaborative', casual: 'casual', formal: 'formal',
}
export const AS_CLAUSE: Record<string, string> = { 'explains why': 'explain the reasons', 'just the advice': 'give the advice without the reasons' }

// Each scale keeps one colour wherever it appears.
export const axisColor = (axis: string) => `var(--color-${axis}, var(--color-accent))`
