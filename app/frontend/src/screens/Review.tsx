import { useState } from 'react'
import type { Draft } from '../api'

/** The drafted questions, before training starts. Any of them can be edited or replaced; a changed
 *  question loses its drafted fact sheet and gets a new one written when the session starts. */
export function Review({ draft, onStart, onBack }: { draft: Draft; onStart: (draft: Draft) => void; onBack: () => void }) {
  const [questions, setQuestions] = useState(draft.items.map((i) => i.question))
  const changed = (i: number) => questions[i].trim() !== draft.items[i].question
  const tooShort = questions.some((q) => q.trim().length < 8)
  const start = () => onStart({
    ...draft,
    items: draft.items.map((item, i) => (changed(i) ? { question: questions[i].trim(), points: null } : item)),
  })
  return (
    <div className="panel mx-auto max-w-3xl">
      <div className="border-b border-gray-200 px-4 py-3">
        <h1 className="panel-title">Your questions</h1>
        <p className="mt-0.5 text-gray-600">The model will speak to you as <span className="font-medium text-gray-900">{draft.role}</span>{draft.topic && <>, about <span className="font-medium text-gray-900">{draft.topic}</span></>}. These are questions you might ask them; edit or replace any.</p>
      </div>
      <ol className="divide-y divide-gray-100">
        {questions.map((q, i) => (
          <li key={i} className="flex items-center gap-3 px-4 py-2.5">
            <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-gray-100 text-xs font-medium text-gray-600">{i + 1}</span>
            <input value={q} maxLength={200} aria-label={`Question ${i + 1}`}
              onChange={(e) => setQuestions(questions.map((old, j) => (j === i ? e.target.value : old)))}
              className="min-w-0 flex-1 rounded-md border border-transparent px-2 py-1.5 hover:border-gray-200 focus:border-teal-600" />
            {changed(i) && <span className="hint shrink-0">edited</span>}
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap items-center gap-3 border-t border-gray-200 px-4 py-3">
        <button className="btn btn-quiet" onClick={onBack}>Back</button>
        <span className="hint">{tooShort ? 'Each question needs at least a few words.' : 'The GPU has been starting while you read.'}</span>
        <button className="btn btn-primary ml-auto" disabled={tooShort} onClick={start}>Start training</button>
      </div>
    </div>
  )
}
