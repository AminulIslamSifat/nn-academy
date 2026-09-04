'use client'

import { useState, useMemo, useCallback } from 'react'
import { CheckCircle2, XCircle, RotateCcw, ChevronDown } from 'lucide-react'
import { useAcademyStore } from '@/lib/store'

interface QuizQuestion {
  id: string
  type: 'multiple-choice' | 'shape-prediction' | 'code-output'
  prompt: string
  code?: string
  options: string[]
  correctIndex: number
  explanation: string
  randomize?: boolean
}

interface QuizProps {
  chapterSlug: string
  questions: QuizQuestion[]
}

function shuffleWithAnswer(questions: QuizQuestion[]): QuizQuestion[] {
  return questions.map((q) => {
    if (!q.randomize) return q
    const indices = q.options.map((_, i) => i)
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[indices[i], indices[j]] = [indices[j], indices[i]]
    }
    const newOptions = indices.map((i) => q.options[i])
    const newCorrectIndex = indices.indexOf(q.correctIndex)
    return { ...q, options: newOptions, correctIndex: newCorrectIndex }
  })
}

export default function Quiz({ chapterSlug, questions }: QuizProps) {
  const [answers, setAnswers] = useState<Record<string, number>>({})
  const [submitted, setSubmitted] = useState(false)
  const [showExplanations, setShowExplanations] = useState<Set<string>>(new Set())
  const { saveQuizScore } = useAcademyStore()

  const shuffledQuestions = useMemo(
    () => shuffleWithAnswer(questions),
    [questions]
  )

  const handleSelect = useCallback((questionId: string, optionIndex: number) => {
    if (submitted) return
    setAnswers((prev) => ({ ...prev, [questionId]: optionIndex }))
  }, [submitted])

  const { markCompleted } = useAcademyStore()

  const handleSubmit = useCallback(async () => {
    setSubmitted(true)
    // Save scores
    let allCorrect = true
    for (const q of shuffledQuestions) {
      const correct = answers[q.id] === q.correctIndex ? 1 : 0
      if (correct === 0) allCorrect = false
      await saveQuizScore(chapterSlug, q.id, correct)
    }
    // Auto-complete chapter if all answers correct
    if (allCorrect && shuffledQuestions.length > 0) {
      await markCompleted(chapterSlug)
    }
  }, [answers, shuffledQuestions, chapterSlug, saveQuizScore, markCompleted])

  const handleRetry = useCallback(() => {
    setAnswers({})
    setSubmitted(false)
    setShowExplanations(new Set())
  }, [])

  const toggleExplanation = useCallback((id: string) => {
    setShowExplanations((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const score = shuffledQuestions.filter(
    (q) => answers[q.id] === q.correctIndex
  ).length

  const allAnswered = shuffledQuestions.every((q) => answers[q.id] !== undefined)

  return (
    <div className="my-8 rounded-xl border border-zinc-800 bg-surface-900/30 p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-zinc-200">Knowledge Check</h3>
        {submitted && (
          <div className="flex items-center gap-3">
            <span className={`text-sm font-medium ${score === shuffledQuestions.length ? 'text-accent-400' : 'text-zinc-400'}`}>
              {score}/{shuffledQuestions.length} correct
            </span>
            <button
              onClick={handleRetry}
              className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition"
            >
              <RotateCcw size={12} /> Retry
            </button>
          </div>
        )}
      </div>

      <div className="space-y-6">
        {shuffledQuestions.map((q, qi) => {
          const selected = answers[q.id]
          const isCorrect = selected === q.correctIndex

          return (
            <div key={q.id} className="rounded-lg border border-zinc-800/50 p-4">
              <p className="text-sm font-medium text-zinc-300 mb-3">
                <span className="text-accent-400 mr-2">Q{qi + 1}.</span>
                {q.prompt}
              </p>

              {q.code && (
                <pre className="mb-3 rounded-lg bg-surface-950 p-3 text-xs font-mono text-zinc-400 overflow-x-auto">
                  {q.code}
                </pre>
              )}

              <div className="space-y-2">
                {q.options.map((opt, oi) => {
                  let optClass = 'border-zinc-700/50 hover:border-zinc-600 text-zinc-300'
                  if (submitted) {
                    if (oi === q.correctIndex) {
                      optClass = 'border-accent-500/50 bg-accent-500/5 text-accent-300'
                    } else if (oi === selected) {
                      optClass = 'border-red-500/50 bg-red-500/5 text-red-300'
                    } else {
                      optClass = 'border-zinc-800 text-zinc-600'
                    }
                  } else if (oi === selected) {
                    optClass = 'border-accent-500/50 bg-accent-500/10 text-accent-300'
                  }

                  return (
                    <button
                      key={oi}
                      onClick={() => handleSelect(q.id, oi)}
                      disabled={submitted}
                      className={`w-full rounded-lg border px-4 py-2.5 text-left text-sm font-mono transition ${optClass}`}
                    >
                      {opt}
                    </button>
                  )
                })}
              </div>

              {/* Post-submit feedback */}
              {submitted && (
                <div className="mt-3">
                  <div className="flex items-center gap-2 text-sm">
                    {isCorrect ? (
                      <CheckCircle2 size={16} className="text-accent-500" />
                    ) : (
                      <XCircle size={16} className="text-red-400" />
                    )}
                    <span className={isCorrect ? 'text-accent-400' : 'text-red-300'}>
                      {isCorrect ? 'Correct!' : 'Incorrect'}
                    </span>
                  </div>

                  <button
                    onClick={() => toggleExplanation(q.id)}
                    className="mt-2 flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 transition"
                  >
                    <ChevronDown size={12} className={showExplanations.has(q.id) ? 'rotate-180' : ''} />
                    {showExplanations.has(q.id) ? 'Hide' : 'Show'} explanation
                  </button>

                  {showExplanations.has(q.id) && (
                    <p className="mt-2 rounded-lg bg-surface-950 p-3 text-sm text-zinc-400">
                      {q.explanation}
                    </p>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Submit button */}
      {!submitted && (
        <button
          onClick={handleSubmit}
          disabled={!allAnswered}
          className="mt-6 w-full rounded-lg bg-accent-500 py-3 font-medium text-surface-950 hover:bg-accent-400 transition disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Submit Answers
        </button>
      )}
    </div>
  )
}
