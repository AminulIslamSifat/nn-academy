'use client'

import { useState, useCallback, useRef } from 'react'
import { Play, RotateCcw, Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { pyodide } from '@/lib/pyodide'

interface PyRunnerProps {
  cellId: string
  defaultCode: string
  timeout?: number
  title?: string
}

type RunState = 'idle' | 'loading' | 'running' | 'success' | 'error'

export default function PyRunner({ cellId, defaultCode, timeout = 5, title }: PyRunnerProps) {
  const [code, setCode] = useState(defaultCode)
  const [output, setOutput] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [state, setState] = useState<RunState>('idle')
  const editorRef = useRef<HTMLTextAreaElement>(null)

  const run = useCallback(async () => {
    setState('loading')
    setOutput(null)
    setError(null)

    try {
      setState('running')
      const result = await pyodide.execute(code, timeout)
      setOutput(result)
      setState('success')
    } catch (err: any) {
      setError(err.message || String(err))
      setState('error')
    }
  }, [code, timeout])

  const reset = useCallback(() => {
    setCode(defaultCode)
    setOutput(null)
    setError(null)
    setState('idle')
  }, [defaultCode])

  return (
    <div className="my-6 rounded-xl border border-zinc-800 bg-surface-900/50 overflow-hidden w-full max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-500/60" />
            <span className="h-2.5 w-2.5 rounded-full bg-yellow-500/60" />
            <span className="h-2.5 w-2.5 rounded-full bg-green-500/60" />
          </div>
          <span className="ml-2 text-sm font-mono text-zinc-500">
            {title || cellId}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={reset}
            className="rounded p-1.5 text-zinc-500 hover:text-zinc-300 transition"
            title="Reset code"
            aria-label="Reset code"
          >
              <RotateCcw size={14} />
          </button>
          <button
            onClick={run}
            disabled={state === 'loading' || state === 'running'}
            className="flex items-center gap-1.5 rounded-lg bg-accent-500/10 px-4 py-2 text-sm font-medium text-accent-400 hover:bg-accent-500/20 transition disabled:opacity-50"
            aria-label="Run code"
          >
            {state === 'loading' || state === 'running' ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Play size={14} />
            )}
            Run
          </button>
        </div>
      </div>

      {/* Code input */}
      <textarea
        ref={editorRef}
        value={code}
        onChange={(e) => setCode(e.target.value)}
        className="w-full resize-y bg-transparent px-5 py-4 font-mono text-sm text-zinc-200 outline-none min-h-[160px] leading-relaxed border-b border-zinc-800/50"
        spellCheck={false}
        aria-label={`Python code editor for ${title || cellId}`}
      />

      {/* Output area */}
      {(output !== null || error !== null) && (
        <div className="px-5 py-4">
          {error ? (
            <div className="flex items-start gap-2 rounded-lg bg-red-500/5 border border-red-500/20 p-3">
              <AlertTriangle size={16} className="text-red-400 mt-0.5 shrink-0" />
              <pre className="text-sm text-red-300 whitespace-pre-wrap font-mono leading-relaxed">{error}</pre>
            </div>
          ) : (
            <div className="flex items-start gap-2">
              <CheckCircle2 size={16} className="text-accent-500 mt-0.5 shrink-0" />
              <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed">{output}</pre>
            </div>
          )}
        </div>
      )}

      {/* Loading state */}
      {state === 'loading' && (
        <div className="px-5 pb-4 text-sm text-zinc-500 flex items-center gap-2">
          <Loader2 size={12} className="animate-spin" />
          Initializing Python runtime...
        </div>
      )}
    </div>
  )
}
