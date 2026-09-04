'use client'

import { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, RotateCcw, ChevronRight } from 'lucide-react'

type VizMode = 'broadcasting' | 'matmul' | 'reshape'

interface VisualizerProps {
  mode: VizMode
  shapeA?: number[]
  shapeB?: number[]
  title?: string
}

// Compute broadcasting result
function computeBroadcast(shapeA: number[], shapeB: number[]): { result: number[]; steps: string[] } {
  const maxDims = Math.max(shapeA.length, shapeB.length)
  const padA = Array(maxDims - shapeA.length).fill(1).concat(shapeA)
  const padB = Array(maxDims - shapeB.length).fill(1).concat(shapeB)
  const result: number[] = []
  const steps: string[] = []

  for (let i = 0; i < maxDims; i++) {
    const a = padA[i]
    const b = padB[i]
    if (a === b) {
      result.push(a)
      steps.push(`dim ${i}: ${a} == ${b} → ${a}`)
    } else if (a === 1) {
      result.push(b)
      steps.push(`dim ${i}: ${a} broadcasts → ${b}`)
    } else if (b === 1) {
      result.push(a)
      steps.push(`dim ${i}: ${b} broadcasts → ${a}`)
    } else {
      steps.push(`dim ${i}: INCOMPATIBLE (${a} vs ${b})`)
      return { result: [], steps }
    }
  }
  return { result, steps }
}

// Compute matmul result
function computeMatmul(shapeA: number[], shapeB: number[]): { result: number[]; steps: string[] } {
  const steps: string[] = []
  if (shapeA.length < 2 || shapeB.length < 2) {
    steps.push('Both arrays must be at least 2D for matmul')
    return { result: [], steps }
  }
  const [m, k1] = [shapeA[shapeA.length - 2], shapeA[shapeA.length - 1]]
  const [k2, n] = [shapeB[shapeB.length - 2], shapeB[shapeB.length - 1]]

  if (k1 !== k2) {
    steps.push(`Inner dimensions don't match: ${k1} ≠ ${k2}`)
    return { result: [], steps }
  }

  steps.push(`A: (..., ${m}, ${k1}) × B: (..., ${k2}, ${n})`)
  steps.push(`Inner dim ${k1} == ${k2} ✓`)
  steps.push(`Result: (..., ${m}, ${n})`)
  return { result: [m, n], steps }
}

function ShapeBox({ shape, label, color }: { shape: number[]; label: string; color: string }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-xs font-medium text-zinc-500">{label}</span>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`rounded-lg border-2 ${color} px-4 py-3 font-mono text-sm`}
      >
        ({shape.join(', ')})
      </motion.div>
    </div>
  )
}

export default function Visualizer({ mode, shapeA = [3, 1], shapeB = [1, 4], title }: VisualizerProps) {
  const [inputA, setInputA] = useState(shapeA.join(', '))
  const [inputB, setInputB] = useState(shapeB.join(', '))
  const [steps, setSteps] = useState<string[]>([])
  const [result, setResult] = useState<number[] | null>(null)
  const [currentStep, setCurrentStep] = useState(-1)
  const [animating, setAnimating] = useState(false)

  const parseShape = (s: string): number[] => {
    return s.split(',').map((x) => parseInt(x.trim())).filter((x) => !isNaN(x) && x > 0)
  }

  const runVisualization = () => {
    const a = parseShape(inputA)
    const b = parseShape(inputB)
    if (a.length === 0 || b.length === 0) return

    let res: { result: number[]; steps: string[] }
    if (mode === 'broadcasting') {
      res = computeBroadcast(a, b)
    } else if (mode === 'matmul') {
      res = computeMatmul(a, b)
    } else {
      res = { result: a, steps: ['Reshape preserves total elements'] }
    }

    setSteps(res.steps)
    setResult(res.result)
    setCurrentStep(-1)
    setAnimating(true)

    // Animate steps
    res.steps.forEach((_, i) => {
      setTimeout(() => setCurrentStep(i), (i + 1) * 600)
    })
    setTimeout(() => setAnimating(false), (res.steps.length + 1) * 600)
  }

  const reset = () => {
    setSteps([])
    setResult(null)
    setCurrentStep(-1)
    setAnimating(false)
  }

  return (
    <div className="my-6 rounded-xl border border-zinc-800 bg-surface-900/30 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-zinc-300">
          {title || `${mode === 'broadcasting' ? 'Broadcasting' : mode === 'matmul' ? 'Matrix Multiply' : 'Reshape'} Visualizer`}
        </h3>
        <div className="flex gap-2">
          <button
            onClick={reset}
            className="rounded p-1.5 text-zinc-500 hover:text-zinc-300 transition"
            aria-label="Reset visualization"
          >
            <RotateCcw size={14} />
          </button>
          <button
            onClick={runVisualization}
            disabled={animating}
            className="flex items-center gap-1.5 rounded-lg bg-accent-500/10 px-3 py-1.5 text-xs font-medium text-accent-400 hover:bg-accent-500/20 transition disabled:opacity-50"
            aria-label="Run visualization"
          >
            <Play size={12} /> Animate
          </button>
        </div>
      </div>

      {/* Shape inputs */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500 font-mono">A =</label>
          <input
            type="text"
            value={inputA}
            onChange={(e) => setInputA(e.target.value)}
            className="w-24 rounded-lg border border-zinc-700 bg-surface-950 px-3 py-1.5 font-mono text-sm text-zinc-200 outline-none focus:border-accent-500/50"
            aria-label="Shape A dimensions (comma-separated)"
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500 font-mono">B =</label>
          <input
            type="text"
            value={inputB}
            onChange={(e) => setInputB(e.target.value)}
            className="w-24 rounded-lg border border-zinc-700 bg-surface-950 px-3 py-1.5 font-mono text-sm text-zinc-200 outline-none focus:border-accent-500/50"
            aria-label="Shape B dimensions (comma-separated)"
          />
        </div>
      </div>

      {/* Visualization area */}
      <div className="min-h-[120px] rounded-lg border border-zinc-800/50 bg-surface-950/50 p-4">
        {steps.length === 0 ? (
          <p className="text-center text-sm text-zinc-600 py-4">
            Enter shapes and click Animate to see the {mode} operation step-by-step.
          </p>
        ) : (
          <div className="space-y-2">
            <AnimatePresence>
              {steps.slice(0, currentStep + 1).map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-2 text-sm"
                >
                  <ChevronRight size={12} className="text-accent-400 shrink-0" />
                  <span className="font-mono text-zinc-300">{step}</span>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Result shape */}
            {result && result.length > 0 && currentStep >= steps.length - 1 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="mt-4 flex items-center justify-center"
              >
                <div className="rounded-lg border-2 border-accent-500/50 bg-accent-500/5 px-6 py-3">
                  <span className="text-xs text-zinc-500 block text-center mb-1">Result Shape</span>
                  <span className="font-mono text-lg text-accent-300">({result.join(', ')})</span>
                </div>
              </motion.div>
            )}

            {result && result.length === 0 && currentStep >= steps.length - 1 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 text-center text-sm text-red-400"
              >
                ❌ Incompatible shapes
              </motion.div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
