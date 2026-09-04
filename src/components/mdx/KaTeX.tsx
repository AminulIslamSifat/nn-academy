'use client'

import { useMemo } from 'react'
import katex from 'katex'

interface KaTeXProps {
  latex: string
  display?: boolean
}

export default function KaTeX({ latex, display = false }: KaTeXProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        displayMode: display,
        throwOnError: false,
        output: 'html',
      })
    } catch (err) {
      return `<span class="text-red-400 font-mono text-sm">${latex}</span>`
    }
  }, [latex, display])

  if (display) {
    return (
      <div
        className="katex-display my-6 overflow-x-auto"
        role="math"
        aria-label={`Equation: ${latex}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }

  return (
    <span
      className="katex-inline"
      aria-label={`Math: ${latex}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
