// MDX Custom Component Registry
// Maps custom JSX tags in MDX content to React components

import dynamic from 'next/dynamic'

const PyRunner = dynamic(() => import('@/components/PyRunner'), {
  ssr: false,
  loading: () => <div className="h-32 animate-pulse rounded-lg bg-surface-800" />,
})

const Quiz = dynamic(() => import('@/components/Quiz'), {
  ssr: false,
  loading: () => <div className="h-64 animate-pulse rounded-xl bg-surface-800" />,
})

const Visualizer = dynamic(() => import('@/components/Visualizer'), {
  ssr: false,
  loading: () => <div className="h-48 animate-pulse rounded-xl bg-surface-800" />,
})

const Editor = dynamic(() => import('@/components/Editor'), {
  ssr: false,
  loading: () => <div className="h-40 animate-pulse rounded-lg bg-surface-800" />,
})

const KaTeX = dynamic(() => import('@/components/mdx/KaTeX'), {
  ssr: false,
  loading: () => <span className="animate-pulse text-zinc-600">...</span>,
})

// Inline math component using KaTeX
function InlineMath({ latex }: { latex: string }) {
  return <KaTeX latex={latex} display={false} />
}

// Block math component
function BlockMath({ latex }: { latex: string }) {
  return <KaTeX latex={latex} display={true} />
}

// Callout / admonition component
function Callout({ type = 'info', title, children }: {
  type?: 'info' | 'warning' | 'tip' | 'danger'
  title?: string
  children: React.ReactNode
}) {
  const styles: Record<string, string> = {
    info: 'border-blue-500/30 bg-blue-500/5',
    warning: 'border-amber-500/30 bg-amber-500/5',
    tip: 'border-accent-500/30 bg-accent-500/5',
    danger: 'border-red-500/30 bg-red-500/5',
  }
  const icons: Record<string, string> = {
    info: 'ℹ️',
    warning: '⚠️',
    tip: '💡',
    danger: '🚨',
  }

  return (
    <div className={`my-4 rounded-lg border p-4 ${styles[type]}`} role="note">
      {title && (
        <div className="mb-2 flex items-center gap-2 font-medium text-zinc-200">
          <span>{icons[type]}</span> {title}
        </div>
      )}
      <div className="text-sm text-zinc-300">{children}</div>
    </div>
  )
}

// Split pane layout for interactive chapters
function SplitPane({ left, right }: { left: React.ReactNode; right: React.ReactNode }) {
  return (
    <div className="grid gap-6 lg:grid-cols-2 my-6">
      <div className="prose-content">{left}</div>
      <div className="interactive-panel">{right}</div>
    </div>
  )
}

export const mdxComponents: Record<string, React.ComponentType<any>> = {
  PyRunner,
  Quiz,
  Visualizer,
  Editor,
  InlineMath,
  BlockMath,
  Callout,
  SplitPane,
}
