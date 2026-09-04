import Link from 'next/link'
import { getManifest } from '@/lib/mdx'
import { BookOpen, Terminal, Brain, Zap, ArrowRight } from 'lucide-react'

export default function HomePage() {
  const manifest = getManifest()

  return (
    <div className="min-h-screen bg-surface-950">
      {/* Hero */}
      <header className="border-b border-zinc-800/50">
        <div className="mx-auto max-w-5xl px-6 py-20 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-accent-500/20 bg-accent-500/5 px-4 py-1.5 text-sm text-accent-400">
            <Zap size={14} />
            Pure NumPy. Zero frameworks.
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-zinc-50 sm:text-5xl">
            NumPy Neural Network Academy
          </h1>
          <p className="mt-4 text-lg text-zinc-400 max-w-2xl mx-auto">
            Master deep learning from the ground up. Interactive chapters with
            in-browser Python execution, tensor visualizers, and rigorous assessments.
          </p>
          <div className="mt-8 flex items-center justify-center gap-4">
            <Link
              href="/chapters/001-arrays-and-shapes"
              className="inline-flex items-center gap-2 rounded-lg bg-accent-500 px-6 py-3 font-medium text-surface-950 hover:bg-accent-400 transition"
            >
              Start Learning <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </header>

      {/* Tracks grid */}
      <main className="mx-auto max-w-5xl px-6 py-16">
        <h2 className="text-xl font-semibold text-zinc-200 mb-8">Curriculum Tracks</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {manifest.tracks.map((track) => (
            <div
              key={track.id}
              className="group rounded-xl border border-zinc-800 bg-surface-900/50 p-6 transition hover:border-accent-500/30 hover:bg-surface-900"
            >
              <div className="mb-3 text-2xl">{track.icon}</div>
              <h3 className="font-medium text-zinc-200 group-hover:text-accent-300 transition">
                {track.title}
              </h3>
              <p className="mt-1 text-sm text-zinc-500">
                {track.chapters.length} chapter{track.chapters.length !== 1 ? 's' : ''}
              </p>
              {track.chapters.length > 0 && (
                <Link
                  href={`/chapters/${track.chapters[0].slug}`}
                  className="mt-4 inline-flex items-center gap-1 text-sm text-accent-400 hover:text-accent-300 transition"
                >
                  Begin <ArrowRight size={12} />
                </Link>
              )}
            </div>
          ))}
        </div>

        {/* Features */}
        <div className="mt-20 grid gap-8 sm:grid-cols-3">
          <div className="text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-500/10">
              <Terminal size={22} className="text-accent-400" />
            </div>
            <h3 className="font-medium text-zinc-200">Live Python</h3>
            <p className="mt-1 text-sm text-zinc-500">Execute NumPy code in-browser via WebAssembly. No server needed.</p>
          </div>
          <div className="text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-500/10">
              <Brain size={22} className="text-accent-400" />
            </div>
            <h3 className="font-medium text-zinc-200">From Scratch</h3>
            <p className="mt-1 text-sm text-zinc-500">Build backprop, optimizers, and architectures with raw array math.</p>
          </div>
          <div className="text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-500/10">
              <BookOpen size={22} className="text-accent-400" />
            </div>
            <h3 className="font-medium text-zinc-200">500+ Modules</h3>
            <p className="mt-1 text-sm text-zinc-500">Structured curriculum from arrays to advanced architectures.</p>
          </div>
        </div>
      </main>

      <footer className="border-t border-zinc-800/50 py-8 text-center text-sm text-zinc-600">
        NumPy Neural Network Academy — Learn by building.
      </footer>
    </div>
  )
}
