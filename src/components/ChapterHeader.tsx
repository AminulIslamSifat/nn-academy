'use client'

import Link from 'next/link'
import { Menu, Clock, Code2, X } from 'lucide-react'
import { useAcademyStore } from '@/lib/store'

interface ChapterHeaderProps {
  title: string
  readTime: number
  codeTime: number
}

export default function ChapterHeader({ title, readTime, codeTime }: ChapterHeaderProps) {
  const { sidebarOpen, setSidebarOpen } = useAcademyStore()

  return (
    <>
      {/* Mobile overlay when sidebar open */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <header className="sticky top-0 z-30 border-b border-zinc-800 bg-surface-950/80 backdrop-blur-sm px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="rounded p-1.5 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 transition lg:hidden"
              aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
            <nav className="flex items-center gap-2 text-sm text-zinc-500" aria-label="Breadcrumb">
              <Link href="/" className="hover:text-zinc-300 transition">Academy</Link>
              <span>/</span>
              <span className="text-zinc-300">{title}</span>
            </nav>
          </div>
          <div className="flex items-center gap-4 text-xs text-zinc-500">
            <span className="flex items-center gap-1">
              <Clock size={12} /> {readTime} min read
            </span>
            <span className="flex items-center gap-1 hidden sm:flex">
              <Code2 size={12} /> {codeTime} min code
            </span>
          </div>
        </div>
      </header>
    </>
  )
}
