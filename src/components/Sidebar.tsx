'use client'

import { useState, useMemo, useCallback } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ChevronRight, ChevronDown, CheckCircle2, BookOpen, X } from 'lucide-react'
import { useAcademyStore } from '@/lib/store'
import type { Manifest } from '@/lib/mdx'

interface SidebarProps {
  manifest: Manifest
}

export default function Sidebar({ manifest }: SidebarProps) {
  const pathname = usePathname()
  const [expandedTracks, setExpandedTracks] = useState<Set<string>>(
    () => new Set(manifest.tracks.map((t) => t.id))
  )
  const { progress, sidebarOpen, setSidebarOpen } = useAcademyStore()

  const toggleTrack = useCallback((trackId: string) => {
    setExpandedTracks((prev) => {
      const next = new Set(prev)
      if (next.has(trackId)) next.delete(trackId)
      else next.add(trackId)
      return next
    })
  }, [])

  const totalCompleted = useMemo(
    () => Object.values(progress).filter((p) => p.completed).length,
    [progress]
  )

  if (!sidebarOpen) return null

  return (
    <aside
      className="fixed left-0 top-0 z-40 h-screen w-72 border-r border-zinc-800 bg-surface-950 overflow-y-auto"
      aria-label="Chapter navigation"
    >
      {/* Header */}
      <div className="sticky top-0 z-10 border-b border-zinc-800 bg-surface-950 px-4 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 font-semibold text-zinc-100">
            <BookOpen size={18} className="text-accent-400" />
            <span className="text-sm">NN Academy</span>
          </Link>
          <button
            onClick={() => setSidebarOpen(false)}
            className="rounded p-1 text-zinc-500 hover:text-white transition lg:hidden"
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>
        {/* Progress bar */}
        <div className="mt-3">
          <div className="flex justify-between text-xs text-zinc-500 mb-1">
            <span>Progress</span>
            <span>{totalCompleted}/{manifest.totalChapters}</span>
          </div>
          <div className="h-1.5 rounded-full bg-zinc-800">
            <div
              className="h-1.5 rounded-full bg-accent-500 transition-all duration-500"
              style={{
                width: `${manifest.totalChapters > 0 ? (totalCompleted / manifest.totalChapters) * 100 : 0}%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Track tree */}
      <nav className="px-3 py-4 space-y-1">
        {manifest.tracks.map((track) => {
          const expanded = expandedTracks.has(track.id)
          const trackCompleted = track.chapters.filter(
            (ch) => progress[ch.slug]?.completed
          ).length

          return (
            <div key={track.id}>
              <button
                onClick={() => toggleTrack(track.id)}
                className="sidebar-item flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800/50 transition"
                aria-expanded={expanded}
              >
                {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                <span className="mr-1">{track.icon}</span>
                <span className="flex-1 text-left">{track.title}</span>
                <span className="text-xs text-zinc-600">
                  {trackCompleted}/{track.chapters.length}
                </span>
              </button>

              {expanded && (
                <div className="ml-6 space-y-0.5 mt-1">
                  {track.chapters.map((chapter) => {
                    const isActive = pathname === `/chapters/${chapter.slug}`
                    const isCompleted = progress[chapter.slug]?.completed

                    return (
                      <Link
                        key={chapter.slug}
                        href={`/chapters/${chapter.slug}`}
                        className={`sidebar-item flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition ${
                          isActive
                            ? 'bg-accent-500/10 text-accent-300 font-medium'
                            : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30'
                        }`}
                        aria-current={isActive ? 'page' : undefined}
                      >
                        {isCompleted ? (
                          <CheckCircle2 size={14} className="text-accent-500 shrink-0" />
                        ) : (
                          <span className="inline-block h-3.5 w-3.5 rounded-full border border-zinc-700 shrink-0" />
                        )}
                        <span className="truncate">{chapter.title}</span>
                      </Link>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
