import { create } from 'zustand'
import Dexie, { type EntityTable } from 'dexie'

// ─── Dexie Schema (versioned) ───────────────────────────────────────────────

interface ProgressRecord {
  slug: string
  completed: boolean
  quizScores: Record<string, number>
  lastVisited: string
  codeAttempts: number
}

interface CodeCellRecord {
  cellId: string
  code: string
  lastRun: string
}

const db = new Dexie('NNAcademyDB') as Dexie & {
  progress: EntityTable<ProgressRecord, 'slug'>
  codeCells: EntityTable<CodeCellRecord, 'cellId'>
}

db.version(1).stores({
  progress: 'slug, completed, lastVisited',
  codeCells: 'cellId, lastRun',
})

// ─── Zustand Store ──────────────────────────────────────────────────────────

interface ChapterProgress {
  completed: boolean
  quizScores: Record<string, number>
  lastVisited: string
}

interface AcademyState {
  progress: Record<string, ChapterProgress>
  sidebarOpen: boolean
  activeCellId: string | null

  // Actions
  setSidebarOpen: (open: boolean) => void
  markCompleted: (slug: string) => void
  saveQuizScore: (slug: string, questionId: string, score: number) => void
  setActiveCell: (cellId: string | null) => void
  loadProgress: () => Promise<void>
  exportProgress: () => string
  importProgress: (json: string) => Promise<void>
}

export const useAcademyStore = create<AcademyState>((set, get) => ({
  progress: {},
  sidebarOpen: true,
  activeCellId: null,

  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  markCompleted: async (slug) => {
    const entry: ProgressRecord = {
      slug,
      completed: true,
      quizScores: get().progress[slug]?.quizScores || {},
      lastVisited: new Date().toISOString(),
      codeAttempts: 0,
    }
    await db.progress.put(entry)
    set((state) => ({
      progress: {
        ...state.progress,
        [slug]: {
          completed: true,
          quizScores: entry.quizScores,
          lastVisited: entry.lastVisited,
        },
      },
    }))
  },

  saveQuizScore: async (slug, questionId, score) => {
    const existing = get().progress[slug] || {
      completed: false,
      quizScores: {},
      lastVisited: new Date().toISOString(),
    }
    const quizScores = { ...existing.quizScores, [questionId]: score }

    const entry: ProgressRecord = {
      slug,
      completed: existing.completed,
      quizScores,
      lastVisited: new Date().toISOString(),
      codeAttempts: 0,
    }
    await db.progress.put(entry)
    set((state) => ({
      progress: {
        ...state.progress,
        [slug]: { ...existing, quizScores },
      },
    }))
  },

  setActiveCell: (cellId) => set({ activeCellId: cellId }),

  loadProgress: async () => {
    try {
      const records = await db.progress.toArray()
      const progress: Record<string, ChapterProgress> = {}
      for (const r of records) {
        progress[r.slug] = {
          completed: r.completed,
          quizScores: r.quizScores,
          lastVisited: r.lastVisited,
        }
      }
      set({ progress })
    } catch (err) {
      console.error('Failed to load progress:', err)
    }
  },

  exportProgress: () => {
    const state = get()
    return JSON.stringify({
      version: 1,
      exportedAt: new Date().toISOString(),
      progress: state.progress,
    }, null, 2)
  },

  importProgress: async (json) => {
    try {
      const data = JSON.parse(json)
      if (data.version !== 1) throw new Error('Unsupported schema version')
      const entries: ProgressRecord[] = Object.entries(data.progress).map(
        ([slug, p]: [string, any]) => ({
          slug,
          completed: p.completed,
          quizScores: p.quizScores || {},
          lastVisited: p.lastVisited || new Date().toISOString(),
          codeAttempts: 0,
        })
      )
      await db.progress.bulkPut(entries)
      set({ progress: data.progress })
    } catch (err) {
      console.error('Import failed:', err)
      throw err
    }
  },
}))
