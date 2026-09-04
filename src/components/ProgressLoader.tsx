'use client'

import { useEffect } from 'react'
import { useAcademyStore } from '@/lib/store'

// Mounts once in root layout to hydrate progress from IndexedDB
export default function ProgressLoader() {
  const loadProgress = useAcademyStore((s) => s.loadProgress)

  useEffect(() => {
    loadProgress()
  }, [loadProgress])

  return null
}
