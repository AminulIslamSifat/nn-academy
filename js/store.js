// Simple localStorage-backed progress store. No Zustand, no Dexie.

const STORAGE_KEY = 'nna_progress'

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
  } catch { return {} }
}

function save(data) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
}

export function getProgress() {
  return load()
}

export function markCompleted(slug) {
  const data = load()
  data[slug] = { ...data[slug], completed: true, lastVisited: new Date().toISOString() }
  save(data)
  window.dispatchEvent(new CustomEvent('nna-progress', { detail: { slug, completed: true } }))
}

export function saveQuizScore(slug, questionId, score) {
  const data = load()
  if (!data[slug]) data[slug] = { completed: false, quizScores: {}, lastVisited: new Date().toISOString() }
  if (!data[slug].quizScores) data[slug].quizScores = {}
  data[slug].quizScores[questionId] = score
  save(data)
}

export function isCompleted(slug) {
  return !!load()[slug]?.completed
}

export function getCompletedCount() {
  return Object.values(load()).filter(p => p.completed).length
}
