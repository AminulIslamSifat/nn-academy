// Chapter page — mounts sidebar, widgets, and handles progress
import { getProgress, getCompletedCount, markCompleted } from './store.js'
import { mountPyRunner, mountQuiz, mountVisualizer, mountKaTeX, mountCallout } from './widgets.js'

let manifest = { tracks: [], totalChapters: 65 }
try {
  const resp = await fetch('/content/manifest.json')
  manifest = await resp.json()
} catch (e) {
  console.error('Failed to load manifest:', e)
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────

const currentSlug = document.body.dataset.slug || ''
const totalChapters = manifest.totalChapters || 65

function renderSidebar() {
  const nav = document.getElementById('sidebar-nav')
  const completedCount = getCompletedCount()
  const progress = getProgress()

  document.getElementById('progress-text').textContent = `${completedCount}/${totalChapters}`
  document.getElementById('progress-bar').style.width = `${(completedCount / totalChapters) * 100}%`

  nav.innerHTML = ''
  for (const track of manifest.tracks) {
    const trackDiv = document.createElement('div')
    const trackCompleted = track.chapters.filter(s => progress[s]?.completed).length

    const btn = document.createElement('button')
    btn.className = 'sidebar-item flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800/50 transition'
    btn.setAttribute('aria-expanded', 'true')
    btn.innerHTML = `<span class="chevron text-zinc-500">▾</span><span class="mr-1">${track.icon}</span><span class="flex-1 text-left">${track.title}</span><span class="text-xs text-zinc-600">${trackCompleted}/${track.chapters.length}</span>`

    const chaptersDiv = document.createElement('div')
    chaptersDiv.className = 'ml-6 space-y-0.5 mt-1'

    for (const slug of track.chapters) {
      const isActive = slug === currentSlug
      const isDone = progress[slug]?.completed
      // Derive a readable title from the slug
      const title = slug.replace(/^\d+-/, '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
      const link = document.createElement('a')
      link.href = `/chapters/${slug}.html`
      link.className = `sidebar-item flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition ${
        isActive ? 'bg-accent-500/10 text-accent-300 font-medium' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30'
      }`
      link.innerHTML = `
        ${isDone
          ? '<span style="color:#10b981;flex-shrink:0">✓</span>'
          : '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;border:1px solid #3f3f46;flex-shrink:0"></span>'}
        <span class="truncate">${title}</span>
      `
      chaptersDiv.appendChild(link)
    }

    let expanded = true
    btn.addEventListener('click', () => {
      expanded = !expanded
      chaptersDiv.style.display = expanded ? 'block' : 'none'
      btn.querySelector('.chevron').textContent = expanded ? '▾' : '▸'
      btn.setAttribute('aria-expanded', String(expanded))
    })

    trackDiv.appendChild(btn)
    trackDiv.appendChild(chaptersDiv)
    nav.appendChild(trackDiv)
  }
}

renderSidebar()

// Listen for progress updates and re-render sidebar
window.addEventListener('nna-progress', () => renderSidebar())

// Sidebar toggle (desktop + mobile)
const mobileBtn = document.getElementById('mobile-menu-btn')
const sidebarCloseBtn = document.getElementById('sidebar-close')
const sidebar = document.getElementById('sidebar')
const overlay = document.getElementById('sidebar-overlay')
let sidebarOpen = window.innerWidth >= 1024 // open on desktop by default

function updateSidebar() {
  if (sidebarOpen) {
    sidebar.classList.remove('hidden')
    if (window.innerWidth < 1024) overlay?.classList.remove('hidden')
    else overlay?.classList.add('hidden')
  } else {
    sidebar.classList.add('hidden')
    overlay?.classList.add('hidden')
  }
}

updateSidebar()

mobileBtn?.addEventListener('click', () => { sidebarOpen = !sidebarOpen; updateSidebar() })
sidebarCloseBtn?.addEventListener('click', () => { sidebarOpen = false; updateSidebar() })
overlay?.addEventListener('click', () => { sidebarOpen = false; updateSidebar() })

// ─── Mount Widgets ───────────────────────────────────────────────────────────

document.querySelectorAll('[data-widget="pyrunner"]').forEach(el => mountPyRunner(el))
document.querySelectorAll('[data-widget="quiz"]').forEach(el => mountQuiz(el))
document.querySelectorAll('[data-widget="visualizer"]').forEach(el => mountVisualizer(el))
document.querySelectorAll('[data-widget="katex"]').forEach(el => mountKaTeX(el))
document.querySelectorAll('[data-widget="callout"]').forEach(el => mountCallout(el))
