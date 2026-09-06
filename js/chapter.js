// Chapter page — mounts sidebar, widgets, and handles progress
import { getProgress, getCompletedCount, markCompleted } from './store.js'
import { mountPyRunner, mountQuiz, mountVisualizer, mountKaTeX, mountCallout } from './widgets.js'

let manifest = { modules: [], totalChapters: 88 }
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

  // Scope the sidebar to the module that contains the current chapter,
  // so you only see the tracks relevant to the module you're studying.
  const allModules = manifest.modules || []
  const activeModule = allModules.find(m => m.tracks.some(t => t.chapters.includes(currentSlug))) || allModules[0]
  const scopedModules = activeModule ? [activeModule] : []

  const moduleChapters = scopedModules.flatMap(m => m.tracks.flatMap(t => t.chapters))
  const moduleCompleted = moduleChapters.filter(s => progress[s]?.completed).length
  document.getElementById('progress-text').textContent = `${moduleCompleted}/${moduleChapters.length}`
  document.getElementById('progress-bar').style.width = `${(moduleCompleted / Math.max(moduleChapters.length, 1)) * 100}%`

  nav.innerHTML = ''
  for (const module of scopedModules) {
    const accent = module.accent || 'var(--accent)'

    // Module heading
    const moduleHeading = document.createElement('div')
    moduleHeading.className = 'sidebar-module-heading'
    moduleHeading.innerHTML = `<i data-lucide="${module.icon}" style="width:13px;height:13px;color:${accent};"></i><span>${module.title}</span>`
    nav.appendChild(moduleHeading)

    for (const track of module.tracks) {
      const trackDiv = document.createElement('div')
      const trackCompleted = track.chapters.filter(s => progress[s]?.completed).length

      const btn = document.createElement('button')
      btn.className = 'sidebar-track-btn'
      btn.setAttribute('aria-expanded', 'true')
      btn.innerHTML = `<span class="chevron" style="color: var(--text-muted); font-size: 0.7rem;">▾</span><i data-lucide="${track.icon}" style="width:14px;height:14px;color:${accent};"></i><span class="flex-1 text-left">${track.title}</span><span style="font-size: 0.7rem; color: var(--text-muted);">${trackCompleted}/${track.chapters.length}</span>`

      const chaptersDiv = document.createElement('div')
      chaptersDiv.className = 'ml-6 space-y-0.5 mt-1'

      for (const slug of track.chapters) {
        const isActive = slug === currentSlug
        const isDone = progress[slug]?.completed
        // Derive a readable title from the slug
        const title = slug.replace(/^\d+-/, '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
        const link = document.createElement('a')
        link.href = `/chapters/${slug}.html`
        link.className = `sidebar-link ${isActive ? 'active' : ''}`
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

  // Link to switch back to the module overview
  const switchLink = document.createElement('a')
  switchLink.href = '/'
  switchLink.className = 'sidebar-link'
  switchLink.style.marginTop = '0.75rem'
  switchLink.innerHTML = '<span style="flex-shrink:0">←</span><span class="truncate">All modules</span>'
  nav.appendChild(switchLink)
}

renderSidebar()
lucide.createIcons()

// Listen for progress updates and re-render sidebar
window.addEventListener('nna-progress', () => { renderSidebar(); lucide.createIcons() })

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
