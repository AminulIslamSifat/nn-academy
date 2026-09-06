// Home page — render module sections (each with its track cards) from manifest
const resp = await fetch('/content/manifest.json')
const manifest = await resp.json()

const container = document.getElementById('modules')

function trackCard(track, accent) {
  const firstSlug = track.chapters[0] || ''
  const count = track.chapters.length
  const card = document.createElement('a')
  card.href = firstSlug ? `/chapters/${firstSlug}.html` : '#'
  card.className = 'track-card glass-card animate-in p-5 flex flex-col gap-3 no-underline group'
  card.style.textDecoration = 'none'
  card.innerHTML = `
    <div class="flex items-center justify-between">
      <i data-lucide="${track.icon}" style="width:18px;height:18px;color:${accent};"></i>
      <span class="text-[11px] font-medium px-2 py-0.5 rounded-full" style="color: var(--text-muted); background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle);">
        ${count} ch
      </span>
    </div>
    <div>
      <h3 class="text-sm font-medium leading-snug" style="color: var(--text-primary);">${track.title}</h3>
      <p class="mt-1 text-xs" style="color: var(--text-muted);">${count} chapter${count !== 1 ? 's' : ''}</p>
    </div>
    <div class="mt-auto pt-2 flex items-center gap-1 text-xs font-medium transition" style="color: ${accent};">
      Begin
      <svg class="transition-transform group-hover:translate-x-0.5" width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M6 12l4-4-4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>
  `
  return card
}

function moduleSection(module, index) {
  const accent = module.accent || 'var(--accent)'
  const chapterCount = module.tracks.reduce((n, t) => n + t.chapters.length, 0)
  const firstSlug = module.tracks[0]?.chapters[0] || ''

  const section = document.createElement('section')
  section.className = 'module-section animate-in'
  section.style.animationDelay = `${index * 90}ms`
  section.id = module.id

  // Header row
  const header = document.createElement('div')
  header.className = 'module-header glass-card p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center gap-6'
  header.innerHTML = `
    <div class="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl" style="background: color-mix(in srgb, ${accent} 14%, transparent); border: 1px solid color-mix(in srgb, ${accent} 22%, transparent);">
      <i data-lucide="${module.icon}" style="width:26px;height:26px;color:${accent};"></i>
    </div>
    <div class="flex-1 min-w-0">
      <div class="flex flex-wrap items-center gap-3">
        <h2 class="text-xl font-semibold tracking-tight" style="color: var(--text-primary);">${module.title}</h2>
        <span class="text-[11px] font-medium px-2.5 py-0.5 rounded-full" style="color: ${accent}; background: color-mix(in srgb, ${accent} 12%, transparent); border: 1px solid color-mix(in srgb, ${accent} 20%, transparent);">
          ${module.tagline}
        </span>
      </div>
      <p class="mt-2 text-sm leading-relaxed" style="color: var(--text-secondary);">${module.description}</p>
      <div class="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs" style="color: var(--text-muted);">
        <span class="flex items-center gap-1.5"><i data-lucide="layers" style="width:14px;height:14px;color:${accent};"></i>${module.tracks.length} tracks</span>
        <span class="flex items-center gap-1.5"><i data-lucide="book-open" style="width:14px;height:14px;color:${accent};"></i>${chapterCount} chapters</span>
      </div>
    </div>
    ${firstSlug ? `<a href="/chapters/${firstSlug}.html" class="btn-primary shrink-0 justify-center" style="background: linear-gradient(135deg, ${accent}, color-mix(in srgb, ${accent} 60%, #000));">Start ${module.title}</a>` : ''}
  `

  // Track grid
  const grid = document.createElement('div')
  grid.className = 'grid gap-3 sm:grid-cols-2 lg:grid-cols-3 mt-4'
  for (const track of module.tracks) {
    grid.appendChild(trackCard(track, accent))
  }

  section.appendChild(header)
  section.appendChild(grid)
  return section
}

for (let i = 0; i < manifest.modules.length; i++) {
  container.appendChild(moduleSection(manifest.modules[i], i))
}

// Render lucide icons after dynamic content is inserted
lucide.createIcons()
