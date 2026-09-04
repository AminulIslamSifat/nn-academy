// Home page — render track cards from manifest
const resp = await fetch('/content/manifest.json')
const manifest = await resp.json()

const grid = document.getElementById('tracks-grid')

for (const track of manifest.tracks) {
  const firstSlug = track.chapters[0] || ''
  const count = track.chapters.length
  const card = document.createElement('a')
  card.href = firstSlug ? `/chapters/${firstSlug}.html` : '#'
  card.className = 'track-card glass-card animate-in p-5 flex flex-col gap-3 no-underline group'
  card.style.textDecoration = 'none'
  card.innerHTML = `
    <div class="flex items-center justify-between">
      <span class="text-xl">${track.icon}</span>
      <span class="text-[11px] font-medium px-2 py-0.5 rounded-full" style="color: var(--text-muted); background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle);">
        ${count} ch
      </span>
    </div>
    <div>
      <h3 class="text-sm font-medium leading-snug" style="color: var(--text-primary);">${track.title}</h3>
      <p class="mt-1 text-xs" style="color: var(--text-muted);">${count} chapter${count !== 1 ? 's' : ''}</p>
    </div>
    <div class="mt-auto pt-2 flex items-center gap-1 text-xs font-medium transition" style="color: var(--accent);">
      Begin
      <svg class="transition-transform group-hover:translate-x-0.5" width="12" height="12" viewBox="0 0 16 16" fill="none"><path d="M6 12l4-4-4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>
  `
  grid.appendChild(card)
}
