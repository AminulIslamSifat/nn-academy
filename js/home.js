// Home page — render track cards from manifest
const resp = await fetch('/content/manifest.json')
const manifest = await resp.json()

const grid = document.getElementById('tracks-grid')

for (const track of manifest.tracks) {
  const firstSlug = track.chapters[0] || ''
  const card = document.createElement('div')
  card.className = 'group rounded-xl border border-zinc-800 bg-surface-900/50 p-6 transition hover:border-accent-500/30 hover:bg-surface-900'
  card.innerHTML = `
    <div class="mb-3 text-2xl">${track.icon}</div>
    <h3 class="font-medium text-zinc-200 group-hover:text-accent-300 transition">${track.title}</h3>
    <p class="mt-1 text-sm text-zinc-500">${track.chapters.length} chapter${track.chapters.length !== 1 ? 's' : ''}</p>
    ${firstSlug ? `<a href="/chapters/${firstSlug}.html" class="mt-4 inline-flex items-center gap-1 text-sm text-accent-400 hover:text-accent-300 transition">Begin →</a>` : ''}
  `
  grid.appendChild(card)
}
