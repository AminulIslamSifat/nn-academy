// Service Worker: NumPy NN Academy
// Caches Pyodide wheels + static chapter content for offline use

const CACHE_NAME = 'nna-cache-v1'
const PYODIDE_CACHE = 'nna-pyodide-v1'

// Pyodide CDN assets to cache for offline
const PYODIDE_ASSETS = [
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js',
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.asm.js',
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.asm.wasm',
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide_py.tar',
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.mjs',
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/numpy-1.26.4-cp312-cp312-emscripten_3_1_58_wasm32.whl',
  'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/packages.json',
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(['/'])
    })
  )
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME && key !== PYODIDE_CACHE)
          .map((key) => caches.delete(key))
      )
    })
  )
  self.clients.claim()
})

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url)

  // Cache Pyodide CDN assets aggressively
  if (url.hostname === 'cdn.jsdelivr.net' && url.pathname.includes('/pyodide/')) {
    event.respondWith(
      caches.open(PYODIDE_CACHE).then(async (cache) => {
        const cached = await cache.match(event.request)
        if (cached) return cached

        try {
          const response = await fetch(event.request)
          if (response.ok) {
            cache.put(event.request, response.clone())
          }
          return response
        } catch (err) {
          // Offline and not cached — return a helpful error
          return new Response('Pyodide assets not available offline', {
            status: 503,
            headers: { 'Content-Type': 'text/plain' },
          })
        }
      })
    )
    return
  }

  // Cache-first for same-origin static assets
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached

        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone()
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, clone)
            })
          }
          return response
        })
      })
    )
  }
})
