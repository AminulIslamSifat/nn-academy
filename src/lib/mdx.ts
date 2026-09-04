import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

// Types
export interface ChapterMeta {
  title: string
  slug: string
  description: string
  track: string
  order: number
  readTime: number
  codeTime: number
  executionTimeout: number
  prerequisites: string[]
}

export interface TrackChapter {
  slug: string
  title: string
}

export interface Track {
  id: string
  title: string
  icon: string
  chapters: TrackChapter[]
}

export interface Manifest {
  version: number
  totalChapters: number
  tracks: Track[]
}

interface ChapterData {
  meta: ChapterMeta
  content: string
}

const CONTENT_DIR = path.join(process.cwd(), 'src/content/chapters')
const MANIFEST_PATH = path.join(process.cwd(), 'src/content/manifest.json')

// Cache
let manifestCache: Manifest | null = null
let chapterCache: Map<string, ChapterData> = new Map()

export function getManifest(): Manifest {
  if (manifestCache) return manifestCache

  const raw = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'))

  // Resolve chapter titles from frontmatter
  const tracks: Track[] = raw.tracks.map((track: any) => ({
    id: track.id,
    title: track.title,
    icon: track.icon,
    chapters: (track.chapters || []).map((slug: string) => {
      const chapter = getChapterBySlug(slug)
      return {
        slug,
        title: chapter?.meta.title || slug,
      }
    }),
  }))

  const totalChapters = tracks.reduce((sum, t) => sum + t.chapters.length, 0)

  manifestCache = { version: raw.version, totalChapters, tracks }
  return manifestCache
}

export function getChapterBySlug(slug: string): ChapterData | null {
  if (chapterCache.has(slug)) {
    return chapterCache.get(slug)!
  }

  const filePath = path.join(CONTENT_DIR, `${slug}.mdx`)
  if (!fs.existsSync(filePath)) return null

  const raw = fs.readFileSync(filePath, 'utf-8')
  const { data, content } = matter(raw)

  const meta: ChapterMeta = {
    title: data.title || slug,
    slug: data.slug || slug,
    description: data.description || '',
    track: data.track || '',
    order: data.order || 0,
    readTime: data.read_time || 10,
    codeTime: data.code_time || 5,
    executionTimeout: data.execution_timeout || 5,
    prerequisites: data.prerequisites || [],
  }

  const chapter: ChapterData = { meta, content }
  chapterCache.set(slug, chapter)
  return chapter
}

export function getAllSlugs(): string[] {
  const manifest = getManifest()
  const slugs: string[] = []
  for (const track of manifest.tracks) {
    for (const ch of track.chapters) {
      slugs.push(ch.slug)
    }
  }
  return slugs
}

export function getTrackForSlug(slug: string): Track | undefined {
  const manifest = getManifest()
  return manifest.tracks.find((t) =>
    t.chapters.some((c) => c.slug === slug)
  )
}
