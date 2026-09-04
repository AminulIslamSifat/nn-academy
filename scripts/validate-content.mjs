#!/usr/bin/env node
// Build-time content validation for NumPy NN Academy
// Checks: slug uniqueness, required frontmatter, quiz answer validity, manifest consistency

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(__dirname, '..')
const CONTENT_DIR = path.join(ROOT, 'src/content/chapters')
const MANIFEST_PATH = path.join(ROOT, 'src/content/manifest.json')

let errors = []
let warnings = []

function error(msg) { errors.push(msg) }
function warn(msg) { warnings.push(msg) }

// ─── 1. Load manifest ───────────────────────────────────────────────────────
let manifest
try {
  manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'))
} catch (e) {
  console.error('❌ FATAL: Cannot read manifest.json:', e.message)
  process.exit(1)
}

const manifestSlugs = new Set()
for (const track of manifest.tracks || []) {
  for (const slug of track.chapters || []) {
    if (manifestSlugs.has(slug)) {
      error(`Duplicate slug in manifest: "${slug}"`)
    }
    manifestSlugs.add(slug)
  }
}

// ─── 2. Validate MDX files ──────────────────────────────────────────────────
const mdxFiles = fs.readdirSync(CONTENT_DIR).filter(f => f.endsWith('.mdx'))
const fileSlugs = new Set()

const REQUIRED_FIELDS = ['title', 'slug', 'description', 'track', 'order', 'read_time', 'code_time']

for (const file of mdxFiles) {
  const filePath = path.join(CONTENT_DIR, file)
  const raw = fs.readFileSync(filePath, 'utf-8')

  // Parse frontmatter
  const fmMatch = raw.match(/^---\n([\s\S]*?)\n---/)
  if (!fmMatch) {
    error(`${file}: Missing frontmatter block`)
    continue
  }

  const fm = {}
  for (const line of fmMatch[1].split('\n')) {
    const colonIdx = line.indexOf(':')
    if (colonIdx > 0) {
      const key = line.slice(0, colonIdx).trim()
      let val = line.slice(colonIdx + 1).trim()
      if (val.startsWith('"') && val.endsWith('"')) val = val.slice(1, -1)
      if (val.startsWith('[') && val.endsWith(']')) {
        val = val.slice(1, -1).split(',').map(s => s.trim().replace(/"/g, '')).filter(Boolean)
      } else if (!isNaN(val) && val !== '') {
        val = Number(val)
      }
      fm[key] = val
    }
  }

  // Check required fields
  for (const field of REQUIRED_FIELDS) {
    if (fm[field] === undefined || fm[field] === '') {
      error(`${file}: Missing required frontmatter field "${field}"`)
    }
  }

  // Slug uniqueness
  const slug = fm.slug
  if (slug) {
    if (fileSlugs.has(slug)) {
      error(`${file}: Duplicate slug "${slug}" (also in another file)`)
    }
    fileSlugs.add(slug)

    const expectedSlug = file.replace('.mdx', '')
    if (slug !== expectedSlug) {
      warn(`${file}: slug "${slug}" doesn't match filename "${expectedSlug}"`)
    }
  }

  if (slug && !manifestSlugs.has(slug)) {
    warn(`${file}: slug "${slug}" not referenced in manifest.json`)
  }

  // Validate quiz correctIndex within bounds
  const quizBlocks = raw.match(/<Quiz[\s\S]*?\/>/g) || []
  for (const block of quizBlocks) {
    const correctIdxMatches = [...block.matchAll(/correctIndex:\s*(\d+)/g)]
    const optionsMatches = [...block.matchAll(/options:\s*\[([\s\S]*?)\]/g)]

    for (let i = 0; i < correctIdxMatches.length; i++) {
      const idx = parseInt(correctIdxMatches[i][1])
      if (i < optionsMatches.length) {
        const optCount = (optionsMatches[i][1].match(/"/g) || []).length / 2
        if (idx >= optCount) {
          error(`${file}: Quiz correctIndex ${idx} >= options count ${optCount}`)
        }
      }
    }
  }

  // Check PyRunner cellId uniqueness
  const cellIds = [...raw.matchAll(/cellId="([^"]+)"/g)].map(m => m[1])
  if (new Set(cellIds).size !== cellIds.length) {
    error(`${file}: Duplicate cellId in PyRunner components`)
  }

  // Validate component references
  const usedComponents = [...raw.matchAll(/<(\w+)[\s\/>]/g)]
    .map(m => m[1])
    .filter(c => /^[A-Z]/.test(c))
  const validComponents = new Set(['PyRunner', 'Quiz', 'Visualizer', 'Editor', 'Callout', 'SplitPane', 'InlineMath', 'BlockMath'])
  for (const comp of usedComponents) {
    if (!validComponents.has(comp)) {
      error(`${file}: Unknown component <${comp}> not in registry`)
    }
  }
}

// ─── 3. Manifest → file existence ───────────────────────────────────────────
for (const slug of manifestSlugs) {
  const filePath = path.join(CONTENT_DIR, `${slug}.mdx`)
  if (!fs.existsSync(filePath)) {
    error(`Manifest references "${slug}" but ${slug}.mdx does not exist`)
  }
}

// ─── 4. Report ──────────────────────────────────────────────────────────────
console.log(`\n📋 Validated ${mdxFiles.length} chapters, ${manifestSlugs.size} manifest entries\n`)

if (warnings.length > 0) {
  console.log(`⚠️  ${warnings.length} warning(s):`)
  for (const w of warnings) console.log(`   • ${w}`)
  console.log()
}

if (errors.length > 0) {
  console.error(`❌ ${errors.length} error(s):`)
  for (const e of errors) console.error(`   • ${e}`)
  console.error('\n🚫 Build blocked. Fix errors above.\n')
  process.exit(1)
} else {
  console.log('✅ All content validation passed.\n')
}
