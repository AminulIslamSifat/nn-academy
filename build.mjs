#!/usr/bin/env node
// Build script: converts MDX chapters → static HTML pages.
// No Next.js, no React, no MDX compiler. Just string processing + marked.

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const CONTENT_DIR = path.join(__dirname, 'content', 'chapters')
const MANIFEST_PATH = path.join(__dirname, 'content', 'manifest.json')
const TEMPLATE_PATH = path.join(__dirname, 'chapter-template.html')
const OUT_DIR = path.join(__dirname, 'chapters')

// Ensure output dir exists
fs.mkdirSync(OUT_DIR, { recursive: true })

const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8'))
const template = fs.readFileSync(TEMPLATE_PATH, 'utf-8')

// Build slug list in order (modules -> tracks -> chapters)
const allSlugs = []
for (const module of manifest.modules) {
  for (const track of module.tracks) {
    for (const slug of track.chapters) {
      allSlugs.push(slug)
    }
  }
}

// Parse frontmatter manually (no gray-matter needed)
function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) return { meta: {}, content: raw }

  const yamlBlock = match[1]
  const content = match[2]
  const meta = {}

  for (const line of yamlBlock.split('\n')) {
    const colonIdx = line.indexOf(':')
    if (colonIdx === -1) continue
    const key = line.slice(0, colonIdx).trim()
    let val = line.slice(colonIdx + 1).trim()
    // Strip quotes
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1)
    }
    // Parse arrays like [a, b, c]
    if (val.startsWith('[') && val.endsWith(']')) {
      val = val.slice(1, -1).split(',').map(s => s.trim().replace(/^["']|["']$/g, ''))
    }
    // Parse numbers
    else if (/^\d+$/.test(val)) {
      val = parseInt(val, 10)
    }
    meta[key] = val
  }

  return { meta, content }
}

// Convert MDX custom components to data-attribute HTML widgets
function convertMdxToWidgets(md) {
  // <PyRunner cellId="..." defaultCode={`...`} timeout={5} title="..." />
  md = md.replace(/<PyRunner\s+([^>]*?)\/>/gs, (_, attrs) => {
    const cellId = extractAttr(attrs, 'cellId') || ''
    const timeout = extractAttr(attrs, 'timeout') || '5'
    const title = extractAttr(attrs, 'title') || ''
    // Handle template literal defaultCode={`...`}
    const codeMatch = attrs.match(/defaultCode=\{`([\s\S]*?)`\}/)
    const defaultCode = codeMatch ? codeMatch[1] : (extractAttr(attrs, 'defaultCode') || '')

    return `<div data-widget="pyrunner" data-cellid="${esc(cellId)}" data-defaultcode="${esc(defaultCode)}" data-timeout="${esc(timeout)}" data-title="${esc(title)}"></div>`
  })

  // Multi-line PyRunner (with children-like props spanning lines)
  md = md.replace(/<PyRunner\s+([\s\S]*?)\/>/gs, (_, attrs) => {
    const cellId = extractAttr(attrs, 'cellId') || ''
    const timeout = extractAttr(attrs, 'timeout') || '5'
    const title = extractAttr(attrs, 'title') || ''
    const codeMatch = attrs.match(/defaultCode=\{`([\s\S]*?)`\}/)
    const defaultCode = codeMatch ? codeMatch[1] : (extractAttr(attrs, 'defaultCode') || '')

    return `<div data-widget="pyrunner" data-cellid="${esc(cellId)}" data-defaultcode="${esc(defaultCode)}" data-timeout="${esc(timeout)}" data-title="${esc(title)}"></div>`
  })

  // <Quiz chapterSlug="..." questions={[...]} />
  md = md.replace(/<Quiz\s+([\s\S]*?)\/>/gs, (_, attrs) => {
    const chapterSlug = extractAttr(attrs, 'chapterSlug') || ''
    // Extract questions array — it's a JS object literal, we'll wrap it as JSON-compatible
    const qMatch = attrs.match(/questions=\{(\[[\s\S]*?\])\}/)
    let questions = '[]'
    if (qMatch) {
      try {
        // Use Function constructor to safely evaluate the JS array literal
        const parsed = new Function('return ' + qMatch[1])()
        questions = JSON.stringify(parsed)
      } catch (e) {
        console.warn(`Failed to parse quiz questions for ${chapterSlug}:`, e.message)
        questions = '[]'
      }
    }

    const qId = `quiz-${chapterSlug || Math.random().toString(36).slice(2)}`
    return `<div data-widget="quiz" data-chapterslug="${esc(chapterSlug)}" data-quizid="${qId}"></div>\n<script type="application/json" id="${qId}">${questions}</script>`
  })

  // <Visualizer mode="..." shapeA={[3,1]} shapeB={[1,4]} title="..." />
  md = md.replace(/<Visualizer\s+([^>]*?)\/>/gs, (_, attrs) => {
    const mode = extractAttr(attrs, 'mode') || 'broadcasting'
    const title = extractAttr(attrs, 'title') || ''
    const shapeAMatch = attrs.match(/shapeA=\{\[([\d,\s]+)\]\}/)
    const shapeBMatch = attrs.match(/shapeB=\{\[([\d,\s]+)\]\}/)
    const shapeA = shapeAMatch ? shapeAMatch[1].trim() : '3,1'
    const shapeB = shapeBMatch ? shapeBMatch[1].trim() : '1,4'

    return `<div data-widget="visualizer" data-mode="${esc(mode)}" data-shapea="${esc(shapeA)}" data-shapeb="${esc(shapeB)}" data-title="${esc(title)}"></div>`
  })

  // <InlineMath latex="..." />
  md = md.replace(/<InlineMath\s+latex="([^"]*)"\s*\/>/g, (_, latex) => {
    return `<span data-widget="katex" data-latex="${esc(latex)}" data-display="false"></span>`
  })

  // <BlockMath latex="..." />
  md = md.replace(/<BlockMath\s+latex="([^"]*)"\s*\/>/g, (_, latex) => {
    return `<div data-widget="katex" data-latex="${esc(latex)}" data-display="true"></div>`
  })

  // <Callout type="..." title="...">content</Callout>
  md = md.replace(/<Callout\s+type="(\w+)"(?:\s+title="([^"]*)")?>\s*([\s\S]*?)<\/Callout>/g, (_, type, title, content) => {
    // Content might have markdown — we'll keep it as-is for now (marked will process it)
    const cleanContent = content.trim().replace(/\n/g, ' ')
    return `<div data-widget="callout" data-type="${esc(type)}" data-callouttitle="${esc(title || '')}" data-content="${esc(cleanContent)}"></div>`
  })

  return md
}

function extractAttr(attrs, name) {
  // Match name="value" or name={'value'} or name={number}
  const strMatch = attrs.match(new RegExp(`${name}="([^"]*)"`))
  if (strMatch) return strMatch[1]
  const braceMatch = attrs.match(new RegExp(`${name}=\{['"]?([^}'"]*)['"]?\}`))
  if (braceMatch) return braceMatch[1]
  return null
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/\n/g, '&#10;')
}

// Simple markdown → HTML converter (no external deps needed for basic rendering)
function markdownToHtml(md) {
  let html = md

  // Code blocks (fenced)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code class="language-${lang || 'text'}">${esc(code.trim())}</code></pre>`
  })

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')

  // Headings
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>')
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>')
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>')
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>')

  // Bold & italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')

  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')

  // Horizontal rules
  html = html.replace(/^---$/gm, '<hr>')

  // Unordered lists
  html = html.replace(/^(\s*)[-*] (.+)$/gm, '$1<li>$2</li>')
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>')

  // Tables (basic)
  html = html.replace(/^\|(.+)\|$/gm, (match, content) => {
    const cells = content.split('|').map(c => c.trim())
    if (cells.every(c => /^[-:]+$/.test(c))) return '' // separator row
    const tag = 'td'
    return '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>'
  })
  html = html.replace(/(<tr>.*<\/tr>\n?)+/g, '<table>$&</table>')

  // Paragraphs (lines not starting with HTML tags or empty)
  html = html.replace(/^(?!<[a-zA-Z/]|$)(.+)$/gm, '<p>$1</p>')

  // Clean up empty paragraphs and double newlines
  html = html.replace(/<p>\s*<\/p>/g, '')
  html = html.replace(/\n{3,}/g, '\n\n')

  return html
}

// ─── Build All Chapters ──────────────────────────────────────────────────────

let built = 0
let errors = 0

for (const slug of allSlugs) {
  const filePath = path.join(CONTENT_DIR, `${slug}.md`)
  if (!fs.existsSync(filePath)) {
    console.error(`  ✗ Missing: ${slug}.md`)
    errors++
    continue
  }

  try {
    const raw = fs.readFileSync(filePath, 'utf-8')
    const { meta, content } = parseFrontmatter(raw)

    // Convert MDX widgets → HTML data attributes
    const converted = convertMdxToWidgets(content)

    // Protect widget divs from markdown parser by replacing with placeholders
    const widgetStore = []
    const protected_ = converted.replace(/<div data-widget=[\s\S]*?<\/div>/g, (match) => {
      const idx = widgetStore.length
      widgetStore.push(match)
      return `\n%%WIDGET_${idx}%%\n`
    })
    // Also protect standalone self-closing widget spans (katex)
    const protected2 = protected_.replace(/<span data-widget=[^>]*><\/span>/g, (match) => {
      const idx = widgetStore.length
      widgetStore.push(match)
      return `%%WIDGET_${idx}%%`
    })

    // Convert markdown → HTML
    let htmlContent = markdownToHtml(protected2)

    // Restore widget divs (handle both bare placeholders and <p>-wrapped ones)
    for (let i = 0; i < widgetStore.length; i++) {
      const tag = `%%WIDGET_${i}%%`
      // Replace <p>%%WIDGET_N%%</p> first, then bare placeholder
      htmlContent = htmlContent.replaceAll(`<p>${tag}</p>`, widgetStore[i])
      htmlContent = htmlContent.replaceAll(tag, widgetStore[i])
    }

    // Find prev/next
    const idx = allSlugs.indexOf(slug)
    const prevSlug = idx > 0 ? allSlugs[idx - 1] : null
    const nextSlug = idx < allSlugs.length - 1 ? allSlugs[idx + 1] : null

    const prevLink = prevSlug
      ? `<a href="/chapters/${prevSlug}.html" class="btn-ghost" style="font-size: 0.8125rem; padding: 0.5rem 1rem;">← Previous</a>`
      : '<div></div>'
    const nextLink = nextSlug
      ? `<a href="/chapters/${nextSlug}.html" class="btn-primary" style="font-size: 0.8125rem; padding: 0.5rem 1rem;">Next →</a>`
      : '<div></div>'

    // Fill template
    let page = template
      .replaceAll('{{TITLE}}', esc(meta.title || slug))
      .replaceAll('{{DESCRIPTION}}', esc(meta.description || ''))
      .replaceAll('{{READ_TIME}}', meta.read_time || 10)
      .replaceAll('{{CODE_TIME}}', meta.code_time || 5)
      .replace('{{CONTENT}}', htmlContent)
      .replace('{{PREV_LINK}}', prevLink)
      .replace('{{NEXT_LINK}}', nextLink)

    // Add slug to body for chapter.js
    page = page.replace('<body ', `<body data-slug="${esc(slug)}" `)

    fs.writeFileSync(path.join(OUT_DIR, `${slug}.html`), page, 'utf-8')
    built++
  } catch (err) {
    console.error(`  ✗ Error building ${slug}: ${err.message}`)
    errors++
  }
}

console.log(`\n✓ Built ${built}/${allSlugs.length} chapters (${errors} errors)`)
console.log(`  Output: ${OUT_DIR}`)
