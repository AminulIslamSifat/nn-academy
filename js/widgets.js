// Vanilla JS widgets — ported directly from old React components.
// Uses same Tailwind classes + CSS classes from styles.css.

import { pyodide } from './pyodide.js'
import { markCompleted, saveQuizScore } from './store.js'

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

// ─── PyRunner (from PyRunner.tsx) ────────────────────────────────────────────

export function mountPyRunner(el) {
  const cellId = el.dataset.cellid || ''
  const defaultCode = el.dataset.defaultcode || ''
  const timeout = parseInt(el.dataset.timeout || '5', 10)
  const title = el.dataset.title || cellId

  let code = defaultCode
  let state = 'idle' // idle | loading | running | success | error

  el.className = 'pyrunner-widget'
  el.innerHTML = `
    <div class="pyrunner-header">
      <div style="display:flex;align-items:center;gap:8px">
        <div class="pyrunner-dots">
          <span class="pyrunner-dot" style="background:rgba(239,68,68,0.6)"></span>
          <span class="pyrunner-dot" style="background:rgba(234,179,8,0.6)"></span>
          <span class="pyrunner-dot" style="background:rgba(34,197,94,0.6)"></span>
        </div>
        <span style="margin-left:8px;font-size:0.875rem;font-family:ui-monospace,'JetBrains Mono',monospace;color:#71717a">${escapeHtml(title)}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <button class="reset-btn rounded p-1.5 text-zinc-500 hover:text-zinc-300 transition" title="Reset code" style="background:none;border:none;cursor:pointer">↺</button>
        <button class="run-btn flex items-center gap-1.5 rounded-lg bg-accent-500/10 px-4 py-2 text-sm font-medium text-accent-400 hover:bg-accent-500/20 transition disabled:opacity-50" style="border:none;cursor:pointer">▶ Run</button>
      </div>
    </div>
    <textarea class="pyrunner-textarea" spellcheck="false" aria-label="Python code editor"></textarea>
    <div class="output-area"></div>
  `

  const textarea = el.querySelector('.pyrunner-textarea')
  const outputArea = el.querySelector('.output-area')
  const runBtn = el.querySelector('.run-btn')
  const resetBtn = el.querySelector('.reset-btn')

  textarea.value = code
  textarea.addEventListener('input', () => { code = textarea.value })

  resetBtn.addEventListener('click', () => {
    code = defaultCode
    textarea.value = code
    outputArea.innerHTML = ''
    outputArea.className = 'output-area'
    state = 'idle'
    updateRunBtn()
  })

  runBtn.addEventListener('click', async () => {
    if (state === 'loading' || state === 'running') return
    state = 'loading'
    updateRunBtn()
    outputArea.innerHTML = '<div class="pyrunner-output" style="color:#71717a;display:flex;align-items:center;gap:8px"><span>⏳</span> Initializing Python runtime...</div>'
    outputArea.className = 'output-area'

    try {
      state = 'running'
      updateRunBtn()
      const result = await pyodide.execute(code, timeout)
      state = 'success'
      outputArea.className = 'pyrunner-output success'
      outputArea.innerHTML = `<div style="display:flex;align-items:flex-start;gap:8px"><span style="color:#10b981;margin-top:2px">✓</span><pre style="white-space:pre-wrap;font-family:ui-monospace,'JetBrains Mono',monospace;margin:0;color:#d4d4d8">${escapeHtml(result)}</pre></div>`
    } catch (err) {
      state = 'error'
      outputArea.className = 'pyrunner-output error'
      outputArea.innerHTML = `<div style="display:flex;align-items:flex-start;gap:8px"><span style="color:#f87171;margin-top:2px">⚠</span><pre style="white-space:pre-wrap;font-family:ui-monospace,'JetBrains Mono',monospace;margin:0;color:#fca5a5">${escapeHtml(err.message || String(err))}</pre></div>`
    }
    updateRunBtn()
  })

  function updateRunBtn() {
    if (state === 'loading' || state === 'running') {
      runBtn.disabled = true
      runBtn.style.opacity = '0.5'
      runBtn.innerHTML = '<span>⏳</span> Running...'
    } else {
      runBtn.disabled = false
      runBtn.style.opacity = '1'
      runBtn.innerHTML = '▶ Run'
    }
  }
}

// ─── Quiz (from Quiz.tsx) ────────────────────────────────────────────────────

export function mountQuiz(el) {
  const chapterSlug = el.dataset.chapterslug || ''
  const quizId = el.dataset.quizid || ''
  let questions
  try {
    const scriptEl = document.getElementById(quizId)
    questions = scriptEl ? JSON.parse(scriptEl.textContent) : []
  } catch {
    el.innerHTML = '<p style="color:#f87171">Failed to parse quiz data</p>'
    return
  }

  // Shuffle options for randomize questions
  const shuffled = questions.map(q => {
    if (!q.randomize) return q
    const indices = q.options.map((_, i) => i)
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]]
    }
    return { ...q, options: indices.map(i => q.options[i]), correctIndex: indices.indexOf(q.correctIndex) }
  })

  const answers = {}
  let submitted = false
  const showExplanations = new Set()

  el.className = 'quiz-widget'
  render()

  function render() {
    const score = shuffled.filter(q => answers[q.id] === q.correctIndex).length
    const allAnswered = shuffled.every(q => answers[q.id] !== undefined)

    let html = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem">
        <h3 style="font-size:1.125rem;font-weight:600;color:#e4e4e7">Knowledge Check</h3>
        ${submitted ? `<div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:0.875rem;font-weight:500;color:${score === shuffled.length ? '#34d399' : '#a1a1aa'}">${score}/${shuffled.length} correct</span>
          <button class="retry-btn" style="display:flex;align-items:center;gap:4px;font-size:0.75rem;color:#71717a;background:none;border:none;cursor:pointer;transition:color 150ms">↺ Retry</button>
        </div>` : ''}
      </div>
      <div class="quiz-questions" style="display:flex;flex-direction:column;gap:1.5rem"></div>
    `

    if (!submitted) {
      html += `<button class="submit-btn" ${!allAnswered ? 'disabled' : ''} style="margin-top:1.5rem;width:100%;border-radius:0.5rem;background:#10b981;border:none;padding:0.75rem;font-weight:500;color:#09090b;cursor:pointer;font-size:0.9rem;transition:background 150ms;${!allAnswered ? 'opacity:0.4;cursor:not-allowed' : ''}">Submit Answers</button>`
    }

    el.innerHTML = html

    const container = el.querySelector('.quiz-questions')
    shuffled.forEach((q, qi) => {
      const selected = answers[q.id]
      const isCorrect = selected === q.correctIndex

      const qDiv = document.createElement('div')
      qDiv.className = 'quiz-question'
      qDiv.innerHTML = `
        <p style="font-size:0.875rem;font-weight:500;color:#d4d4d8;margin-bottom:0.75rem">
          <span style="color:#34d399;margin-right:8px">Q${qi + 1}.</span>${escapeHtml(q.prompt)}
        </p>
        ${q.code ? `<pre style="margin-bottom:0.75rem;border-radius:0.5rem;background:#09090b;padding:0.75rem;font-size:0.75rem;font-family:ui-monospace,'JetBrains Mono',monospace;color:#a1a1aa;overflow-x:auto">${escapeHtml(q.code)}</pre>` : ''}
        <div class="options-container" style="display:flex;flex-direction:column;gap:0.5rem"></div>
        ${submitted ? `
          <div style="margin-top:0.75rem">
            <div style="display:flex;align-items:center;gap:8px;font-size:0.875rem">
              <span style="color:${isCorrect ? '#10b981' : '#f87171'}">${isCorrect ? '✓' : '✗'}</span>
              <span style="color:${isCorrect ? '#34d399' : '#fca5a5'}">${isCorrect ? 'Correct!' : 'Incorrect'}</span>
            </div>
            <button class="expl-toggle" style="margin-top:0.5rem;display:flex;align-items:center;gap:4px;font-size:0.75rem;color:#71717a;background:none;border:none;cursor:pointer;transition:color 150ms">
              <span style="transition:transform 150ms;display:inline-block;${showExplanations.has(q.id) ? 'transform:rotate(180deg)' : ''}">▾</span>
              ${showExplanations.has(q.id) ? 'Hide' : 'Show'} explanation
            </button>
            <div class="expl-area"></div>
          </div>
        ` : ''}
      `

      const optsContainer = qDiv.querySelector('.options-container')
      q.options.forEach((opt, oi) => {
        const btn = document.createElement('button')
        btn.className = 'quiz-option'
        btn.textContent = opt
        if (submitted) {
          btn.disabled = true
          if (oi === q.correctIndex) btn.classList.add('correct')
          else if (oi === selected) btn.classList.add('wrong')
          else btn.classList.add('dimmed')
        } else if (oi === selected) {
          btn.classList.add('selected')
        }
        btn.addEventListener('click', () => {
          if (submitted) return
          answers[q.id] = oi
          render()
        })
        optsContainer.appendChild(btn)
      })

      if (submitted) {
        const explArea = qDiv.querySelector('.expl-area')
        const toggleBtn = qDiv.querySelector('.expl-toggle')
        if (showExplanations.has(q.id)) {
          explArea.innerHTML = `<p style="margin-top:0.5rem;border-radius:0.5rem;background:#09090b;padding:0.75rem;font-size:0.875rem;color:#a1a1aa">${escapeHtml(q.explanation)}</p>`
        }
        toggleBtn.addEventListener('click', () => {
          if (showExplanations.has(q.id)) showExplanations.delete(q.id)
          else showExplanations.add(q.id)
          render()
        })
      }

      container.appendChild(qDiv)
    })

    // Bind submit/retry
    const submitBtn = el.querySelector('.submit-btn')
    if (submitBtn) {
      submitBtn.addEventListener('click', async () => {
        submitted = true
        for (const q of shuffled) {
          const correct = answers[q.id] === q.correctIndex ? 1 : 0
          saveQuizScore(chapterSlug, q.id, correct)
        }
        const allCorrect = shuffled.every(q => answers[q.id] === q.correctIndex)
        if (allCorrect && shuffled.length > 0) markCompleted(chapterSlug)
        render()
      })
    }

    const retryBtn = el.querySelector('.retry-btn')
    if (retryBtn) {
      retryBtn.addEventListener('click', () => {
        Object.keys(answers).forEach(k => delete answers[k])
        submitted = false
        showExplanations.clear()
        render()
      })
    }
  }
}

// ─── Visualizer (from Visualizer.tsx) ────────────────────────────────────────

export function mountVisualizer(el) {
  const mode = el.dataset.mode || 'broadcasting'
  const shapeA = (el.dataset.shapea || '3,1').split(',').map(Number)
  const shapeB = (el.dataset.shapeb || '1,4').split(',').map(Number)
  const title = el.dataset.title || `${mode === 'broadcasting' ? 'Broadcasting' : mode === 'matmul' ? 'Matrix Multiply' : 'Reshape'} Visualizer`

  let inputA = shapeA.join(', ')
  let inputB = shapeB.join(', ')
  let steps = []
  let result = null
  let currentStep = -1
  let animating = false

  el.className = 'viz-widget'
  render()

  function computeBroadcast(a, b) {
    const maxDims = Math.max(a.length, b.length)
    const padA = Array(maxDims - a.length).fill(1).concat(a)
    const padB = Array(maxDims - b.length).fill(1).concat(b)
    const res = [], stps = []
    for (let i = 0; i < maxDims; i++) {
      if (padA[i] === padB[i]) { res.push(padA[i]); stps.push(`dim ${i}: ${padA[i]} == ${padB[i]} → ${padA[i]}`) }
      else if (padA[i] === 1) { res.push(padB[i]); stps.push(`dim ${i}: ${padA[i]} broadcasts → ${padB[i]}`) }
      else if (padB[i] === 1) { res.push(padA[i]); stps.push(`dim ${i}: ${padB[i]} broadcasts → ${padA[i]}`) }
      else { stps.push(`dim ${i}: INCOMPATIBLE (${padA[i]} vs ${padB[i]})`); return { result: [], steps: stps } }
    }
    return { result: res, steps: stps }
  }

  function computeMatmul(a, b) {
    const stps = []
    if (a.length < 2 || b.length < 2) { stps.push('Both arrays must be at least 2D'); return { result: [], steps: stps } }
    const [m, k1] = [a[a.length - 2], a[a.length - 1]]
    const [k2, n] = [b[b.length - 2], b[b.length - 1]]
    if (k1 !== k2) { stps.push(`Inner dimensions don't match: ${k1} ≠ ${k2}`); return { result: [], steps: stps } }
    stps.push(`A: (..., ${m}, ${k1}) × B: (..., ${k2}, ${n})`)
    stps.push(`Inner dim ${k1} == ${k2} ✓`)
    stps.push(`Result: (..., ${m}, ${n})`)
    return { result: [m, n], steps: stps }
  }

  function runViz() {
    const a = inputA.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x) && x > 0)
    const b = inputB.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x) && x > 0)
    if (!a.length || !b.length) return

    let res
    if (mode === 'broadcasting') res = computeBroadcast(a, b)
    else if (mode === 'matmul') res = computeMatmul(a, b)
    else res = { result: a, steps: ['Reshape preserves total elements'] }

    steps = res.steps
    result = res.result
    currentStep = -1
    animating = true
    render()

    res.steps.forEach((_, i) => {
      setTimeout(() => { currentStep = i; renderSteps() }, (i + 1) * 600)
    })
    setTimeout(() => { animating = false; render() }, (res.steps.length + 1) * 600)
  }

  function renderSteps() {
    const canvas = el.querySelector('.viz-canvas')
    if (!canvas) return
    const existing = canvas.querySelectorAll('.viz-step')
    existing.forEach((s, i) => {
      if (i <= currentStep) s.classList.add('visible')
    })
    // Add new steps
    for (let i = existing.length; i <= currentStep && i < steps.length; i++) {
      const div = document.createElement('div')
      div.className = 'viz-step'
      div.innerHTML = `<span style="color:#34d399;flex-shrink:0">›</span><span style="font-family:ui-monospace,'JetBrains Mono',monospace;color:#d4d4d8">${escapeHtml(steps[i])}</span>`
      canvas.insertBefore(div, canvas.querySelector('.viz-result-area'))
      requestAnimationFrame(() => div.classList.add('visible'))
    }
    // Show result when done
    if (currentStep >= steps.length - 1) {
      const resultArea = canvas.querySelector('.viz-result-area')
      if (result && result.length > 0) {
        resultArea.innerHTML = `<div class="viz-result-box" style="animation:fadeIn 300ms ease"><span style="font-size:0.75rem;color:#71717a;display:block;margin-bottom:4px">Result Shape</span><span style="font-family:ui-monospace,'JetBrains Mono',monospace;font-size:1.125rem;color:#6ee7b7">(${result.join(', ')})</span></div>`
      } else if (result && result.length === 0) {
        resultArea.innerHTML = '<div style="text-align:center;font-size:0.875rem;color:#f87171;animation:fadeIn 300ms ease">❌ Incompatible shapes</div>'
      }
    }
  }

  function render() {
    el.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem">
        <h3 style="font-size:0.875rem;font-weight:600;color:#d4d4d8">${escapeHtml(title)}</h3>
        <div style="display:flex;gap:8px">
          <button class="viz-reset rounded p-1.5 text-zinc-500 hover:text-zinc-300 transition" style="background:none;border:none;cursor:pointer" aria-label="Reset">↺</button>
          <button class="viz-run flex items-center gap-1.5 rounded-lg bg-accent-500/10 px-3 py-1.5 text-xs font-medium text-accent-400 hover:bg-accent-500/20 transition disabled:opacity-50" style="border:none;cursor:pointer" ${animating ? 'disabled' : ''} aria-label="Run visualization">▶ Animate</button>
        </div>
      </div>
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:1rem;margin-bottom:1.5rem">
        <div style="display:flex;align-items:center;gap:8px">
          <label style="font-size:0.75rem;color:#71717a;font-family:ui-monospace,monospace">A =</label>
          <input type="text" class="input-a" value="${escapeHtml(inputA)}" style="width:6rem;border-radius:0.5rem;border:1px solid #3f3f46;background:#09090b;padding:0.375rem 0.75rem;font-family:ui-monospace,monospace;font-size:0.875rem;color:#e4e4e7;outline:none" aria-label="Shape A">
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <label style="font-size:0.75rem;color:#71717a;font-family:ui-monospace,monospace">B =</label>
          <input type="text" class="input-b" value="${escapeHtml(inputB)}" style="width:6rem;border-radius:0.5rem;border:1px solid #3f3f46;background:#09090b;padding:0.375rem 0.75rem;font-family:ui-monospace,monospace;font-size:0.875rem;color:#e4e4e7;outline:none" aria-label="Shape B">
        </div>
      </div>
      <div class="viz-canvas">
        ${steps.length === 0 ? '<p style="text-align:center;font-size:0.875rem;color:#52525b;padding:1rem 0">Enter shapes and click Animate to see the operation step-by-step.</p>' : ''}
        <div class="viz-result-area" style="margin-top:1rem"></div>
      </div>
    `

    if (steps.length > 0) renderSteps()

    el.querySelector('.input-a').addEventListener('input', e => { inputA = e.target.value })
    el.querySelector('.input-b').addEventListener('input', e => { inputB = e.target.value })
    el.querySelector('.viz-run').addEventListener('click', runViz)
    el.querySelector('.viz-reset').addEventListener('click', () => {
      steps = []; result = null; currentStep = -1; animating = false
      inputA = shapeA.join(', '); inputB = shapeB.join(', ')
      render()
    })
  }
}

// ─── KaTeX ───────────────────────────────────────────────────────────────────

export function mountKaTeX(el) {
  const latex = el.dataset.latex || ''
  const display = el.dataset.display === 'true'

  if (!window.katex) {
    const script = document.createElement('script')
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'
    script.onload = () => renderMath(el, latex, display)
    document.head.appendChild(script)
  } else {
    renderMath(el, latex, display)
  }
}

function renderMath(el, latex, display) {
  try {
    window.katex.render(latex, el, { displayMode: display, throwOnError: false })
  } catch {
    el.textContent = latex
    el.style.color = '#f87171'
  }
}

// ─── Callout ─────────────────────────────────────────────────────────────────

export function mountCallout(el) {
  const type = el.dataset.type || 'info'
  const title = el.dataset.callouttitle || ''
  const content = el.dataset.content || ''
  const icons = { info: 'ℹ️', warning: '⚠️', tip: '💡', danger: '🚨' }

  el.className = `callout-widget callout-${type}`
  el.innerHTML = `
    ${title ? `<div style="margin-bottom:0.5rem;display:flex;align-items:center;gap:8px;font-weight:500;color:#e4e4e7"><span>${icons[type] || ''}</span> ${escapeHtml(title)}</div>` : ''}
    <div style="font-size:0.875rem;color:#d4d4d8;line-height:1.6">${content}</div>
  `
}
