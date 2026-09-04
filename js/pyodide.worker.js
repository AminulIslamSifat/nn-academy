// Pyodide WebWorker — all Python execution happens here, NEVER on main thread

let pyodide = null
let isLoading = false
let currentExecId = null

const PYODIDE_VERSION = '0.26.4'
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

async function loadPyodideRuntime() {
  if (pyodide || isLoading) return
  isLoading = true

  try {
    importScripts(`${PYODIDE_URL}pyodide.js`)
    pyodide = await self.loadPyodide({
      indexURL: PYODIDE_URL,
      stdout: (text) => {
        postMessage({ type: 'stdout', data: text, id: currentExecId })
      },
      stderr: (text) => {
        postMessage({ type: 'stderr', data: text, id: currentExecId })
      },
    })

    await pyodide.loadPackage('numpy')
    postMessage({ type: 'ready' })
  } catch (err) {
    postMessage({ type: 'error', error: `Failed to initialize Pyodide: ${err.message}` })
  } finally {
    isLoading = false
  }
}

self.onmessage = async (e) => {
  const { type, id, code, timeout } = e.data

  if (type === 'execute') {
    currentExecId = id

    try {
      if (!pyodide) await loadPyodideRuntime()

      pyodide.setStdout({ batched: (text) => postMessage({ type: 'stdout', data: text, id }) })
      pyodide.setStderr({ batched: (text) => postMessage({ type: 'stderr', data: text, id }) })

      const result = await pyodide.runPythonAsync(code)
      let output = ''
      if (result !== undefined && result !== null) {
        try { output = String(result) } catch { output = '[object]' }
      }

      postMessage({ type: 'result', id, data: output })
    } catch (err) {
      postMessage({ type: 'error', id, error: err.message || String(err) })
    } finally {
      currentExecId = null
    }
  }
}

loadPyodideRuntime()
