/// <reference lib="webworker" />

// Pyodide WebWorker — all Python execution happens here, NEVER on main thread

let pyodide: any = null
let isLoading = false

const PYODIDE_VERSION = '0.26.4'
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

async function loadPyodide() {
  if (pyodide || isLoading) return
  isLoading = true

  try {
    importScripts(`${PYODIDE_URL}pyodide.js`)
    pyodide = await (self as any).loadPyodide({
      indexURL: PYODIDE_URL,
      stdout: (text: string) => {
        postMessage({ type: 'stdout', data: text, id: currentExecId })
      },
      stderr: (text: string) => {
        postMessage({ type: 'stderr', data: text, id: currentExecId })
      },
    })

    // Load NumPy
    await pyodide.loadPackage('numpy')

    postMessage({ type: 'ready' })
  } catch (err: any) {
    postMessage({ type: 'error', error: `Failed to initialize Pyodide: ${err.message}` })
  } finally {
    isLoading = false
  }
}

let currentExecId: string | null = null

self.onmessage = async (e: MessageEvent) => {
  const { type, id, code, timeout } = e.data

  if (type === 'execute') {
    currentExecId = id

    try {
      if (!pyodide) {
        await loadPyodide()
      }

      // Redirect stdout/stderr for this execution
      pyodide.setStdout({
        batched: (text: string) => {
          postMessage({ type: 'stdout', data: text, id })
        },
      })
      pyodide.setStderr({
        batched: (text: string) => {
          postMessage({ type: 'stderr', data: text, id })
        },
      })

      // Execute the code
      const result = await pyodide.runPythonAsync(code)

      // Format result
      let output = ''
      if (result !== undefined && result !== null) {
        try {
          // Try to convert numpy arrays to string representation
          output = String(result)
        } catch {
          output = repr(result)
        }
      }

      postMessage({ type: 'result', id, data: output })
    } catch (err: any) {
      postMessage({ type: 'error', id, error: err.message || String(err) })
    } finally {
      currentExecId = null
    }
  }
}

function repr(obj: any): string {
  try {
    if (obj && typeof obj.toString === 'function') {
      return obj.toString()
    }
    return String(obj)
  } catch {
    return '[object]'
  }
}

// Initialize on load
loadPyodide()
