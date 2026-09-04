// Pyodide Worker Manager
// All Pyodide execution happens in a WebWorker. Never on main thread.

export interface ExecRequest {
  type: 'execute'
  id: string
  code: string
  timeout: number
}

export interface ExecResponse {
  type: 'result' | 'error' | 'stdout' | 'stderr' | 'ready' | 'timeout'
  id?: string
  data?: string
  error?: string
}

type ExecCallback = {
  resolve: (output: string) => void
  reject: (error: string) => void
  stdout: string[]
  stderr: string[]
}

class PyodideManager {
  private worker: Worker | null = null
  private ready = false
  private pending: Map<string, ExecCallback> = new Map()
  private watchdogTimer: ReturnType<typeof setTimeout> | null = null
  private executionCount = 0
  private readonly MAX_EXECUTIONS_BEFORE_RECYCLE = 50

  async init(): Promise<void> {
    if (this.worker && this.ready) return

    return new Promise((resolve, reject) => {
      try {
        this.worker = new Worker(
          new URL('../workers/pyodide.worker.ts', import.meta.url),
          { type: 'module' }
        )

        this.worker.onmessage = (e: MessageEvent<ExecResponse>) => {
          this.handleMessage(e.data)
          if (e.data.type === 'ready') {
            this.ready = true
            resolve()
          }
        }

        this.worker.onerror = (err) => {
          console.error('Pyodide worker error:', err)
          this.rejectAll('Worker crashed unexpectedly')
          this.respawn()
          reject(new Error('Worker initialization failed'))
        }
      } catch (err) {
        reject(err)
      }
    })
  }

  private handleMessage(msg: ExecResponse) {
    switch (msg.type) {
      case 'stdout': {
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.data) cb.stdout.push(msg.data)
        break
      }
      case 'stderr': {
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.data) cb.stderr.push(msg.data)
        break
      }
      case 'result': {
        this.clearWatchdog()
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.id) {
          const output = cb.stdout.join('')
          const result = msg.data ? `${output}${output ? '\n' : ''}${msg.data}` : output
          cb.resolve(result || '(no output)')
          this.pending.delete(msg.id)
        }
        this.checkRecycle()
        break
      }
      case 'error': {
        this.clearWatchdog()
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.id) {
          const stderr = cb.stderr.join('')
          cb.reject(msg.error || stderr || 'Unknown error')
          this.pending.delete(msg.id)
        }
        this.checkRecycle()
        break
      }
      case 'timeout': {
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.id) {
          cb.reject(`Execution timed out after ${msg.data}s`)
          this.pending.delete(msg.id)
        }
        break
      }
    }
  }

  async execute(code: string, timeout: number = 5): Promise<string> {
    await this.init()

    const id = `exec_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, stdout: [], stderr: [] })

      // Watchdog: terminate worker on timeout, NOT AbortController
      this.watchdogTimer = setTimeout(() => {
        this.terminateWorker()
        const cb = this.pending.get(id)
        if (cb) {
          cb.reject(`Execution timed out after ${timeout}s. Worker terminated.`)
          this.pending.delete(id)
        }
        // Log to localStorage for debugging
        const errors = JSON.parse(localStorage.getItem('nna_errors') || '[]')
        errors.push({ type: 'timeout', code: code.slice(0, 200), at: new Date().toISOString() })
        localStorage.setItem('nna_errors', JSON.stringify(errors.slice(-20)))
        // Respawn for next execution
        this.ready = false
        this.worker = null
      }, timeout * 1000)

      this.worker!.postMessage({
        type: 'execute',
        id,
        code,
        timeout,
      } satisfies ExecRequest)
    })
  }

  private clearWatchdog() {
    if (this.watchdogTimer) {
      clearTimeout(this.watchdogTimer)
      this.watchdogTimer = null
    }
  }

  private checkRecycle() {
    this.executionCount++
    if (this.executionCount >= this.MAX_EXECUTIONS_BEFORE_RECYCLE) {
      this.terminateWorker()
      this.ready = false
      this.worker = null
      this.executionCount = 0
    }
  }

  private terminateWorker() {
    this.clearWatchdog()
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
    this.ready = false
  }

  private respawn() {
    this.terminateWorker()
    // Will re-init on next execute() call
  }

  private rejectAll(reason: string) {
    for (const [id, cb] of this.pending) {
      cb.reject(reason)
    }
    this.pending.clear()
  }

  destroy() {
    this.terminateWorker()
    this.pending.clear()
  }
}

// Singleton
export const pyodide = new PyodideManager()
