// Pyodide Worker Manager — vanilla JS port of the original TypeScript version.
// All Python execution happens in a WebWorker. Never on main thread.

class PyodideManager {
  constructor() {
    this.worker = null
    this.ready = false
    this.pending = new Map()
    this.watchdogTimer = null
    this.executionCount = 0
    this.MAX_EXECUTIONS_BEFORE_RECYCLE = 50
  }

  async init() {
    if (this.worker && this.ready) return

    return new Promise((resolve, reject) => {
      try {
        this.worker = new Worker('/js/pyodide.worker.js')

        this.worker.onmessage = (e) => {
          this._handleMessage(e.data)
          if (e.data.type === 'ready') {
            this.ready = true
            resolve()
          }
        }

        this.worker.onerror = (err) => {
          console.error('Pyodide worker error:', err)
          this._rejectAll('Worker crashed unexpectedly')
          this._respawn()
          reject(new Error('Worker initialization failed'))
        }
      } catch (err) {
        reject(err)
      }
    })
  }

  _handleMessage(msg) {
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
        this._clearWatchdog()
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.id) {
          const output = cb.stdout.join('')
          const result = msg.data ? `${output}${output ? '\n' : ''}${msg.data}` : output
          cb.resolve(result || '(no output)')
          this.pending.delete(msg.id)
        }
        this._checkRecycle()
        break
      }
      case 'error': {
        this._clearWatchdog()
        const cb = msg.id ? this.pending.get(msg.id) : null
        if (cb && msg.id) {
          const stderr = cb.stderr.join('')
          cb.reject(msg.error || stderr || 'Unknown error')
          this.pending.delete(msg.id)
        }
        this._checkRecycle()
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

  async execute(code, timeout = 5) {
    await this.init()
    const id = `exec_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, stdout: [], stderr: [] })

      this.watchdogTimer = setTimeout(() => {
        this._terminateWorker()
        const cb = this.pending.get(id)
        if (cb) {
          cb.reject(`Execution timed out after ${timeout}s. Worker terminated.`)
          this.pending.delete(id)
        }
        this.ready = false
        this.worker = null
      }, timeout * 1000)

      this.worker.postMessage({ type: 'execute', id, code, timeout })
    })
  }

  _clearWatchdog() {
    if (this.watchdogTimer) { clearTimeout(this.watchdogTimer); this.watchdogTimer = null }
  }

  _checkRecycle() {
    this.executionCount++
    if (this.executionCount >= this.MAX_EXECUTIONS_BEFORE_RECYCLE) {
      this._terminateWorker()
      this.ready = false
      this.worker = null
      this.executionCount = 0
    }
  }

  _terminateWorker() {
    this._clearWatchdog()
    if (this.worker) { this.worker.terminate(); this.worker = null }
    this.ready = false
  }

  _respawn() { this._terminateWorker() }

  _rejectAll(reason) {
    for (const [, cb] of this.pending) cb.reject(reason)
    this.pending.clear()
  }

  destroy() { this._terminateWorker(); this.pending.clear() }
}

export const pyodide = new PyodideManager()
