'use client'

import { useEffect, useRef, useState } from 'react'
import { Settings2 } from 'lucide-react'

interface EditorProps {
  value: string
  onChange?: (value: string) => void
  language?: string
  readOnly?: boolean
  height?: string
}

export default function Editor({
  value,
  onChange,
  language = 'python',
  readOnly = false,
  height = '200px',
}: EditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const editorRef = useRef<any>(null)
  const [useMonaco, setUseMonaco] = useState(false)

  // ─── CodeMirror 6 (default) ───────────────────────────────────────────────
  useEffect(() => {
    if (useMonaco) return

    let mounted = true

    async function initCodeMirror() {
      if (!containerRef.current || !mounted) return

      const { EditorView, basicSetup } = await import('codemirror')
      const { EditorState } = await import('@codemirror/state')
      const { python } = await import('@codemirror/lang-python')
      const { oneDark } = await import('@codemirror/theme-one-dark')
      const { keymap } = await import('@codemirror/view')
      const { defaultKeymap } = await import('@codemirror/commands')

      if (!mounted || !containerRef.current) return

      const extensions = [
        basicSetup,
        python(),
        oneDark,
        keymap.of(defaultKeymap),
        EditorView.theme({
          '&': { height, fontSize: '0.875rem' },
          '.cm-content': { fontFamily: 'var(--font-jetbrains), monospace' },
        }),
      ]

      if (readOnly) {
        extensions.push(EditorState.readOnly.of(true))
      }

      if (onChange) {
        extensions.push(
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              onChange(update.state.doc.toString())
            }
          })
        )
      }

      const state = EditorState.create({ doc: value, extensions })
      editorRef.current = new EditorView({
        state,
        parent: containerRef.current,
      })
    }

    initCodeMirror()

    return () => {
      mounted = false
      if (editorRef.current) {
        editorRef.current.destroy()
        editorRef.current = null
      }
    }
  }, [useMonaco]) // Re-init when toggling

  // ─── Monaco (opt-in, lazy loaded) ─────────────────────────────────────────
  useEffect(() => {
    if (!useMonaco) return

    let mounted = true
    let monacoEditor: any = null

    async function initMonaco() {
      if (!containerRef.current || !mounted) return

      // Dynamic import — Monaco is heavy (~2MB), only load when opted in
      const monaco = await import('monaco-editor')

      if (!mounted || !containerRef.current) return

      monacoEditor = monaco.editor.create(containerRef.current, {
        value,
        language: 'python',
        theme: 'vs-dark',
        readOnly,
        minimap: { enabled: false },
        fontSize: 14,
        fontFamily: 'var(--font-jetbrains), monospace',
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        padding: { top: 12 },
      })

      if (onChange) {
        monacoEditor.onDidChangeModelContent(() => {
          onChange(monacoEditor.getValue())
        })
      }

      editorRef.current = monacoEditor
    }

    initMonaco()

    return () => {
      mounted = false
      if (monacoEditor) {
        monacoEditor.dispose()
        monacoEditor = null
      }
    }
  }, [useMonaco])

  return (
    <div className="relative">
      {/* Monaco toggle */}
      <div className="absolute right-2 top-2 z-10">
        <button
          onClick={() => setUseMonaco(!useMonaco)}
          className="flex items-center gap-1 rounded px-2 py-1 text-[10px] text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50 transition"
          title={useMonaco ? 'Switch to CodeMirror (lightweight)' : 'Switch to Monaco (advanced)'}
        >
          <Settings2 size={10} />
          {useMonaco ? 'CodeMirror' : 'Monaco'}
        </button>
      </div>

      <div
        ref={containerRef}
        className="overflow-hidden rounded-lg border border-zinc-800"
        style={{ height }}
      />
    </div>
  )
}
