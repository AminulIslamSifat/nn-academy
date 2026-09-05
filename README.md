# NumPy Neural Network Academy

> Pure NumPy. Zero frameworks. Learn deep learning from the ground up.

An interactive curriculum that teaches neural networks by building them with raw array math. 65 chapters across 10 tracks, in-browser Python execution via WebAssembly, and rigorous assessments — no PyTorch, no TensorFlow, just you and `numpy`.

***

## Curriculum

| Track | Chapters | Description |
|---|---|---|
| 📐 NumPy Foundations | 12 | Arrays, shapes, indexing, slicing, dtypes, copy vs view |
| ⚡ Vectorization & Broadcasting | 2 | Eliminating loops, broadcasting rules |
| 🔧 Universal Functions (ufunc) | 12 | Arithmetic, trig, logs, set ops, custom ufuncs |
| 🧠 Neural Network Primitives | 1 | Linear layer from scratch |
| 🌱 NN: Beginner | 8 | Activations, loss, backprop, optimizers, MNIST, regularization |
| 🔥 NN: Intermediate | 10 | Conv2D, pooling, batchnorm, RNN, LSTM, seq2seq, OCR capstone |
| 🚀 NN: Advanced | 10 | Attention, transformers, GPT from scratch, RLHF, scaling laws |
| 🎲 Random & Distributions | 4 | Sampling, probability, Monte Carlo, noise augmentation |
| ∂ Calculus & Autodiff | 3 | Derivatives, chain rule, computational graphs |
| 🏋️ Training & Optimization | 3 | Mini-batch, gradient clipping, initialization strategies |

***

## Tech Stack

- **Next.js 14** — App Router, static export
- **MDX** — Chapter content with embedded interactive components
- **CodeMirror 6 / Monaco** — In-browser Python editor
- **Pyodide (WASM)** — Client-side NumPy execution, no server
- **KaTeX** — Math rendering
- **Zustand + Dexie** — Progress tracking (localStorage + IndexedDB)
- **Framer Motion** — Animations
- **Tailwind CSS + Typography** — Styling

***

## Getting Started

### Prerequisites

- Node.js ≥ 18
- npm or pnpm

### Install & Run

```bash
git clone https://github.com/AminulIslamSifat/nn-academy
cd nn-academy
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Build for Production

```bash
npm run build    # validates content first, then builds
npm start        # preview production build
```

### Validate Content

```bash
npm run validate   # runs scripts/validate-content.mjs
```

This checks all 65 chapter MDX files for structural correctness before build.

***

## Project Structure

```
src/
├── app/              # Next.js App Router pages
│   ├── chapters/     # Dynamic chapter routes
│   ├── layout.tsx
│   └── page.tsx      # Landing page
├── components/       # Interactive UI (editors, visualizers, quizzes)
├── content/
│   ├── chapters/     # 65 MDX chapter files
│   └── manifest.json # Track/chapter registry
├── lib/              # MDX loader, utils, helpers
├── styles/           # Global CSS, Tailwind overrides
└── workers/          # Web Workers for Pyodide
scripts/
├── validate-content.mjs   # CI/content validation
├── generate_chapters_batch1.py
├── gen_ufuncs.py
├── expand_quizzes.py
└── fix_quiz_code.py
```

***

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start dev server |
| `npm run build` | Validate + production build |
| `npm run validate` | Check chapter MDX integrity |
| `python scripts/generate_chapters_batch1.py` | Batch-generate chapter drafts |
| `python scripts/gen_ufuncs.py` | Generate ufunc chapter content |
| `python scripts/expand_quizzes.py` | Expand quiz questions |
| `python scripts/fix_quiz_code.py` | Fix code blocks in quizzes |

***

## Contributing

1. Edit or add chapters in `src/content/chapters/`
2. Register new chapters in `src/content/manifest.json`
3. Run `npm run validate` to verify structure
4. Test locally with `npm run dev`

Chapter MDX files support frontmatter, KaTeX math, and embedded React components for interactive exercises.

***

## License

Private project. All rights reserved.
