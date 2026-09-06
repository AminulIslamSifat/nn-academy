# Academia

> Learn by building. Two rigorous tracks taught from first principles — no hand-waving.

An interactive curriculum with in-browser execution, live visualizers, and rigorous assessments. 88 chapters across 2 modules and 18 tracks. No server, no setup — everything runs in the browser.

***

## Modules

### 🧠 Neural Networks — *Pure NumPy · Zero Frameworks*

Build deep learning from the ground up with raw array math. 57 chapters across 10 tracks.

| Track | Chapters | Description |
|---|---|---|
| 📐 NumPy Foundations | 12 | Arrays, shapes, indexing, slicing, dtypes, copy vs view |
| ⚡ Vectorization & Broadcasting | 2 | Eliminating loops, broadcasting rules |
| 🔧 Universal Functions (ufunc) | 4 | Creation, math ops, aggregation/calculus, trig & sets |
| 🧠 Neural Network Primitives | 1 | Linear layer from scratch |
| 🌱 NN: Beginner | 8 | Activations, loss, backprop, optimizers, MNIST, regularization |
| 🔥 NN: Intermediate | 10 | Conv2D, pooling, batchnorm, RNN, LSTM, seq2seq, OCR capstone |
| 🚀 NN: Advanced | 10 | Attention, transformers, GPT from scratch, RLHF, scaling laws |
| 🎲 Random & Distributions | 4 | Sampling, probability, Monte Carlo, noise augmentation |
| ∂ Calculus & Autodiff | 3 | Derivatives, chain rule, computational graphs |
| 🏋️ Training & Optimization | 3 | Mini-batch, gradient clipping, initialization strategies |

### 🛡️ Cybersecurity — *Offense · Defense · First Principles*

Learn to think like an attacker and defend like a professional. 31 chapters across 8 tracks.

| Track | Chapters | Description |
|---|---|---|
| 🛡️ Security Foundations | 4 | Vocabulary, threat landscape, CIA triad, risk management |
| 🌐 Network & Crypto Security | 4 | Network attacks, firewalls/IDS/IPS, applied cryptography, wireless |
| 🧩 Web & Application Security | 4 | OWASP Top 10, injection, authentication, secure coding |
| 🎯 Offensive & Defensive | 3 | Penetration testing, malware analysis, incident response |
| ☁️ Cloud & DevSecOps Security | 4 | Shared responsibility, IAM, containers/K8s, CI/CD security |
| 📊 Security Operations | 4 | SOC/SIEM, threat intelligence, forensics, zero trust |
| 💻 Hands-On Security Tooling | 4 | Nmap, proxy testing, password attacks, Python automation |
| ⚖️ Governance & Human Factor | 4 | Compliance, architecture, social engineering, careers |

***

## Tech Stack

- **Static site** — plain HTML/CSS/JS, no framework
- **Custom MDX → HTML build** — `build.mjs` converts chapter Markdown into static pages
- **Pyodide (WASM)** — client-side Python/NumPy execution, no server
- **KaTeX** — math rendering
- **Lucide** — icons
- **Tailwind CSS + Typography** — styling

***

## Getting Started

### Prerequisites

- Node.js ≥ 18

### Build the chapters

```bash
node build.mjs
```

This reads `content/manifest.json`, converts every `content/chapters/*.md` file via `chapter-template.html`, and writes the results into `chapters/`.

### Serve locally

Any static server works, for example:

```bash
python -m http.server 8000
# or
npx serve .
```

Then open [http://localhost:8000](http://localhost:8000).

***

## Project Structure

```
content/
├── chapters/     # Markdown chapter sources (NN + Cybersecurity)
└── manifest.json # Module → track → chapter registry
css/
└── styles.css    # Global theme
js/
├── home.js       # Home-page module/track rendering
├── chapter.js    # Sidebar + widget mounting
├── widgets.js    # PyRunner, quiz, visualizer, KaTeX, callout
├── store.js      # Progress persistence
└── pyodide*.js   # In-browser Python runtime
build.mjs         # MDX → static HTML build script
chapter-template.html
index.html        # Landing page
```

***

## Adding Content

1. Add or edit a chapter in `content/chapters/`.
2. Register its slug under the right module/track in `content/manifest.json`.
3. Run `node build.mjs` to regenerate the static pages.

***

## License

Private project. All rights reserved.
