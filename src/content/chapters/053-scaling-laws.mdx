---
title: "Scaling Laws & Emergent Abilities"
slug: "053-scaling-laws"
description: "Predictable relationships between compute, data, parameters, and performance. Understand why bigger models suddenly unlock new capabilities."
track: "nn-advanced"
order: 8
read_time: 20
code_time: 10
execution_timeout: 10
prerequisites: ["050-gpt-from-scratch"]
---

# Scaling Laws & Emergent Abilities

The most surprising finding in deep learning: ==performance follows predictable power laws==. Double the compute → predictable improvement. No architectural changes needed.

## Neural Scaling Laws

Kaplan et al. (2020) discovered:

<BlockMath latex="L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}" />

where <InlineMath latex="L" /> is loss, <InlineMath latex="N" /> is parameters, <InlineMath latex="D" /> is dataset size, <InlineMath latex="C" /> is compute.

<PyRunner
  cellId="053-cell-1"
  defaultCode={`import numpy as np

# Simulate scaling law: L(N) = (N_c / N)^alpha
N_c = 8.8e13  # critical parameter count
alpha = 0.076

params = [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]
losses = [(N_c / N)**alpha for N in params]

print("Neural Scaling Law: L(N) ∝ N^(-α)")
print(f"{'Parameters':>12} | {'Loss':>8} | Improvement")
print("─" * 45)
prev = None
for N, L in zip(params, losses):
    imp = f"{(prev/L):.2f}×" if prev else "—"
    label = {1e6:"1M",1e7:"10M",1e8:"100M",1e9:"1B",1e10:"10B",1e11:"100B"}[N]
    print(f"{label:>12} | {L:8.3f} | {imp}")
    prev = L

print(f"\n💡 Each 10× increase in params → ~{(10**alpha):.2f}× lower loss")
print(f"   Predictable, smooth, no surprises... until emergent abilities!")
`}
/>

## Chinchilla Scaling

Hoffmann et al. (2022) refined: ==parameters and data should scale together==. Most models were under-trained (too many params, too little data).

Optimal ratio: ~20 tokens per parameter.

| Model | Params | Tokens | Ratio | Efficient? |
|-------|--------|--------|-------|------------|
| GPT-3 | 175B | 300B | 1.7 | ❌ Over-parameterized |
| Chinchilla | 70B | 1.4T | 20 | ✅ Optimal |
| LLaMA-2 | 70B | 2T | 28 | ✅ Slightly over-data |

> [!IMPORTANT] Practical Implication
> For a fixed compute budget, it's better to train a smaller model on more data than a larger model on less data. This changed how every major lab designs models.

## Emergent Abilities

Some capabilities appear ==suddenly== at specific scales:

| Ability | Emerges At | Example |
|---------|-----------|---------|
| Arithmetic | ~1B params | Multi-digit addition |
| Chain-of-thought | ~10B params | Step-by-step reasoning |
| In-context learning | ~10B params | Few-shot from examples |
| Code generation | ~30B params | Functional programs |
| Instruction following | ~50B+ params | Zero-shot task completion |

<PyRunner
  cellId="053-cell-2"
  defaultCode={`import numpy as np

# Simulate emergent ability (sharp phase transition)
scales = np.logspace(6, 11, 50)  # 1M to 100B
threshold = 1e10  # 10B params

# Smooth scaling + sharp emergence
smooth_perf = 0.3 * np.log10(scales / 1e6) / 5
emergent = 1 / (1 + np.exp(-(np.log10(scales) - np.log10(threshold)) * 3))
total = np.clip(smooth_perf + 0.5 * emergent, 0, 1)

print("Emergent Ability Simulation:")
print(f"{'Scale':>10} | {'Performance':>11} | Note")
print("─" * 42)
checkpoints = [1e6, 1e7, 1e8, 1e9, 5e9, 1e10, 5e10, 1e11]
for s in checkpoints:
    idx = np.argmin(np.abs(scales - s))
    p = total[idx]
    note = "🔥 EMERGENCE" if abs(s - threshold)/threshold < 0.5 else ""
    label = f"{s/1e9:.0f}B" if s >= 1e9 else f"{s/1e6:.0f}M"
    print(f"{label:>10} | {p:11.3f} | {note}")

print(f"\n⚡ Performance jumps sharply around {threshold/1e9:.0f}B parameters")
print(f"   Below threshold: gradual improvement")
print(f"   Above threshold: qualitatively new capabilities")
`}
/>

## Why Do Emergent Abilities Appear?

Leading hypotheses:

1. **Phase transitions**: Sufficient capacity enables compositional reasoning
2. **Metric artifacts**: Smooth underlying improvement crosses task-specific thresholds
3. **Data coverage**: Larger models see enough examples of rare patterns
4. **Representation richness**: More dimensions enable disentangled concepts

> [!NOTE] Debate
> Some researchers argue emergence is partly an artifact of discrete evaluation metrics. Continuous metrics often show smooth improvement. But the practical reality remains: certain USEFUL capabilities only appear above specific scales.

<Quiz
  chapterSlug="053-scaling-laws"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What did Chinchilla scaling reveal about GPT-3?",
      options: ["GPT-3 was too small", "GPT-3 had too many parameters relative to its training data; a smaller model with more data would perform equally well at lower cost", "GPT-3 used the wrong architecture", "GPT-3 needed more layers"],
      correctIndex: 1,
      explanation: "GPT-3 (175B params, 300B tokens) was severely over-parameterized. Chinchilla showed a 70B model trained on 1.4T tokens matches GPT-3 performance at a fraction of the inference cost.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What are emergent abilities?",
      options: ["Features added by fine-tuning", "Capabilities that appear suddenly at specific model scales, absent in smaller models regardless of training", "Marketing terms", "Architectural innovations"],
      correctIndex: 1,
      explanation: "Chain-of-thought reasoning, in-context learning, and arithmetic appear abruptly around 10B+ parameters. Smaller models trained on the same data simply cannot perform these tasks.",
      randomize: false,
    }
  ]}
/>
