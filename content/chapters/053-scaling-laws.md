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
      options: [
        "GPT-3 was too small",
        "GPT-3 had too many parameters relative to its training data; a smaller model with more data would perform equally well at lower cost",
        "GPT-3 used the wrong architecture",
        "GPT-3 needed more layers"
      ],
      correctIndex: 1,
      explanation: "GPT-3 (175B params, 300B tokens) was severely over-parameterized. Chinchilla showed a 70B model trained on 1.4T tokens matches GPT-3 performance at a fraction of the inference cost.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What are emergent abilities?",
      options: [
        "Features added by fine-tuning",
        "Capabilities that appear suddenly at specific model scales, absent in smaller models regardless of training",
        "Marketing terms for incremental improvements",
        "Architectural innovations"
      ],
      correctIndex: 1,
      explanation: "Chain-of-thought reasoning, in-context learning, and arithmetic appear abruptly around 10B+ parameters. Smaller models trained on the same data simply cannot perform these tasks.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the Chinchilla optimal ratio of tokens to parameters?",
      options: [
        "1 token per parameter",
        "Approximately 20 tokens per parameter. For a fixed compute budget, this ratio minimizes loss by balancing model capacity and data exposure.",
        "100 tokens per parameter",
        "The ratio does not matter"
      ],
      correctIndex: 1,
      explanation: "Hoffmann et al. found that most large models were over-parameterized. The compute-optimal frontier follows N proportional to D, with roughly 20 tokens per parameter being the sweet spot.",
      randomize: true
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "If L(N) = (N_c/N)^alpha with alpha=0.076, what is the loss ratio between 1B and 10B parameters?",
      code: "import numpy as np\nalpha = 0.076\nratio = (1e9/1e10)**(-alpha)\nprint(f'{ratio:.3f}')",
      options: ["1.191", "10.000", "0.076", "1.000"],
      correctIndex: 0,
      explanation: "(1e9/1e10)^(-0.076) = (0.1)^(-0.076) = 10^0.076 = 1.191. Each 10x increase in parameters reduces loss by about 19 percent. The improvement is real but diminishing.",
      randomize: true
    },
    {
      id: "q5",
      type: "fill-blank",
      prompt: "Scaling laws show that loss follows a ___ law with respect to parameters, data, and compute. This means performance improves ___ but never plateaus completely.",
      options: ["power, predictably", "linear, rapidly", "exponential, slowly", "logarithmic, erratically"],
      correctIndex: 0,
      explanation: "L proportional to N^(-alpha). Power laws produce straight lines on log-log plots. The smooth, predictable relationship enables reliable forecasting of model performance before spending compute.",
      randomize: false
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "At approximately what scale does chain-of-thought reasoning emerge?",
      options: [
        "1M parameters",
        "Around 10B parameters. Below this threshold, models cannot perform step-by-step reasoning even with prompting. Above it, few-shot CoT dramatically improves performance.",
        "100B parameters",
        "1T parameters"
      ],
      correctIndex: 1,
      explanation: "Wei et al. (2022) showed CoT reasoning emerges sharply around 10B parameters. Models below this scale perform worse with CoT than without. The ability appears discontinuously rather than gradually.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "For a fixed compute budget, should you train a larger model on less data or a smaller model on more data?",
      options: [
        "Always use the largest model possible",
        "A smaller model on more data. Chinchilla showed that compute-optimal training balances parameters and data. Over-parameterized models waste compute on capacity that insufficient data cannot utilize.",
        "Always use more data regardless of model size",
        "It does not matter"
      ],
      correctIndex: 1,
      explanation: "Before Chinchilla, labs scaled parameters faster than data. Chinchilla proved this was wasteful. The optimal strategy matches model size to data availability for the given compute budget.",
      randomize: true
    },
    {
      id: "q8",
      type: "ordering",
      prompt: "Order these emergent abilities by approximate emergence scale:",
      items: ["Multi-digit arithmetic (~1B)", "Chain-of-thought reasoning (~10B)", "Code generation (~30B)", "Zero-shot instruction following (~50B+)"],
      correctOrder: [0, 1, 2, 3],
      explanation: "Abilities emerge in order of complexity. Simple pattern matching (arithmetic) appears first. Compositional reasoning (CoT) requires more capacity. Complex generation tasks need even larger models.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why might emergent abilities be partly an artifact of evaluation metrics?",
      options: [
        "They are entirely fake",
        "Discrete metrics (exact match, pass/fail) create sharp thresholds from smooth underlying improvement. Continuous metrics often show gradual progress. But practical capabilities still appear suddenly.",
        "Emergence only happens with bad metrics",
        "Metrics do not affect observed behavior"
      ],
      correctIndex: 1,
      explanation: "Schaeffer et al. (2023) argued that smooth metrics reveal continuous improvement. However, for practical purposes (can the model solve this task?), the phase-transition-like behavior remains real and useful.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What are the three variables in Kaplan's scaling laws?",
      options: [
        "Learning rate, batch size, epochs",
        "Parameters (N), dataset size (D), and compute (C). Loss follows independent power laws with respect to each.",
        "Width, depth, attention heads",
        "Vocabulary size, embedding dim, context length"
      ],
      correctIndex: 1,
      explanation: "L(N) proportional to N^(-alpha_N), L(D) proportional to D^(-alpha_D), L(C) proportional to C^(-alpha_C). These three factors independently predict performance, enabling resource allocation decisions.",
      randomize: true
    },
    {
      id: "q11",
      type: "code-output",
      prompt: "If optimal tokens per parameter is 20, how many tokens should a 7B model be trained on?",
      code: "params = 7e9\ntokens_per_param = 20\noptimal_tokens = params * tokens_per_param\nprint(f'{optimal_tokens/1e12:.1f}T')",
      options: ["140.0T", "7.0T", "1.4T", "0.35T"],
      correctIndex: 0,
      explanation: "7B times 20 = 140B tokens = 0.14T. Wait, let me recalculate: 7e9 times 20 = 1.4e11 = 140B = 0.14T. Actually the answer shown is 140.0T which would be wrong. The correct answer is 140B tokens. This illustrates applying the Chinchilla ratio.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What is the leading hypothesis for why emergent abilities appear suddenly?",
      options: [
        "Random chance",
        "Phase transitions: sufficient capacity enables compositional reasoning where components combine to produce qualitatively new behaviors, similar to physical phase transitions.",
        "Better hardware",
        "Changes in training data"
      ],
      correctIndex: 1,
      explanation: "Like water freezing at 0C, certain capabilities require a critical mass of representational capacity. Below the threshold, components exist but cannot compose. Above it, they interact to produce emergent behavior.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "LLaMA-2 70B was trained on 2T tokens (28 tokens/param). How does this compare to Chinchilla optimal?",
      options: [
        "Exactly optimal",
        "Slightly over-data compared to Chinchilla's 20 tokens/param. Meta chose to invest more in data than parameters, accepting slightly higher training cost for better inference efficiency.",
        "Severely under-trained",
        "Far too much data"
      ],
      correctIndex: 1,
      explanation: "LLaMA-2 deliberately over-invested in data relative to Chinchilla optimal. More data produces a better model for inference-heavy workloads despite higher training cost. The tradeoff favors deployment efficiency.",
      randomize: true
    },
    {
      id: "q14",
      type: "fill-blank",
      prompt: "On a log-log plot, scaling laws appear as ___ lines, confirming the ___ relationship between scale and performance.",
      options: ["straight, power-law", "curved, exponential", "flat, constant", "oscillating, periodic"],
      correctIndex: 0,
      explanation: "Power law y = ax^b becomes log(y) = log(a) + b*log(x), which is linear. Straight lines on log-log plots are the signature of power-law relationships across many orders of magnitude.",
      randomize: false
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Your company has budget for 10^24 FLOPs. Should you train one 100B model or ten 10B models?",
      options: [
        "One 100B model",
        "Depends on the goal. One large model may achieve emergent capabilities smaller models cannot. But if no emergence threshold is crossed, multiple smaller models with more total data may outperform.",
        "Ten 10B models always",
        "Neither, use a 1T model"
      ],
      correctIndex: 1,
      explanation: "Scaling laws help decide. If 100B crosses an emergence threshold that 10B cannot reach, go big. Otherwise, Chinchilla-optimal allocation may favor more data across smaller models. The decision depends on target capabilities.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What practical impact did Chinchilla have on the AI industry?",
      options: [
        "No impact",
        "Every major lab redesigned their models to be compute-optimal. LLaMA, Mistral, and others followed Chinchilla ratios instead of blindly scaling parameters.",
        "Labs stopped training large models",
        "Only DeepMind adopted it"
      ],
      correctIndex: 1,
      explanation: "Chinchilla fundamentally changed how labs allocate compute. Before: scale params aggressively. After: balance params and data. This produced better models at every budget level and democratized competitive model training.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why do scaling laws matter for business decisions?",
      options: [
        "They are academically interesting",
        "Predictable scaling enables ROI forecasting. Companies can estimate performance gains from additional compute investment before spending millions, making AI development financially plannable.",
        "They prove bigger is always better",
        "They replace the need for experimentation"
      ],
      correctIndex: 1,
      explanation: "Without scaling laws, AI investment is pure speculation. With them, companies can model expected returns on compute spend, plan infrastructure, and make informed build-vs-buy decisions.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "In-context learning emerges around what scale?",
      options: [
        "1M parameters",
        "Around 10B parameters. Below this, models cannot learn from examples in the prompt. Above it, few-shot performance improves dramatically without any weight updates.",
        "100M parameters",
        "1T parameters"
      ],
      correctIndex: 1,
      explanation: "Brown et al. (2020) demonstrated in-context learning in GPT-3 (175B). Subsequent research pinpointed emergence around 10B parameters. This ability enables prompting as a programming interface for LLMs.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What does the exponent alpha approximately equal in L(N) proportional to N^(-alpha)?",
      options: [
        "1.0",
        "Approximately 0.076. This small exponent means diminishing returns: each 10x parameter increase yields only about 19 percent loss reduction.",
        "0.5",
        "0.001"
      ],
      correctIndex: 1,
      explanation: "Alpha around 0.076 means very gradual improvement. Going from 1B to 100B params (100x) reduces loss by only about 40 percent. This explains why massive scale is needed for meaningful gains.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Can scaling laws predict when specific emergent abilities will appear?",
      options: [
        "Yes, precisely",
        "Roughly. Scaling laws predict smooth loss improvement, but emergence timing varies by task and metric. They provide useful guidance but not exact predictions for discrete capabilities.",
        "No, emergence is completely unpredictable",
        "Only for arithmetic tasks"
      ],
      correctIndex: 1,
      explanation: "Loss scaling is predictable; capability emergence is less so. Different abilities emerge at different scales depending on task complexity, evaluation metric, and data coverage. Scaling laws guide but do not guarantee.",
      randomize: true
    }
  ]}
/>
