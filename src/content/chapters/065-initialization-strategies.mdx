---
title: "Weight Initialization Strategies"
slug: "065-initialization-strategies"
description: "Why zero initialization fails, how Xavier and He initialization keep signal flowing through deep networks, and practical initialization recipes."
track: "training"
order: 3
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["057-probability-distributions", "030-backpropagation"]
---

# Weight Initialization Strategies

Bad initialization kills deep networks before training even starts. ==The right initialization keeps activations and gradients at reasonable scales across all layers.==

## Why Not Zeros?

If all weights are zero, every neuron in a layer computes the same output and receives the same gradient. They'll never differentiate — ==symmetry is never broken==.

<PyRunner
  cellId="065-cell-1"
  defaultCode={`import numpy as np

# Zero init: all neurons identical
np.random.seed(42)
X = np.random.randn(4, 10)
W_zero = np.zeros((10, 5))
b_zero = np.zeros(5)

out_zero = np.maximum(0, X @ W_zero + b_zero)
print("Zero initialization:")
print(f"  All outputs identical: {np.all(out_zero == out_zero[:, :1])}")
print(f"  All gradients will be identical → neurons never specialize")

# Random init: neurons differentiate
W_rand = np.random.randn(10, 5) * 0.1
out_rand = np.maximum(0, X @ W_rand)
print(f"\nRandom initialization:")
print(f"  Output std across neurons: {out_rand.std(axis=1).mean():.4f}")
print(f"  Neurons produce different outputs ✅")
`}
/>

## Xavier/Glorot Initialization

For tanh/sigmoid activations. Keeps activation variance constant across layers:

<BlockMath latex="W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)" />

## He Initialization

For ReLU activations. Accounts for ReLU zeroing half the values:

<BlockMath latex="W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)" />

<PyRunner
  cellId="065-cell-2"
  defaultCode={`import numpy as np

np.random.seed(42)

# Track activation variance through 10 layers
n_in = 256
depth = 10
X = np.random.randn(64, n_in)

print(f"Activation variance through {depth} layers (64 samples, 256 dims):")
print(f"{'Init':<12} | {'Layer 1':>8} | {'Layer 5':>8} | {'Layer 10':>9} | Stable?")
print("─" * 55)

for name, scale_fn in [
    ("Too small", lambda n: 0.001),
    ("Too large", lambda n: 1.0),
    ("Xavier", lambda n: np.sqrt(2.0/(n+n))),
    ("He", lambda n: np.sqrt(2.0/n)),
]:
    h = X.copy()
    variances = []
    for layer in range(depth):
        W = np.random.randn(n_in, n_in) * scale_fn(n_in)
        h = np.maximum(0, h @ W)  # ReLU
        variances.append(h.var())
    
    stable = "✅" if 0.1 < variances[-1] < 10 else "❌"
    print(f"{name:<12} | {variances[0]:8.4f} | {variances[4]:8.4f} | {variances[9]:9.4f} | {stable}")

print(f"\n💡 He init keeps variance ~1 across all layers with ReLU")
print(f"   Too small → vanishing activations")
print(f"   Too large → exploding activations")
`}
/>

## Initialization Cheat Sheet

| Activation | Init | Formula |
|-----------|------|---------|
| ReLU | He | <InlineMath latex="\sigma = \sqrt{2/n_{in}}" /> |
| Tanh/Sigmoid | Xavier | <InlineMath latex="\sigma = \sqrt{2/(n_{in}+n_{out})}" /> |
| Linear | Xavier | Same as above |
| Batch Norm layers | Any | BN compensates |
| Embedding | Small normal | <InlineMath latex="\sigma = 0.01-0.02" /> |

> [!IMPORTANT] Practical Rule
> Use He initialization for any network with ReLU. Use Xavier for tanh/sigmoid. If using BatchNorm, initialization matters less but He is still a safe default.

<Quiz
  chapterSlug="065-initialization-strategies"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does He initialization use sqrt(2/fan_in) instead of Xavier's sqrt(2/(fan_in+fan_out))?",
      options: ["It's arbitrary", "ReLU zeros out ~half the activations, halving the variance; He compensates by using only fan_in (effectively doubling the variance compared to Xavier)", "He is always better", "Xavier doesn't work with ReLU"],
      correctIndex: 1,
      explanation: "With ReLU, E[ReLU(x)²] ≈ 0.5 × E[x²] for symmetric x. To maintain unit variance through ReLU, the weight variance needs to be 2/fan_in instead of 2/(fan_in+fan_out).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What happens if you initialize all weights to the same small non-zero value (e.g., 0.01)?",
      options: ["Works fine", "All neurons in each layer still compute identical outputs and receive identical gradients — symmetry is not broken", "Better than random", "Only works for CNNs"],
      correctIndex: 1,
      explanation: "Any constant initialization (zero or non-zero) creates symmetry: all neurons are identical and stay identical throughout training. Random initialization breaks this symmetry.",
      randomize: false,
    }
  ]}
/>
