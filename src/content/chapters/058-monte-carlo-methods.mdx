---
title: "Monte Carlo Methods"
slug: "058-monte-carlo-methods"
description: "Estimate integrals, expectations, and probabilities by random sampling. The foundation of variational inference, reinforcement learning, and Bayesian deep learning."
track: "random"
order: 3
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["057-probability-distributions"]
---

# Monte Carlo Methods

Can't compute an integral analytically? ==Sample randomly and average.== Monte Carlo turns impossible math into trivial computation — at the cost of noise that decreases with more samples.

## Estimating Pi

The classic example. Throw darts at a square with an inscribed circle:

<BlockMath latex="\pi \approx 4 \cdot \frac{\text{points inside circle}}{\text{total points}}" />

<PyRunner
  cellId="058-cell-1"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

for n in [100, 1000, 10000, 100000, 1000000]:
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    inside = np.sum(x**2 + y**2 <= 1)
    pi_est = 4 * inside / n
    error = abs(pi_est - np.pi)
    print(f"  n={n:>8,}: π ≈ {pi_est:.6f} (error: {error:.6f})")

print(f"\n✅ Error decreases as 1/sqrt(n) — the Monte Carlo convergence rate")
`}
/>

## Estimating Expectations

The core idea: <InlineMath latex="E[f(X)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i)" /> where <InlineMath latex="x_i \sim p(x)" />.

<PyRunner
  cellId="058-cell-2"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

# Estimate E[x²] where x ~ N(0,1). Analytical answer: Var(X) + E[X]² = 1
samples = rng.standard_normal(100000)
estimate = np.mean(samples**2)

print(f"E[x²] for x ~ N(0,1):")
print(f"  Monte Carlo: {estimate:.4f}")
print(f"  Analytical:  1.0000")
print(f"  Error:       {abs(estimate - 1.0):.4f}")

# Estimate E[sigmoid(x)] where x ~ N(0,1)
sigmoid = 1 / (1 + np.exp(-samples))
print(f"\nE[sigmoid(x)] for x ~ N(0,1):")
print(f"  Monte Carlo: {sigmoid.mean():.4f}")
print(f"  (No closed form — MC is the ONLY way!)")
`}
/>

## Monte Carlo in Deep Learning

| Application | What's Estimated |
|------------|------------------|
| Variational Autoencoders | Evidence lower bound (ELBO) |
| Reinforcement Learning | Expected return of policies |
| Bayesian Neural Networks | Posterior predictive distribution |
| Dropout as Approximation | Bayesian inference (Gal & Ghahramani) |
| Diffusion Models | Denoising score matching |

> [!NOTE] Why Monte Carlo Matters for LLMs
> Temperature sampling IS Monte Carlo. Each generated token is a sample from the model's predicted distribution. Beam search approximates the mode; sampling approximates the full distribution.

<Quiz
  chapterSlug="058-monte-carlo-methods"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the convergence rate of Monte Carlo estimation?",
      options: ["O(1/n)", "O(1/sqrt(n)) — error halves when you quadruple samples", "O(1/n²)", "O(log n)"],
      correctIndex: 1,
      explanation: "By the central limit theorem, the standard error of the mean scales as sigma/sqrt(n). To halve the error, you need 4x more samples. This is dimension-independent — MC's key advantage.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How is temperature sampling in LLMs related to Monte Carlo?",
      options: ["It's unrelated", "Each generated token is a Monte Carlo sample from the model's softmax distribution; the full generated sequence is a Monte Carlo estimate of the model's output distribution", "Only beam search is Monte Carlo", "Monte Carlo is only for training"],
      correctIndex: 1,
      explanation: "Sampling tokens from the softmax distribution is literally drawing from a categorical distribution — a Monte Carlo process. The generated text is one sample path from the model's learned distribution.",
      randomize: false,
    }
  ]}
/>
