---
title: "Probability Distributions"
slug: "057-probability-distributions"
description: "Sample from uniform, normal, Bernoulli, categorical, and more. Understand which distribution to use for weight initialization, noise, and data generation."
track: "random"
order: 2
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["056-random-sampling"]
---

# Probability Distributions

Every random operation in deep learning draws from a distribution. ==Choosing the right one matters== — wrong initialization distributions cause vanishing/exploding gradients.

## Uniform Distribution

Equal probability across a range:

<BlockMath latex="X \sim U(a, b), \quad f(x) = \frac{1}{b-a}" />

```python
rng.uniform(low=0, high=1, size=1000)
```

Used for: Xavier/Glorot initialization, random crops, dropout masks.

## Normal (Gaussian) Distribution

The bell curve. Most important distribution in ML:

<BlockMath latex="X \sim \mathcal{N}(\mu, \sigma^2), \quad f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}" />

<PyRunner
  cellId="057-cell-1"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

# Standard normal
samples = rng.standard_normal(10000)
print(f"Standard Normal N(0,1):")
print(f"  Mean: {samples.mean():.4f} (expect 0)")
print(f"  Std:  {samples.std():.4f} (expect 1)")
print(f"  Min:  {samples.min():.2f}, Max: {samples.max():.2f}")

# Custom mean and std
weights = rng.normal(loc=0, scale=0.01, size=(100, 50))
print(f"\nWeight init N(0, 0.01):")
print(f"  Shape: {weights.shape}")
print(f"  Mean: {weights.mean():.6f}")
print(f"  Std:  {weights.std():.4f}")

# He initialization for ReLU networks
fan_in = 784
he_weights = rng.normal(0, np.sqrt(2.0/fan_in), size=(fan_in, 256))
print(f"\nHe init for 784→256 ReLU layer:")
print(f"  Std: {he_weights.std():.4f} (expect {np.sqrt(2.0/fan_in):.4f})")
`}
/>

> [!IMPORTANT] Why Normal for Weights?
> The central limit theorem: sums of many small random values tend toward normal. Since each neuron computes a weighted sum, normal initialization keeps activations well-behaved.

## Bernoulli Distribution

Binary outcome (0 or 1) with probability p:

```python
rng.binomial(n=1, p=0.5, size=100)  # coin flips
```

Used for: dropout masks, binary classification targets.

## Categorical Distribution

Pick from K categories with given probabilities:

<PyRunner
  cellId="057-cell-2"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

# Simulate softmax output → sample a class
logits = np.array([2.0, 1.0, 0.5, 0.1])
probs = np.exp(logits) / np.exp(logits).sum()

samples = rng.choice(len(probs), size=10000, p=probs)
counts = np.bincount(samples, minlength=len(probs))

print("Categorical sampling from softmax logits:")
print(f"  Logits: {logits}")
print(f"  Probs:  {probs.round(3)}")
print(f"  Samples (10K): {counts}")
print(f"  Empirical:     {(counts/10000).round(3)}")
print(f"\n✅ Empirical frequencies match theoretical probabilities")
`}
/>

## Distribution Cheat Sheet for Deep Learning

| Use Case | Distribution | Parameters |
|----------|-------------|------------|
| He init (ReLU) | Normal | <InlineMath latex="\sigma = \sqrt{2/n_{in}}" /> |
| Xavier init (tanh) | Normal/Uniform | <InlineMath latex="\sigma = \sqrt{2/(n_{in}+n_{out})}" /> |
| Dropout mask | Bernoulli | p = keep_prob |
| Data augmentation noise | Normal | <InlineMath latex="\sigma = 0.01-0.1" /> |
| GAN latent vector | Normal | N(0, 1) |
| Exploration (RL) | Categorical/Normal | Varies |

<Quiz
  chapterSlug="057-probability-distributions"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does He initialization use std = sqrt(2/fan_in) specifically for ReLU networks?",
      options: ["It's arbitrary", "ReLU zeros out half the activations, so the variance needs to be doubled to compensate; sqrt(2/fan_in) maintains unit variance through ReLU layers", "It prevents overflow", "It matches the uniform distribution"],
      correctIndex: 1,
      explanation: "ReLU sets negative values to zero, halving the variance. He initialization compensates by scaling up by sqrt(2), keeping activation variance stable across layers.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Which distribution generates dropout masks?",
      options: ["Normal", "Uniform", "Bernoulli (each neuron independently kept with probability p)", "Poisson"],
      correctIndex: 2,
      explanation: "Dropout independently keeps each neuron with probability p. That's exactly a Bernoulli trial per neuron: 1 (keep) with prob p, 0 (drop) with prob 1-p.",
      randomize: false,
    }
  ]}
/>
