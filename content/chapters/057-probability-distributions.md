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
      prompt: "What does rng.uniform(low=0, high=1, size=1000) sample from?",
      options: ["Normal distribution", "Uniform distribution where every value in [0, 1) has equal probability", "Bernoulli distribution", "Exponential distribution"],
      correctIndex: 1,
      explanation: "Uniform distribution: flat PDF over [low, high). Every sub-interval of equal width has equal probability. Used for Xavier init and random data augmentation parameters.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why is the normal distribution so important in machine learning?",
      options: ["It's the only distribution NumPy supports", "The central limit theorem: sums of many independent random variables tend toward normal. Since neurons compute weighted sums, normal initialization keeps activations well-behaved.", "It's computationally cheapest", "All real-world data is normally distributed"],
      correctIndex: 1,
      explanation: "CLT makes normal the natural distribution for aggregated signals. Weight initialization, noise injection, and latent spaces all leverage this property.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "Approximately what values do you expect for mean and std?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nsamples = rng.standard_normal(100000)\nprint(f'Mean: {samples.mean():.4f}')\nprint(f'Std:  {samples.std():.4f}')",
      options: ["Mean ≈ 0, Std ≈ 1", "Mean ≈ 1, Std ≈ 0", "Mean ≈ 0.5, Std ≈ 0.5", "Mean ≈ -1, Std ≈ 2"],
      correctIndex: 0,
      explanation: "standard_normal samples from N(0, 1). With 100K samples, empirical mean and std converge very close to theoretical values 0 and 1.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the difference between rng.normal(loc=0, scale=0.01) and rng.standard_normal()?",
      options: ["No difference", "standard_normal always has μ=0, σ=1. normal() lets you specify custom mean (loc) and standard deviation (scale).", "normal is faster", "standard_normal uses uniform internally"],
      correctIndex: 1,
      explanation: "standard_normal() is a convenience function for N(0,1). normal(loc, scale) generalizes to any mean and variance. For weight init, you typically need custom scale.",
      randomize: true
    },
    {
      id: "q5",
      type: "fill-blank",
      prompt: "He initialization for ReLU networks uses std = sqrt(___ / fan_in). Xavier/Glorot initialization uses std = sqrt(___ / (fan_in + fan_out)).",
      options: ["2, 2", "1, 1", "2, 1", "1, 2"],
      correctIndex: 0,
      explanation: "He: σ = √(2/n_in) compensates for ReLU halving variance. Xavier: σ = √(2/(n_in+n_out)) balances forward and backward signal for symmetric activations like tanh.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Why does He initialization use std = sqrt(2/fan_in) specifically for ReLU?",
      options: ["It's an arbitrary convention", "ReLU zeros ~half the activations, halving variance. The factor of 2 compensates, maintaining unit variance through layers.", "It prevents overflow", "It matches the uniform distribution"],
      correctIndex: 1,
      explanation: "For ReLU with symmetric input, E[ReLU(x)²] = E[x²]/2. To keep output variance = input variance, we need Var(w) = 2/fan_in instead of 1/fan_in.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "Which distribution generates dropout masks?",
      options: ["Normal", "Uniform", "Bernoulli — each neuron independently kept with probability p", "Poisson"],
      correctIndex: 2,
      explanation: "Dropout = independent Bernoulli trial per neuron. Keep (1) with probability p, drop (0) with probability 1-p. Implemented as (rng.random(shape) < p).astype(float).",
      randomize: true
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What does this simulate?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nflips = rng.binomial(n=1, p=0.7, size=10000)\nprint(f'Fraction of 1s: {flips.mean():.3f}')",
      options: ["≈ 0.7 (coin biased toward heads)", "≈ 0.5 (fair coin)", "≈ 0.3", "≈ 1.0"],
      correctIndex: 0,
      explanation: "binomial(n=1, p=0.7) is Bernoulli(p=0.7). Each trial returns 1 with prob 0.7, 0 with prob 0.3. Mean of many trials converges to p=0.7.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "How do you sample from a categorical distribution using NumPy?",
      options: ["rng.normal()", "rng.choice(categories, p=probabilities) — draws from discrete categories with specified probabilities", "rng.uniform()", "rng.binomial()"],
      correctIndex: 1,
      explanation: "Categorical sampling = weighted random choice. After softmax produces class probabilities, rng.choice selects a class proportional to those probabilities.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "In autoregressive text generation, why sample from the categorical distribution instead of taking argmax?",
      options: ["Sampling is faster", "Argmax always picks the most likely token, producing repetitive/deterministic text. Sampling introduces diversity while respecting the model's confidence.", "Argmax doesn't work with softmax", "Sampling gives lower perplexity"],
      correctIndex: 1,
      explanation: "Greedy decoding (argmax) collapses to mode-seeking behavior. Categorical sampling with temperature control balances quality and diversity.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What happens if you initialize all weights to zero instead of using a random distribution?",
      options: ["Training works fine but slower", "All neurons in a layer compute identical outputs and receive identical gradients — symmetry is never broken, and the network can't learn", "The model converges faster", "Zero init is recommended for ReLU"],
      correctIndex: 1,
      explanation: "Symmetric initialization → symmetric gradients → symmetric updates forever. Random initialization breaks symmetry so each neuron can specialize.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "For a layer with fan_in=784 and fan_out=256 using He initialization, what is the standard deviation?",
      code: "import numpy as np\nfan_in = 784\nstd = np.sqrt(2.0 / fan_in)\nprint(f'{std:.4f}')",
      options: ["≈ 0.0505", "≈ 0.0357", "≈ 0.0714", "≈ 0.0253"],
      correctIndex: 0,
      explanation: "σ = √(2/784) = √0.002551 ≈ 0.0505. This small std ensures initial activations don't explode or vanish through the layer.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "When would you use uniform initialization instead of normal?",
      options: ["Never — normal is always better", "Xavier/Glorot initialization can use either. Uniform bounds guarantee no extreme outlier weights, which some architectures prefer.", "Only for bias terms", "Uniform is deprecated"],
      correctIndex: 1,
      explanation: "Xavier uniform uses U[-√(6/(n_in+n_out)), +√(6/(n_in+n_out))]. Bounded support prevents rare but extreme initial weights that normal can produce.",
      randomize: true
    },
    {
      id: "q14",
      type: "fill-blank",
      prompt: "A GAN's latent vector z is typically sampled from ___(0, ___). This provides a smooth, continuous space for the generator to map from.",
      options: ["N, 1", "U, 1", "Bernoulli, 0.5", "Poisson, λ"],
      correctIndex: 0,
      explanation: "GANs sample z ~ N(0, I) — standard multivariate normal. The smooth, unbounded space allows interpolation between generated samples and covers diverse outputs.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "You're adding noise to training data for regularization. Which distribution and scale is typical?",
      options: ["Uniform [0, 1]", "Normal with σ = 0.01-0.1, scaled to match the data range", "Bernoulli p=0.5", "Exponential λ=1"],
      correctIndex: 1,
      explanation: "Gaussian noise with small σ adds gentle perturbation without dominating the signal. Scale should be much smaller than data variance to regularize without corrupting.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What's wrong with initializing weights from N(0, 1) for a deep network?",
      options: ["Nothing — it's ideal", "Variance grows multiplicatively through layers: after L layers, activation variance ≈ (fan_in)^L, causing exploding activations and NaN losses", "N(0,1) is too narrow", "Normal distribution can't be used for weights"],
      correctIndex: 1,
      explanation: "Each layer multiplies variance by fan_in × Var(w). With Var(w)=1 and fan_in=256, variance doubles every layer. Scaled init (He/Xavier) keeps variance ≈ 1 throughout.",
      randomize: true
    },
    {
      id: "q17",
      type: "code-output",
      prompt: "What does this demonstrate about categorical sampling?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nprobs = np.array([0.1, 0.1, 0.1, 0.7])\nsamples = rng.choice(4, size=10000, p=probs)\ncounts = np.bincount(samples, minlength=4)\nprint((counts / 10000).round(2))",
      options: ["[0.1, 0.1, 0.1, 0.7] — empirical frequencies match theoretical probabilities", "[0.25, 0.25, 0.25, 0.25] — uniform", "[0.0, 0.0, 0.0, 1.0] — always picks max", "Error: invalid probabilities"],
      correctIndex: 0,
      explanation: "Law of large numbers: with enough samples, empirical frequencies converge to true probabilities. Category 3 appears ~70% of the time as expected.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Which initialization is appropriate for a network using tanh activations?",
      options: ["He initialization (σ = √(2/n_in))", "Xavier/Glorot initialization (σ = √(2/(n_in+n_out))) because tanh is symmetric around 0", "N(0, 1)", "All zeros"],
      correctIndex: 1,
      explanation: "tanh is symmetric (unlike ReLU), so no factor-of-2 correction needed. Xavier balances forward and backward variance for symmetric activations.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "In reinforcement learning exploration, why might you use a categorical distribution over actions?",
      options: ["Actions are continuous", "Discrete action spaces naturally map to categorical: each action has a probability, and sampling explores stochastically", "RL doesn't use probability", "Only normal works for RL"],
      correctIndex: 1,
      explanation: "Policy networks output action probabilities via softmax. Categorical sampling selects actions proportionally, balancing exploitation (high-prob actions) and exploration.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "A network with 10 layers, each fan_in=256, uses N(0, 0.01) initialization. What problem might arise?",
      options: ["Exploding gradients", "Vanishing activations: σ=0.01 is very small, and variance shrinks multiplicatively through 10 layers, making later-layer activations near-zero", "Perfect initialization", "NaN loss immediately"],
      correctIndex: 1,
      explanation: "Var(output) ≈ (fan_in × σ²)^L = (256 × 0.0001)^10 = 0.0256^10 ≈ 0. Activations collapse to zero. Proper scaled init keeps variance ≈ 1 at every depth.",
      randomize: true
    }
  ]}
/>
