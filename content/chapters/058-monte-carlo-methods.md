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
      prompt: "What is the core idea of Monte Carlo estimation?",
      options: ["Solve equations analytically", "Estimate integrals and expectations by drawing random samples and averaging — turning impossible math into trivial computation", "Use gradient descent", "Memorize lookup tables"],
      correctIndex: 1,
      explanation: "E[f(X)] ≈ (1/N)Σf(xᵢ) where xᵢ ~ p(x). No closed-form needed. Just sample, evaluate, average. Works in any dimension.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the convergence rate of Monte Carlo estimation?",
      options: ["O(1/n)", "O(1/√n) — error halves when you quadruple the number of samples", "O(1/n²)", "O(log n)"],
      correctIndex: 1,
      explanation: "Standard error = σ/√n by CLT. Dimension-independent convergence is MC's superpower — unlike numerical integration which suffers from curse of dimensionality.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "In the pi estimation, what does np.sum(x**2 + y**2 <= 1) count?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nn = 100000\nx = rng.uniform(-1, 1, n)\ny = rng.uniform(-1, 1, n)\ninside = np.sum(x**2 + y**2 <= 1)\npi_est = 4 * inside / n\nprint(f'Inside: {inside}, π ≈ {pi_est:.4f}')",
      options: ["Points inside the unit circle (distance from origin ≤ 1)", "Total points generated", "Points outside the circle", "Points on the boundary"],
      correctIndex: 0,
      explanation: "x²+y² ≤ 1 defines the unit circle. Points satisfying this are inside. Ratio inside/total ≈ area_circle/area_square = π/4, so π ≈ 4 × ratio.",
      randomize: true
    },
    {
      id: "q4",
      type: "fill-blank",
      prompt: "To estimate E[f(X)] via Monte Carlo, draw N samples xᵢ ~ p(x) and compute ___. The estimate improves as N ___.",
      options: ["(1/N)Σf(xᵢ), increases", "Σf(xᵢ), decreases", "max(f(xᵢ)), increases", "f(mean(x)), increases"],
      correctIndex: 0,
      explanation: "MC estimator = sample mean of f evaluated at random draws. By law of large numbers, converges to true expectation as N → ∞.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Why can't we compute E[sigmoid(x)] for x ~ N(0,1) analytically?",
      options: ["sigmoid is too simple", "The integral ∫ sigmoid(x)·normal_pdf(x) dx has no closed-form solution — Monte Carlo is the practical way to estimate it", "NumPy doesn't support it", "It equals zero"],
      correctIndex: 1,
      explanation: "Many real-world expectations lack analytical solutions. MC bypasses this by replacing integration with sampling. This is why MC is indispensable in ML.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "How is temperature sampling in LLMs related to Monte Carlo?",
      options: ["Unrelated", "Each generated token is a sample from the softmax distribution — the entire generation process is Monte Carlo sampling from the model's learned distribution", "Only beam search uses MC", "MC is only for training"],
      correctIndex: 1,
      explanation: "Autoregressive generation draws each token from P(xₜ|x<t) via categorical sampling. The generated sequence is one MC sample path from the joint distribution.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "Which deep learning application uses Monte Carlo to estimate the evidence lower bound (ELBO)?",
      options: ["GANs", "Variational Autoencoders (VAEs) — the reparameterization trick enables gradient-based optimization of an MC-estimated ELBO", "CNNs", "Decision trees"],
      correctIndex: 1,
      explanation: "VAEs optimize ELBO = E_q[log p(x|z)] - KL(q||p). The expectation is estimated via MC sampling from q(z|x). Reparameterization makes this differentiable.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "In reinforcement learning, what does Monte Carlo estimate?",
      options: ["Model weights", "Expected return (cumulative reward) of a policy by averaging returns from sampled episodes", "State transitions", "Action probabilities"],
      correctIndex: 1,
      explanation: "MC methods in RL estimate V(s) or Q(s,a) by averaging actual returns from complete episodes. No model of environment dynamics needed — pure sample-based learning.",
      randomize: true
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "If you increase samples from 1,000 to 100,000, by roughly what factor does the MC error decrease?",
      code: "import numpy as np\nprint(f'Error ratio: sqrt(1000/100000) = {np.sqrt(1000/100000):.4f}')",
      options: ["10× smaller", "√100 ≈ 10× smaller", "100× smaller", "Same error"],
      correctIndex: 1,
      explanation: "Error ∝ 1/√n. Increasing n by 100× reduces error by √100 = 10×. This slow convergence is MC's main limitation.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What is the connection between dropout and Bayesian inference?",
      options: ["No connection", "Gal & Ghahramani showed dropout at test time approximates Monte Carlo sampling from a Bayesian posterior — enabling uncertainty estimation", "Dropout replaces Bayesian methods entirely", "Bayesian networks don't use dropout"],
      correctIndex: 1,
      explanation: "MC Dropout: run multiple forward passes with dropout enabled at test time. The variance across predictions approximates posterior uncertainty. Cheap Bayesian approximation.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why is Monte Carlo's dimension-independent convergence important?",
      options: ["It isn't — low-dimensional problems are most common", "Numerical integration grids grow exponentially with dimensions (curse of dimensionality). MC's O(1/√n) rate is the same regardless of dimension.", "MC only works in 1D", "Higher dimensions converge faster"],
      correctIndex: 1,
      explanation: "A 100-point grid in 10D needs 100¹⁰ = 10²⁰ evaluations. MC needs only ~10⁴ samples for similar accuracy regardless of dimension. This makes high-dimensional integration feasible.",
      randomize: true
    },
    {
      id: "q12",
      type: "fill-blank",
      prompt: "In the pi estimation, π ≈ ___ × (points_inside / total_points). The factor comes from the ratio of ___ area to ___ area.",
      options: ["4, circle, square", "2, circle, triangle", "π, square, circle", "1, circle, rectangle"],
      correctIndex: 0,
      explanation: "Square area = 4 (from -1 to 1 on each axis). Circle area = π·r² = π. Ratio = π/4. So π = 4 × (inside/total).",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What is the reparameterization trick in VAEs?",
      options: ["Changing the activation function", "Rewriting z ~ N(μ,σ²) as z = μ + σ·ε where ε ~ N(0,1), making the sampling operation differentiable w.r.t. μ and σ", "Using a different optimizer", "Removing the KL term"],
      correctIndex: 1,
      explanation: "Direct sampling z ~ N(μ,σ²) blocks gradients. Reparameterization moves randomness to ε (fixed noise), allowing backprop through μ and σ. Essential for training VAEs.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "How do diffusion models use Monte Carlo?",
      options: ["They don't", "Generation is iterative denoising starting from random Gaussian noise — each step is guided by a learned score function, and the initial noise is an MC sample", "Only for loss computation", "For weight initialization only"],
      correctIndex: 1,
      explanation: "Diffusion models learn to reverse a noise process. Generation starts from z ~ N(0,I) (MC sample) and iteratively denoises. Each generated image is one MC sample from the data distribution.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "You estimate E[x²] for x~N(0,1) with 1000 samples and get 0.95. With 10000 samples you get 1.01. What explains the difference?",
      options: ["Bug in the code", "Monte Carlo estimates are noisy — finite samples give approximate results that fluctuate around the true value (1.0)", "The distribution changed", "10000 samples is too many"],
      correctIndex: 1,
      explanation: "MC estimates have variance ∝ 1/n. Both 0.95 and 1.01 are reasonable estimates of 1.0. More samples reduce but never eliminate noise.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What distinguishes Monte Carlo methods from deterministic numerical integration?",
      options: ["MC is always more accurate", "MC uses random sampling with probabilistic convergence; deterministic methods use fixed grids/quadrature with guaranteed error bounds but suffer from dimensionality", "They're identical", "Deterministic methods are randomized"],
      correctIndex: 1,
      explanation: "MC trades deterministic guarantees for dimension-independence. In low dimensions, quadrature wins. In high dimensions (>5), MC is often the only feasible approach.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "In Bayesian neural networks, what does Monte Carlo approximate?",
      options: ["Forward pass speed", "The posterior predictive distribution: integrating over all possible weight configurations weighted by their posterior probability", "Learning rate scheduling", "Batch normalization"],
      correctIndex: 1,
      explanation: "P(y|x,D) = ∫P(y|x,w)P(w|D)dw. This integral is intractable. MC approximates it by sampling weights from the posterior and averaging predictions.",
      randomize: true
    },
    {
      id: "q18",
      type: "fill-blank",
      prompt: "Beam search approximates the ___ of the output distribution. Temperature sampling approximates the ___ distribution.",
      options: ["mode, full", "mean, partial", "variance, full", "median, truncated"],
      correctIndex: 0,
      explanation: "Beam search finds the highest-probability sequence (mode). Sampling draws diverse sequences from the full distribution. Mode-seeking vs distribution-covering.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Why does the pi estimation use uniform(-1, 1) instead of uniform(0, 1)?",
      options: ["No reason — either works equally", "The unit circle is centered at origin, spanning [-1,1] on both axes. Using [0,1] would only cover one quadrant and need a different formula.", "uniform(0,1) is slower", "Negative numbers improve precision"],
      correctIndex: 1,
      explanation: "Circle x²+y²≤1 spans [-1,1]². Sampling from [0,1]² covers only the first quadrant. You could use [0,1] with π≈4×ratio, but [-1,1] is the natural domain.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which statement best summarizes when to use Monte Carlo?",
      options: ["Always prefer analytical solutions", "Use MC when analytical integration is impossible or impractical (high dimensions, complex distributions, no closed form) — accept noisy estimates that improve with more samples", "Never use MC in production", "MC only works for estimating pi"],
      correctIndex: 1,
      explanation: "MC is the tool of last resort AND first choice for high-dimensional problems. It's ubiquitous in modern ML precisely because most interesting quantities lack closed forms.",
      randomize: true
    }
  ]}
/>
