---
title: "Random Sampling in NumPy"
slug: "056-random-sampling"
description: "Generate random numbers, shuffle data, and sample from arrays. Understand seeds, reproducibility, and the new Generator API."
track: "random"
order: 1
read_time: 18
code_time: 12
execution_timeout: 10
prerequisites: ["001-arrays-and-shapes"]
---

# Random Sampling in NumPy

Neural networks start with random weights. Training shuffles data every epoch. Dropout randomly zeros neurons. ==Randomness is everywhere in deep learning== — and it must be reproducible.

## The Generator API (Modern)

NumPy 1.17+ introduced `np.random.Generator` — faster, better statistically, and the recommended way:

```python
import numpy as np

rng = np.random.default_rng(seed=42)
```

<PyRunner
  cellId="056-cell-1"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

print("Uniform [0, 1):", rng.random(5).round(4))
print("Integers [0, 100):", rng.integers(0, 100, 5))
print("Normal (mean=0, std=1):", rng.standard_normal(5).round(4))

# Reproducibility
rng2 = np.random.default_rng(42)
print(f"\nSame seed → same output: {np.array_equal(rng.random(3), rng2.random(3))}")
`}
/>

> [!IMPORTANT] Always Set a Seed
> Without a seed, every run produces different results. For debugging and reproducibility, always set one. In research papers, report your seed.

## Shuffling Data

Critical for training — prevents the model from learning order-dependent patterns:

```python
rng = np.random.default_rng(42)
X = np.arange(10)
rng.shuffle(X)  # in-place
```

<PyRunner
  cellId="056-cell-2"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

# Shuffle X and y together (critical for supervised learning!)
X = np.arange(10)
y = X * 2 + 1

indices = rng.permutation(len(X))
X_shuffled = X[indices]
y_shuffled = y[indices]

print("Original:  ", X)
print("Shuffled:  ", X_shuffled)
print("Labels stay aligned:", y_shuffled)
print(f"\n✅ Use permutation indices to shuffle X and y TOGETHER")
print(f"   Never shuffle them independently!")
`}
/>

## Sampling With and Without Replacement

```python
rng.choice(a, size, replace=True/False, p=probs)
```

<PyRunner
  cellId="056-cell-3"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)
population = np.array([10, 20, 30, 40, 50])

# With replacement (bootstrap)
bootstrap = rng.choice(population, size=8, replace=True)
print(f"Bootstrap sample: {bootstrap}")

# Without replacement (subset)
subset = rng.choice(population, size=3, replace=False)
print(f"Random subset:    {subset}")

# Weighted sampling
probs = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # favor 50
weighted = rng.choice(population, size=10, p=probs)
print(f"Weighted sample:  {weighted}")
print(f"  50 appears {np.sum(weighted==50)}/10 times (expected ~6)")
`}
/>

## Train/Test Split

```python
def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

<Quiz
  chapterSlug="056-random-sampling"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is randomness essential in deep learning?",
      options: ["It isn't — deterministic models are always better", "Weight initialization, data shuffling, dropout, and sampling all require randomness for proper training and generalization", "Only for data augmentation", "Randomness makes models faster"],
      correctIndex: 1,
      explanation: "Without random initialization, all neurons start identical and learn the same thing. Without shuffling, models memorize order. Dropout and sampling are inherently stochastic.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does np.random.default_rng(seed=42) create?",
      options: ["A global random seed", "An independent Generator instance using the PCG64 algorithm, seeded for reproducibility", "A list of 42 random numbers", "A deprecated random function"],
      correctIndex: 1,
      explanation: "default_rng() creates a Generator object with its own state. Unlike the legacy global np.random, each Generator is independent and thread-safe.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nrng1 = np.random.default_rng(42)\nrng2 = np.random.default_rng(42)\na = rng1.random(3)\nb = rng2.random(3)\nprint(np.array_equal(a, b))",
      options: ["True", "False", "Error", "None"],
      correctIndex: 0,
      explanation: "Same seed → same initial state → same sequence of random numbers. This is the foundation of reproducibility in experiments.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why must X and y be shuffled using the SAME permutation indices?",
      options: ["For computational speed", "Shuffling independently breaks the correspondence between inputs and labels — X[0] would get paired with a wrong y value", "NumPy requires it syntactically", "It doesn't matter if they're shuffled separately"],
      correctIndex: 1,
      explanation: "Each X[i] maps to y[i]. Independent shuffling destroys this mapping. Use indices = rng.permutation(n), then X[indices] and y[indices] to keep alignment.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What's the difference between rng.shuffle(X) and rng.permutation(X)?",
      options: ["No difference", "shuffle modifies X in-place and returns None; permutation returns a new shuffled array without modifying the original", "permutation is slower", "shuffle only works on 1D arrays"],
      correctIndex: 1,
      explanation: "shuffle is in-place (memory efficient, destructive). permutation is out-of-place (safe, creates copy). Use permutation when you need the original intact or want index arrays.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "In rng.choice(population, size=8, replace=True), what does replace=True mean?",
      options: ["Replace values in the original array", "Sampling with replacement: the same element can be selected multiple times (bootstrap sampling)", "Replace NaN values", "Use replacement sampling algorithm"],
      correctIndex: 1,
      explanation: "With replacement means each draw is independent — previously selected items go back into the pool. Essential for bootstrap resampling. Without replacement guarantees unique selections.",
      randomize: true
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "How many unique values can appear in this sample?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\npop = np.array([10, 20, 30, 40, 50])\nsample = rng.choice(pop, size=3, replace=False)\nprint(len(sample), len(np.unique(sample)))",
      options: ["3 3", "3 1", "5 3", "3 5"],
      correctIndex: 0,
      explanation: "replace=False guarantees no duplicates. Drawing 3 from 5 without replacement always gives exactly 3 unique values.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does the p parameter in rng.choice control?",
      options: ["Padding value", "Probability weights for each element — higher probability means more likely to be selected", "Number of permutations", "Precision of random numbers"],
      correctIndex: 1,
      explanation: "p must sum to 1.0 and have the same length as the population. Weighted sampling is used in importance sampling, class-balanced batching, and token sampling in language models.",
      randomize: true
    },
    {
      id: "q9",
      type: "fill-blank",
      prompt: "In train_test_split with test_ratio=0.2 and 1000 samples, the split index is int(1000 × ___) = ___, giving ___ training and ___ test samples.",
      options: ["0.8, 800, 800, 200", "0.2, 200, 200, 800", "0.5, 500, 500, 500", "1.0, 1000, 1000, 0"],
      correctIndex: 0,
      explanation: "split = int(n × (1 - test_ratio)) = int(1000 × 0.8) = 800. First 800 shuffled indices → train, last 200 → test.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Why should you report your random seed in research papers?",
      options: ["It's required by law", "Reproducibility: others must be able to replicate your exact results. Different seeds can produce meaningfully different outcomes in small experiments", "Seeds are fashionable", "Journals require it for formatting"],
      correctIndex: 1,
      explanation: "Randomness affects initialization, data ordering, dropout masks, and sampling. Without the seed, results are irreproducible. Always report seeds in experimental sections.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why is the new Generator API preferred over np.random.seed()?",
      options: ["It generates prettier numbers", "Better statistical properties (PCG64), faster generation, independent instances avoid global state bugs in multi-threaded code", "The old API was removed in NumPy 2.0", "It uses less memory"],
      correctIndex: 1,
      explanation: "Legacy np.random uses Mersenne Twister with global mutable state. Generator uses PCG64, is instance-based (no shared state), and passes stricter statistical tests.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What distribution does rng.standard_normal() sample from?",
      options: ["Uniform [0, 1)", "Normal/Gaussian with mean=0 and std=1", "Binomial", "Exponential"],
      correctIndex: 1,
      explanation: "Standard normal: bell curve centered at 0 with unit variance. Used extensively for weight initialization (Xavier, He init) and noise injection.",
      randomize: true
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What range of values can rng.integers(0, 10, size=5) produce?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nresult = rng.integers(0, 10, size=5)\nprint(result.min(), result.max())",
      options: ["0 to 9 (inclusive)", "0 to 10 (inclusive)", "1 to 10", "0 to 9.999"],
      correctIndex: 0,
      explanation: "rng.integers(low, high) samples from [low, high) — exclusive upper bound. So integers(0, 10) gives values in {0, 1, ..., 9}.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "In the train_test_split function, why use rng.permutation instead of rng.shuffle?",
      options: ["permutation is faster", "permutation returns an index array that can be applied to both X and y consistently; shuffle modifies in-place and returns None", "shuffle doesn't work on integers", "No practical difference"],
      correctIndex: 1,
      explanation: "We need the same reordering for both X and y. permutation gives us explicit indices we apply to both arrays. shuffle would require separate calls that break alignment.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What is bootstrap sampling and why is it useful?",
      options: ["Booting up a neural network", "Sampling WITH replacement to create datasets of the same size — used for estimating confidence intervals and model uncertainty", "Removing outliers", "Initializing weights"],
      correctIndex: 1,
      explanation: "Bootstrap draws n samples with replacement from n originals. Some appear multiple times, some not at all. Repeating this gives empirical distributions for statistics like accuracy.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Two researchers run the same experiment with the same seed but get different results. What's the most likely cause?",
      options: ["NumPy is broken", "Different NumPy versions may use different default algorithms in default_rng(); also check if ALL sources of randomness (data loading, GPU ops) are seeded", "Seeds don't actually work", "Hardware differences"],
      correctIndex: 1,
      explanation: "default_rng() defaults changed across versions. Also, PyTorch/TF have separate RNGs. Data loaders may shuffle independently. Full reproducibility requires seeding every source.",
      randomize: true
    },
    {
      id: "q17",
      type: "fill-blank",
      prompt: "rng.random(shape) samples from a ___ distribution on [___, ___). rng.standard_normal(shape) samples from a ___ distribution.",
      options: ["uniform, 0, 1, normal", "normal, 0, 1, uniform", "uniform, -1, 1, normal", "binomial, 0, 1, poisson"],
      correctIndex: 0,
      explanation: "rng.random → Uniform[0,1). rng.standard_normal → Normal(μ=0, σ=1). These are the two most common distributions in ML.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "You want to select 100 samples from a dataset of 10000 for validation. Which approach is correct?",
      options: ["Take the first 100 samples", "rng.choice(10000, size=100, replace=False) to randomly select 100 unique indices", "rng.choice(10000, size=100, replace=True)", "Take every 100th sample"],
      correctIndex: 1,
      explanation: "Random selection without replacement ensures unbiased, non-overlapping validation set. First-100 or every-100th may have systematic bias. With-replacement could pick duplicates.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Why does shuffling data every epoch improve training?",
      options: ["It makes training faster", "Prevents the model from learning spurious order-dependent patterns and ensures each mini-batch is a representative sample of the full dataset", "Shuffling is required by NumPy", "It increases the learning rate"],
      correctIndex: 1,
      explanation: "If data is sorted (e.g., all cats then all dogs), early batches teach only 'cat' features. Shuffling ensures each batch has mixed classes, giving stable gradient estimates.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which statement about reproducibility in deep learning is TRUE?",
      options: ["Setting one seed guarantees full reproducibility", "Full reproducibility requires seeding NumPy, the DL framework, Python's random module, AND ensuring deterministic operations (some GPU ops are nondeterministic)", "Reproducibility doesn't matter in practice", "Only the model seed matters"],
      correctIndex: 1,
      explanation: "Multiple RNG sources exist: np.random, torch.manual_seed, random.seed, CUDA ops. Each must be seeded. Some GPU operations (atomicAdd) are inherently nondeterministic.",
      randomize: true
    }
  ]}
/>
