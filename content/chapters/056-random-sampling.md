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
      prompt: "Why must you shuffle X and y using the SAME permutation indices?",
      options: ["For speed", "Shuffling independently breaks the correspondence between inputs and labels, making training impossible", "NumPy requires it", "It doesn't matter"],
      correctIndex: 1,
      explanation: "Each X[i] corresponds to y[i]. If you shuffle them independently, X[0] might get paired with y[7]'s label. The model would learn wrong input-output mappings.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why use np.random.default_rng() instead of np.random.seed()?",
      options: ["It's slower but safer", "The new Generator API has better statistical properties, is faster, and avoids global state issues that cause bugs in multi-threaded code", "They're identical", "The old API is deprecated and removed"],
      correctIndex: 1,
      explanation: "The legacy np.random uses global state shared across all code. Generator instances are independent, thread-safe, and use the superior PCG64 algorithm.",
      randomize: false,
    }
  ]}
/>
