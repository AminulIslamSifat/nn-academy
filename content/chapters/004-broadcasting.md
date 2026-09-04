---
title: "Broadcasting Rules"
slug: "004-broadcasting"
description: "The elegant system NumPy uses to combine arrays of different shapes. Master the three rules and never reshape unnecessarily again."
track: "vectorization"
order: 2
read_time: 16
code_time: 10
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes", "003-vectorization"]
---

# Broadcasting Rules

Broadcasting lets NumPy perform operations on arrays of different shapes without explicit copying. It's how a bias vector of shape `(128,)` gets added to a batch of shape `(32, 128)` seamlessly.

## The Three Rules

NumPy compares shapes element-wise starting from the **trailing dimensions** (rightmost):

1. **Same size** → compatible
2. **One of them is 1** → the 1 is stretched to match the other
3. **Neither is 1 and they differ** → ❌ error

<Callout type="tip" title="Mental model">
Think of broadcasting as "virtually stretching" the smaller array along dimensions of size 1 — no actual memory copy happens.
</Callout>

```python
import numpy as np

# (32, 128) + (128,) → works! (128,) is treated as (1, 128)
batch = np.random.randn(32, 128)
bias = np.random.randn(128)
result = batch + bias  # shape (32, 128)
```

<PyRunner
  cellId="004-cell-1"
  defaultCode={`import numpy as np

# Broadcasting: adding bias to a batch
batch = np.random.randn(4, 3)
bias = np.array([10, 20, 30])  # shape (3,)

print("Batch shape:", batch.shape)
print("Bias shape:", bias.shape)

result = batch + bias
print("Result shape:", result.shape)
print("Result:")
print(result.round(2))
`}
/>

## Dimension Alignment

Shapes are aligned from the right. Missing dimensions are treated as 1:

```
(3, 4, 5) + (4, 5) → (3, 4, 5)   # (4,5) becomes (1, 4, 5)
(3, 1, 5) + (3, 4, 1) → (3, 4, 5) # both stretch
(3, 4) + (3,) → ERROR              # 4 ≠ 3, neither is 1
```

<Visualizer mode="broadcasting" defaultShapeA={[3, 1]} defaultShapeB={[1, 4]} />

<PyRunner
  cellId="004-cell-2"
  defaultCode={`import numpy as np

# Outer product via broadcasting
col = np.array([[1], [2], [3]])  # shape (3, 1)
row = np.array([[10, 20, 30, 40]])  # shape (1, 4)

outer = col * row  # shape (3, 4)
print("col (3,1) × row (1,4):")
print(outer)
print("Result shape:", outer.shape)
`}
/>

## Practical Patterns

### Row-wise normalization
```python
X = np.random.randn(100, 10)
mean = X.mean(axis=0)   # shape (10,)
std = X.std(axis=0)     # shape (10,)
X_norm = (X - mean) / std  # broadcasts over all 100 rows
```

### Column scaling
```python
scales = np.array([[2], [3], [5]])  # shape (3, 1)
M = np.ones((3, 4))
scaled = M * scales  # each row multiplied by its scale
```

<PyRunner
  cellId="004-cell-3"
  defaultCode={`import numpy as np

# Row-wise standardization
np.random.seed(42)
X = np.random.randn(5, 3) * 10 + 50  # mean~50, std~10

mean = X.mean(axis=0)
std = X.std(axis=0)
X_norm = (X - mean) / std

print("Original mean:", X.mean(axis=0).round(2))
print("Original std:", X.std(axis=0).round(2))
print("Normalized mean:", X_norm.mean(axis=0).round(10))
print("Normalized std:", X_norm.std(axis=0).round(2))
`}
/>

<Quiz
  chapterSlug="004-broadcasting"
  questions={[
    {
      id: "q1",
      type: "shape-prediction",
      prompt: "What is the result shape of (5, 1, 3) + (1, 4, 1)?",
      options: ["(5, 4, 3)", "(5, 4, 1)", "Error — incompatible", "(1, 1, 1)"],
      correctIndex: 0,
      explanation: "Aligning right: 3 vs 1 → 3, 1 vs 4 → 4, 5 vs 1 → 5. Result: (5, 4, 3).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Does broadcasting allocate new memory for the stretched array?",
      options: ["No — it's a virtual view", "Yes — it creates a copy", "Only for 3D+ arrays", "Depends on the dtype"],
      correctIndex: 0,
      explanation: "Broadcasting is a virtual operation. NumPy iterates over the smaller array without copying it into the larger shape. Memory efficient!",
      randomize: false,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What happens?",
      code: "import numpy as np\na = np.ones((3, 4))\nb = np.ones((3,))\nprint((a + b).shape)",
      options: ["ValueError: operands could not be broadcast", "(3, 4)", "(3, 3)", "(6,)"],
      correctIndex: 0,
      explanation: "Shapes (3,4) and (3,) align from the right: 4 vs 3 → neither is 1 and they differ → broadcast error.",
      randomize: false,
    }
  ]}
/>
