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
      type: "multiple-choice",
      prompt: "What is broadcasting in NumPy?",
      options: ["Sending arrays over the network", "A set of rules that allows arithmetic between arrays of different shapes by virtually stretching dimensions of size 1", "Automatically reshaping arrays to match", "Converting dtypes during operations"],
      correctIndex: 1,
      explanation: "Broadcasting lets you combine arrays of different shapes without explicit copying or reshaping. Dimensions of size 1 are 'stretched' to match the other array's size along that axis.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How does NumPy align shapes when checking broadcast compatibility?",
      options: ["From the left (leading dimensions)", "From the right (trailing dimensions); missing leading dimensions are treated as 1", "By total number of elements", "Alphabetically by axis name"],
      correctIndex: 1,
      explanation: "NumPy compares shapes starting from the last dimension and works leftward. A shape (4,) is treated as (1, 4) when paired with (3, 4). This is why (3,4)+(4,) works but (3,4)+(3,) fails.",
      randomize: true,
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "What is the result shape of (5, 1, 3) + (1, 4, 1)?",
      options: ["(5, 4, 3)", "(5, 4, 1)", "Error — incompatible", "(1, 1, 1)"],
      correctIndex: 0,
      explanation: "Align from right: dim2: 3 vs 1 → 3, dim1: 1 vs 4 → 4, dim0: 5 vs 1 → 5. All compatible. Result: (5, 4, 3).",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Does broadcasting allocate new memory for the stretched array?",
      options: ["No — it's a virtual operation that reuses the smaller array's data", "Yes — it creates a full-size copy", "Only for 3D+ arrays", "Depends on the dtype"],
      correctIndex: 0,
      explanation: "Broadcasting is purely virtual. NumPy iterates over the smaller array repeatedly without allocating a larger copy. This makes it extremely memory-efficient.",
      randomize: false,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What happens?",
      code: "import numpy as np\na = np.ones((3, 4))\nb = np.ones((3,))\nprint((a + b).shape)",
      options: ["ValueError: operands could not be broadcast", "(3, 4)", "(3, 3)", "(6,)"],
      correctIndex: 0,
      explanation: "Shapes (3,4) and (3,) align from the right: trailing dims are 4 vs 3. Neither is 1 and they differ → ValueError. The (3,) doesn't match the last dimension of (3,4).",
      randomize: false,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nb = np.array([10, 20, 30, 40])\nprint((a + b)[0])",
      options: ["[10 21 32 43]", "[10 20 30 40]", "[0 1 2 3]", "Error"],
      correctIndex: 0,
      explanation: "b has shape (4,), which broadcasts with (3,4). Row 0 of a is [0,1,2,3]. Adding b: [0+10, 1+20, 2+30, 3+40] = [10, 21, 32, 43].",
      randomize: false,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "In neural networks, why does adding bias b of shape (128,) to output of shape (32, 128) work?",
      options: ["NumPy automatically reshapes b", "Broadcasting treats (128,) as (1, 128), matching the last dimension of (32, 128) and applying the same bias to every sample in the batch", "It only works when batch_size=1", "The bias gets tiled internally"],
      correctIndex: 1,
      explanation: "This is the most common broadcasting pattern in deep learning. The bias vector aligns with the feature dimension and broadcasts across the batch dimension. No copy, no reshape needed.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the result shape?",
      code: "import numpy as np\ncol = np.array([[1], [2], [3]])\nrow = np.array([[10, 20]])\nprint((col + row).shape)",
      options: ["(3, 2)", "(3, 1)", "(1, 2)", "Error"],
      correctIndex: 0,
      explanation: "(3,1) + (1,2): trailing dims 1 vs 2 → 2, leading dims 3 vs 1 → 3. Result: (3, 2). This is the outer-addition pattern using broadcasting.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Which pair of shapes is NOT broadcastable?",
      options: ["(3, 1) and (1, 4)", "(5,) and (5,)", "(3, 4) and (3,)", "(2, 3, 4) and (3, 4)"],
      correctIndex: 2,
      explanation: "(3,4) and (3,) align from the right: 4 vs 3. Neither is 1 and they differ → incompatible. The others all work: (3,1)+(1,4)→(3,4), (5,)+(5,)→(5,), (2,3,4)+(3,4)→(2,3,4).",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.ones((2, 1, 3))\nb = np.ones((1, 4, 1))\nprint((a * b).shape)",
      options: ["(2, 4, 3)", "(2, 1, 3)", "(1, 4, 1)", "Error"],
      correctIndex: 0,
      explanation: "Right-aligned comparison: 3 vs 1→3, 1 vs 4→4, 2 vs 1→2. All compatible. Result: (2, 4, 3).",
      randomize: false,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why is broadcasting essential for efficient neural network code?",
      options: ["It makes code shorter", "It enables applying biases, normalization statistics, masks, and other per-feature or per-sample operations to entire batches without explicit loops, reshaping, or tiling", "It prevents shape errors", "It's required by NumPy"],
      correctIndex: 1,
      explanation: "Without broadcasting, you'd need to manually reshape or tile every bias/mask/stat to match batch dimensions. Broadcasting handles this implicitly with zero memory overhead.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nX = np.random.randn(4, 3)\nmean = X.mean(axis=0)\nprint((X - mean).shape)",
      options: ["(4, 3)", "(4,)", "(3,)", "()"],
      correctIndex: 0,
      explanation: "mean(axis=0) produces shape (3,). X has shape (4,3). Broadcasting subtracts the column means from every row. Result keeps shape (4,3).",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What does np.newaxis do?",
      options: ["Creates a brand new array with copied data", "Inserts a dimension of size 1 at the specified position, changing the shape for broadcasting purposes", "Adds an element to the array", "Reshapes to 1D"],
      correctIndex: 1,
      explanation: "a[:, np.newaxis] converts shape (N,) to (N,1). This is crucial for making arrays broadcastable in specific ways, like outer products or row-wise operations.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nb = a[:, np.newaxis]\nprint(b.shape)",
      options: ["(3, 1)", "(1, 3)", "(3,)", "(3, 3)"],
      correctIndex: 0,
      explanation: "np.newaxis at position 1 inserts a new axis after the first. Shape (3,) becomes (3, 1). This column vector can now broadcast with row vectors.",
      randomize: false,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Can a scalar (0-d value) broadcast with any array?",
      options: ["No, scalars must be wrapped in an array", "Yes — scalars have shape () which is compatible with every shape via broadcasting", "Only with 1D arrays", "Only with square matrices"],
      correctIndex: 1,
      explanation: "Scalars have zero dimensions, so there are no dimension conflicts. a + 5 adds 5 to every element regardless of a's shape.",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nb = np.array([[10], [20]])\nprint((a + b).shape)",
      options: ["(2, 3)", "(2, 1)", "(1, 3)", "Error"],
      correctIndex: 0,
      explanation: "(2,3) + (2,1): trailing dims 3 vs 1→3, leading dims 2 vs 2→2. Compatible. Result: (2,3). Column vector broadcasts across columns.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is the 'outer product' pattern using broadcasting?",
      options: ["Using np.outer(a, b)", "Reshape a to (N,1) and b to (1,M), then multiply: result is (N,M) where element (i,j) = a[i]*b[j]", "Using a @ b", "Using a * b directly"],
      correctIndex: 1,
      explanation: "Column (N,1) × Row (1,M) → (N,M) matrix. Each element is the product of corresponding entries. Pure broadcasting, no special function needed.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\nX = np.random.randn(100, 10)\nstd = X.std(axis=0, keepdims=True)\nprint(std.shape)",
      options: ["(1, 10)", "(10,)", "(100, 10)", "(100, 1)"],
      correctIndex: 0,
      explanation: "keepdims=True preserves the reduced axis as size 1 instead of removing it. std(axis=0) on (100,10) gives (1,10) instead of (10,), enabling direct broadcasting with (100,10).",
      randomize: false,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Why does (3,4) + (4,) work but (3,4) + (3,) fail?",
      options: ["Arbitrary design choice", "Alignment is from the right: (4,) aligns with last dim 4 → match. (3,) aligns with last dim 4 → 3≠4, neither is 1 → fail", "Left-to-right alignment", "NumPy checks total element count"],
      correctIndex: 1,
      explanation: "This is THE key broadcasting rule. Shapes align from trailing dimensions. Understanding this prevents the vast majority of shape-related bugs in NumPy code.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In row-wise normalization (X - mean) / std, why must mean and std have shape (D,) or (1, D)?",
      options: ["Convention only", "Shape (D,) aligns with the last dimension of (N, D) via broadcasting, applying the same per-feature statistic to every row. Shape (N,) would try to align with columns and fail", "They must be scalars", "Shape doesn't matter for division"],
      correctIndex: 1,
      explanation: "Mean/std computed along axis=0 have shape (D,). This broadcasts correctly with (N,D) by applying each feature's statistic to all N rows. If you accidentally got shape (N,), it would try to match against the column dimension and either fail or produce wrong results.",
      randomize: true,
    }
  ]}
/>
