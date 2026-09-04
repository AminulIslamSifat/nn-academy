---
title: "Arrays & Shapes"
slug: "001-arrays-and-shapes"
description: "The fundamental building block of NumPy — the ndarray. Learn how data lives in contiguous memory and why shape matters."
track: "numpy-foundations"
order: 1
read_time: 12
code_time: 8
execution_timeout: 5
prerequisites: []
---

# Arrays & Shapes

Everything in deep learning is a tensor. In NumPy, that tensor is the `ndarray` — a contiguous block of typed memory with a *shape* that tells you how to interpret it.

## Why Not Python Lists?

Python lists store pointers to objects. NumPy arrays store raw values in contiguous memory. The difference is **orders of magnitude** in speed for numerical work.

```python
import numpy as np

# A simple 1D array
a = np.array([1.0, 2.0, 3.0, 4.0])
print(a)        # [1. 2. 3. 4.]
print(a.shape)  # (4,)
print(a.dtype)  # float64
```

<PyRunner
  cellId="001-cell-1"
  defaultCode={`import numpy as np

a = np.array([1.0, 2.0, 3.0, 4.0])
print("Array:", a)
print("Shape:", a.shape)
print("Dtype:", a.dtype)
print("Memory (bytes):", a.nbytes)
`}
/>

## Shape: The Lens on Memory

The same 12 numbers can be viewed as a vector, a row, a column, or a matrix. Shape doesn't move data — it changes how you *index* into it.

```python
data = np.arange(12)  # 0 through 11

as_row = data.reshape(1, 12)    # shape (1, 12)
as_col = data.reshape(12, 1)    # shape (12, 1)
as_mat = data.reshape(3, 4)     # shape (3, 4)
```

<PyRunner
  cellId="001-cell-2"
  defaultCode={`import numpy as np

data = np.arange(12)
print("Flat:", data.shape)

as_mat = data.reshape(3, 4)
print("Matrix (3×4):")
print(as_mat)

as_3d = data.reshape(2, 2, 3)
print("\n3D tensor (2×2×3):")
print(as_3d)
`}
/>

## Creating Arrays

<Callout type="tip" title="Common constructors">
Use `np.zeros`, `np.ones`, `np.full`, `np.arange`, `np.linspace`, and `np.random.randn` to create arrays without Python loops.
</Callout>

```python
zeros = np.zeros((3, 4))       # 3×4 matrix of 0s
ones = np.ones((2, 2, 2))      # 2×2×2 tensor of 1s
randn = np.random.randn(5, 3)  # 5×3 from standard normal
```

<PyRunner
  cellId="001-cell-3"
  defaultCode={`import numpy as np

zeros = np.zeros((3, 4))
randn = np.random.randn(3, 3)

print("Zeros:")
print(zeros)
print("\nRandom normal:")
print(randn)
print("\nMean ≈ 0?", randn.mean().round(2))
`}
/>

## The ndim, size, nbytes Trinity

Every array has three properties you'll check constantly:

| Property | Meaning |
|----------|---------|
| `ndim` | Number of axes (dimensions) |
| `shape` | Tuple of axis lengths |
| `size` | Total elements (`prod(shape)`) |
| `nbytes` | Memory footprint (`size × itemsize`) |

<Quiz
  chapterSlug="001-arrays-and-shapes"
  questions={[
    {
      id: "q1",
      type: "shape-prediction",
      prompt: "What is the shape of np.arange(24).reshape(2, 3, 4)?",
      options: ["(2, 3, 4)", "(24,)", "(4, 3, 2)", "(2, 12)"],
      correctIndex: 0,
      explanation: "reshape(2, 3, 4) creates a 3D tensor with 2 blocks of 3 rows and 4 columns. Total elements: 2×3×4 = 24. ✓",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does np.zeros((2, 3)).size return?",
      code: "import numpy as np\nx = np.zeros((2, 3))\nprint(x.size)",
      options: ["6", "2", "3", "(2, 3)"],
      correctIndex: 0,
      explanation: "size returns the total number of elements: 2 × 3 = 6.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Which creates a 4×4 identity matrix?",
      options: ["np.eye(4)", "np.ones((4,4))", "np.diag(4)", "np.identity_matrix(4)"],
      correctIndex: 0,
      explanation: "np.eye(n) creates an n×n identity matrix. np.ones creates all-ones, np.diag with a scalar creates a diagonal matrix with that value.",
      randomize: true,
    }
  ]}
/>
