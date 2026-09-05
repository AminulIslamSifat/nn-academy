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
      type: "multiple-choice",
      prompt: "What fundamental property distinguishes a NumPy ndarray from a Python list?",
      options: ["ndarrays can hold mixed types", "ndarrays store data in contiguous memory with a fixed dtype, enabling vectorized operations", "ndarrays are always faster", "ndarrays support negative indexing"],
      correctIndex: 1,
      explanation: "Python lists store pointers to heterogeneous objects. ndarrays store raw typed values contiguously, which enables SIMD vectorization and cache-friendly access patterns.",
      randomize: true,
    },
    {
      id: "q2",
      type: "shape-prediction",
      prompt: "What is the shape of np.arange(24).reshape(2, 3, 4)?",
      options: ["(2, 3, 4)", "(24,)", "(4, 3, 2)", "(2, 12)"],
      correctIndex: 0,
      explanation: "reshape(2, 3, 4) creates a 3D tensor with 2 blocks of 3 rows and 4 columns. Total elements: 2×3×4 = 24.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does np.zeros((2, 3)).size return?",
      code: "import numpy as np\nx = np.zeros((2, 3))\nprint(x.size)",
      options: ["6", "2", "3", "(2, 3)"],
      correctIndex: 0,
      explanation: "size returns the total number of elements: 2 × 3 = 6.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which creates a 4×4 identity matrix?",
      options: ["np.eye(4)", "np.ones((4,4))", "np.diag(4)", "np.identity_matrix(4)"],
      correctIndex: 0,
      explanation: "np.eye(n) creates an n×n identity matrix. np.ones creates all-ones, np.diag with a scalar creates a 0-d array.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is printed?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nprint(a.ndim, a.shape, a.dtype)",
      options: ["1 (3,) int64", "3 (1,) float64", "1 (3,) int32", "0 (3,) int64"],
      correctIndex: 0,
      explanation: "A 1D array of integers has ndim=1, shape=(3,), and default integer dtype (int64 on most platforms).",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "If an array has shape (2, 3, 4) and dtype float64, how many bytes does it occupy?",
      options: ["192", "24", "96", "48"],
      correctIndex: 0,
      explanation: "Total elements = 2×3×4 = 24. float64 = 8 bytes each. 24 × 8 = 192 bytes.",
      randomize: true,
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "What does this print?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nprint(a[1, 2])",
      options: ["6", "5", "7", "10"],
      correctIndex: 0,
      explanation: "Row 1 is [4,5,6,7]. Index 2 of that row is 6.",
      randomize: false,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does np.linspace(0, 1, 5) produce?",
      options: ["[0.0, 0.25, 0.5, 0.75, 1.0]", "[0, 1, 2, 3, 4]", "[0.0, 0.2, 0.4, 0.6, 0.8]", "[0.0, 0.5, 1.0]"],
      correctIndex: 0,
      explanation: "linspace generates 5 evenly spaced values from 0 to 1 inclusive: step = (1-0)/(5-1) = 0.25.",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.full((2, 3), 7)\nprint(a.shape, a[0, 0])",
      options: ["(2, 3) 7", "(3, 2) 7", "(2, 3) 0", "(6,) 7"],
      correctIndex: 0,
      explanation: "np.full(shape, fill_value) creates an array of given shape filled with the specified value.",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Why does reshape sometimes fail with a ValueError?",
      options: ["The array is too large", "The new shape must have the same total number of elements as the original", "reshape only works on 2D arrays", "The dtype doesn't match"],
      correctIndex: 1,
      explanation: "reshape requires prod(new_shape) == prod(old_shape). You can't reshape 12 elements into (3, 5) because 3×5=15≠12.",
      randomize: true,
    },
    {
      id: "q11",
      type: "code-output",
      prompt: "What does this print?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nprint(a.T.shape)",
      options: ["(3, 2)", "(2, 3)", "(6,)", "(1, 6)"],
      correctIndex: 0,
      explanation: "Transpose swaps axes: (2, 3).T → (3, 2). Rows become columns.",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What is the difference between np.array() and np.asarray()?",
      options: ["No difference", "np.array always copies; np.asarray avoids copying if input is already an ndarray with matching dtype", "np.asarray is slower", "np.array only works with lists"],
      correctIndex: 1,
      explanation: "np.asarray returns the input unchanged if it's already a compatible ndarray, avoiding unnecessary memory allocation.",
      randomize: true,
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.array([[1, 2], [3, 4]])\nprint(a.flatten().shape)",
      options: ["(4,)", "(2, 2)", "(1, 4)", "(2,)"],
      correctIndex: 0,
      explanation: "flatten() returns a 1D copy of the array. 2×2 = 4 elements → shape (4,).",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What does axis=0 mean in a 2D array operation like np.sum(a, axis=0)?",
      options: ["Sum across columns (result has one value per column)", "Sum across rows (result has one value per row)", "Sum everything", "Sum diagonally"],
      correctIndex: 0,
      explanation: "axis=0 collapses the first dimension (rows), summing DOWN each column. Result shape: (num_columns,).",
      randomize: true,
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.random.randn(3, 4)\nprint(a.mean(axis=1).shape)",
      options: ["(3,)", "(4,)", "(3, 4)", "()"],
      correctIndex: 0,
      explanation: "axis=1 collapses columns, computing mean across each row. 3 rows → shape (3,).",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why is np.arange preferred over Python range for creating numerical sequences?",
      options: ["It supports floats", "It returns an ndarray enabling vectorized math, while range returns a lazy iterator of Python ints", "Both are equivalent", "range is deprecated"],
      correctIndex: 1,
      explanation: "np.arange produces a contiguous array ready for vectorized operations. range produces Python int objects requiring conversion.",
      randomize: true,
    },
    {
      id: "q17",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.zeros((3,))\nb = np.ones((3,))\nprint((a + b).dtype)",
      options: ["float64", "int64", "bool", "object"],
      correctIndex: 0,
      explanation: "zeros and ones default to float64. Addition preserves the dtype: float64.",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "What does keepdims=True do in np.sum(a, axis=0, keepdims=True)?",
      options: ["Nothing special", "Preserves the reduced axis as size 1 instead of removing it, maintaining dimensionality for broadcasting", "Keeps the original array unchanged", "Returns a scalar"],
      correctIndex: 1,
      explanation: "Without keepdims, sum(axis=0) on (3,4) gives (4,). With keepdims=True, it gives (1,4), preserving the 2D structure.",
      randomize: true,
    },
    {
      id: "q19",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1.5, 2.7, 3.2])\nprint(a.astype(int))",
      options: ["[1 2 3]", "[2 3 3]", "[1.5 2.7 3.2]", "Error"],
      correctIndex: 0,
      explanation: "astype(int) truncates toward zero (not rounds). 1.5→1, 2.7→2, 3.2→3.",
      randomize: false,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In deep learning, why do we typically use float32 instead of float64?",
      options: ["float64 is less accurate", "float32 uses half the memory and is faster on GPUs, with sufficient precision for gradient-based optimization", "float64 causes overflow", "NumPy doesn't support float64"],
      correctIndex: 1,
      explanation: "float32 provides ~7 decimal digits of precision, adequate for training. Half the memory means larger batches and faster computation on GPU hardware optimized for FP32.",
      randomize: true,
    }
  ]}
/>
