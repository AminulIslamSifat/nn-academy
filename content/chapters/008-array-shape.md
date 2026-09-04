---
title: "Array Shape & Dimensions"
slug: "008-array-shape"
description: "Deep dive into .shape, .ndim, .size, and how NumPy represents multi-dimensional data in flat memory."
track: "numpy-foundations"
order: 5
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes"]
---

# Array Shape & Dimensions

Shape is the single most important property of a NumPy array. It tells you the size along each axis.

## Shape, ndim, size

<PyRunner
  cellId="008-cell-1"
  defaultCode={`import numpy as np

scalar = np.array(42)
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4], [5, 6]])
tensor = np.zeros((2, 3, 4))

for name, arr in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("tensor", tensor)]:
    print(f"{name:8s} shape={str(arr.shape):12s} ndim={arr.ndim}  size={arr.size}")
`}
/>

## Row Vectors vs Column Vectors

In neural networks, the distinction between `(n,)` and `(n, 1)` matters:

<PyRunner
  cellId="008-cell-2"
  defaultCode={`import numpy as np

row = np.array([1, 2, 3])       # shape (3,)
col = np.array([[1], [2], [3]]) # shape (3, 1)

print(f"Row vector: shape={row.shape}, ndim={row.ndim}")
print(f"Col vector: shape={col.shape}, ndim={col.ndim}")

# They broadcast differently!
bias = np.array([10, 20, 30])
print(f"
row + bias = {row + bias}")         # (3,) + (3,) → (3,)
print(f"col + bias = {(col + bias).shape}")    # (3,1) + (3,) → (3,3) via broadcasting!
`}
/>

<Callout type="warning" title="Shape Gotcha">
A 1D array of shape `(n,)` is neither a row nor column vector — it's just a sequence. Matrix multiplication treats it contextually.
</Callout>

## Adding and Removing Dimensions

<PyRunner
  cellId="008-cell-3"
  defaultCode={`import numpy as np

a = np.array([1, 2, 3])  # (3,)

# Add dimension with np.newaxis (alias for None)
row = a[np.newaxis, :]   # (1, 3) — row vector
col = a[:, np.newaxis]   # (3, 1) — column vector

print(f"Original: {a.shape}")
print(f"Row:      {row.shape}")
print(f"Col:      {col.shape}")

# expand_dims is more readable
print(f"
expand_dims(a, 0): {np.expand_dims(a, 0).shape}")
print(f"expand_dims(a, 1): {np.expand_dims(a, 1).shape}")

# Squeeze removes size-1 dimensions
padded = np.zeros((1, 3, 1, 5))
print(f"
Padded: {padded.shape}")
print(f"Squeezed: {padded.squeeze().shape}")
`}
/>

## Strides: How Shape Maps to Memory

Strides tell NumPy how many bytes to step in memory to move along each axis:

<PyRunner
  cellId="008-cell-4"
  defaultCode={`import numpy as np

a = np.arange(12, dtype=np.int32).reshape(3, 4)
print(f"Shape:   {a.shape}")
print(f"Strides: {a.strides}")
print(f"Itemsize: {a.itemsize} bytes")
print(f"
Memory layout (row-major / C-order):")
print(a)
print(f"
To go from a[i,j] to a[i,j+1]: step {a.strides[1]} bytes (1 element)")
print(f"To go from a[i,j] to a[i+1,j]: step {a.strides[0]} bytes (4 elements)")
`}
/>

<Quiz
  chapterSlug="008-array-shape"
  questions={[
    {
      id: "q1",
      type: "shape-prediction",
      prompt: "What is the shape of np.zeros((2, 3, 4)).squeeze()?",
      options: ["(2, 3, 4)", "(6, 4)", "(2, 12)", "Error"],
      correctIndex: 0,
      explanation: "squeeze() only removes dimensions of size 1. Since no dimension is 1, shape stays (2, 3, 4).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a[np.newaxis, :] do to a 1D array of shape (5,)?",
      options: ["Adds a leading dimension → (1, 5)", "Adds a trailing dimension → (5, 1)", "Does nothing", "Flattens the array"],
      correctIndex: 0,
      explanation: "np.newaxis inserts a new axis at that position. [newaxis, :] adds dim at axis 0.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is arr.strides for arr = np.arange(6, dtype=np.float64).reshape(2, 3)?",
      code: "import numpy as np\narr = np.arange(6, dtype=np.float64).reshape(2, 3)\nprint(arr.strides)",
      options: ["(24, 8)", "(8, 24)", "(16, 8)", "(8, 16)"],
      correctIndex: 0,
      explanation: "Row-major: stride[0] = 3 elements × 8 bytes = 24, stride[1] = 1 element × 8 bytes = 8.",
      randomize: true,
    }
  ]}
/>
