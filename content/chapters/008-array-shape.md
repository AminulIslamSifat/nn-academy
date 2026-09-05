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
      type: "multiple-choice",
      prompt: "What does the .shape attribute of a NumPy array represent?",
      options: ["Total number of elements", "A tuple indicating the size along each axis/dimension", "Memory size in bytes", "The data type"],
      correctIndex: 1,
      explanation: "shape is a tuple where each element is the length of that axis. (3, 4) means 3 rows and 4 columns. A scalar has shape ().",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.zeros((2, 3, 4))\nprint(a.shape, a.ndim, a.size)",
      options: ["(2, 3, 4) 3 24", "24 3 (2, 3, 4)", "(2, 3, 4) 24 3", "(4, 3, 2) 3 24"],
      correctIndex: 0,
      explanation: "shape=(2,3,4), ndim=3 (three axes), size=2*3*4=24 (total elements). These three properties describe any array completely.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the difference between shape (5,) and shape (5, 1)?",
      options: ["No practical difference", "(5,) is 1D with 5 elements; (5,1) is 2D with 5 rows and 1 column — they broadcast differently with other arrays", "One is a view, one is a copy", "Different dtypes"],
      correctIndex: 1,
      explanation: "This distinction matters enormously in neural networks. (5,) + (5,) gives element-wise (5,). But (5,1) + (5,) broadcasts to (5,5) outer addition. Getting this wrong causes silent shape bugs.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array(42)\nprint(a.shape, a.ndim)",
      options: ["() 0", "(1,) 1", "(42,) 1", "Error"],
      correctIndex: 0,
      explanation: "A 0-dimensional array (scalar) has empty shape tuple () and ndim=0. It holds a single value but has no axes.",
      randomize: false,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does np.newaxis do?",
      options: ["Creates a new independent array", "Inserts a new axis of size 1 at the specified position, increasing ndim by 1", "Adds an element to the array", "Removes an axis"],
      correctIndex: 1,
      explanation: "a[np.newaxis, :] converts (N,) to (1,N). a[:, np.newaxis] converts (N,) to (N,1). This is essential for controlling broadcasting behavior.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nprint(a[np.newaxis, :].shape)",
      options: ["(1, 3)", "(3, 1)", "(3,)", "(1, 1, 3)"],
      correctIndex: 0,
      explanation: "np.newaxis at position 0 inserts a leading dimension. (3,) becomes (1, 3) — a row vector.",
      randomize: false,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What does squeeze() do?",
      options: ["Flattens the array to 1D", "Removes ALL axes of size 1 from the shape", "Adds dimensions", "Transposes the array"],
      correctIndex: 1,
      explanation: "squeeze converts (1, 3, 1, 5) to (3, 5) by removing both size-1 axes. It does NOT flatten — non-singleton dimensions are preserved.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.zeros((1, 3, 1, 5))\nprint(a.squeeze().shape)",
      options: ["(3, 5)", "(1, 3, 1, 5)", "(3, 1, 5)", "(15,)"],
      correctIndex: 0,
      explanation: "squeeze removes all size-1 axes: (1,3,1,5) -> (3,5). Both leading and middle singleton dims are removed.",
      randomize: false,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why does shape matter so much in neural network code?",
      options: ["It's just for printing", "Every operation (matmul, broadcasting, loss computation) depends on correct shapes; shape mismatches are the #1 source of bugs in deep learning", "Shape determines dtype", "GPUs require specific shapes"],
      correctIndex: 1,
      explanation: "Wrong shapes cause either crashes (ValueError) or worse — silent wrong results. Always print shapes during development. Shape discipline is the most important debugging skill in NumPy-based ML.",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(24).reshape(2, 3, 4)\nprint(a.shape[1])",
      options: ["3", "2", "4", "24"],
      correctIndex: 0,
      explanation: "shape is a tuple. shape[1] accesses the second dimension: 3.",
      randomize: false,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What does expand_dims(a, axis=1) do?",
      options: ["Same as a[:, np.newaxis] — inserts a size-1 axis at position 1", "Removes axis 1", "Duplicates axis 1", "Flattens axis 1"],
      correctIndex: 0,
      explanation: "expand_dims is the readable alternative to np.newaxis indexing. expand_dims(a, 1) on shape (3,) gives (3,1). More explicit about intent.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.ones((2, 1, 3))\nprint(a.squeeze(axis=1).shape)",
      options: ["(2, 3)", "(2, 1, 3)", "(1, 3)", "(2,)"],
      correctIndex: 0,
      explanation: "squeeze(axis=1) only removes axis 1 if it's size 1. (2,1,3) -> (2,3). Unlike squeeze() with no args, this targets a specific axis.",
      randomize: false,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What are strides?",
      options: ["Step sizes for loops", "The number of bytes to step in memory to move one position along each axis", "Array dimensions", "Memory addresses"],
      correctIndex: 1,
      explanation: "Strides define how shape maps to flat memory. For a (3,4) float64 array, strides are (32, 8): skip 32 bytes to next row, 8 bytes to next column.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\narr = np.arange(6, dtype=np.float64).reshape(2, 3)\nprint(arr.strides)",
      options: ["(24, 8)", "(8, 24)", "(16, 8)", "(8, 16)"],
      correctIndex: 0,
      explanation: "Row-major (C-order): stride[0] = 3 elements * 8 bytes = 24 (skip a full row). stride[1] = 1 element * 8 bytes = 8 (next column).",
      randomize: false,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Why does transpose return a view instead of a copy?",
      options: ["Performance optimization only", "Transpose just swaps the stride values without moving any data in memory — zero cost", "It actually does copy", "Only for square matrices"],
      correctIndex: 1,
      explanation: "Transposing (3,4) to (4,3) just swaps strides from (32,8) to (8,32). The data stays in place. This is why .T is free regardless of array size.",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.zeros((4, 1, 3, 1))\nprint(a.ndim, a.size)",
      options: ["4 12", "4 4", "2 12", "3 12"],
      correctIndex: 0,
      explanation: "ndim=4 (four axes). size = 4*1*3*1 = 12 total elements. Size-1 axes count toward ndim but don't add elements.",
      randomize: false,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "In batched neural network code, what does shape (B, C, H, W) typically represent?",
      options: ["Random variable names", "Batch x Channels x Height x Width — the standard image tensor format in PyTorch", "Bytes per element", "Buffer configuration"],
      correctIndex: 1,
      explanation: "NCHW is the standard convention in PyTorch and many CNN implementations. B=batch size, C=color/feature channels, H=height, W=width.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nb = a[:, np.newaxis, :]\nprint(b.shape)",
      options: ["(2, 1, 3)", "(2, 3, 1)", "(1, 2, 3)", "(2, 3)"],
      correctIndex: 0,
      explanation: "np.newaxis at position 1 inserts a new axis between the existing two. (2,3) -> (2,1,3). Useful for adding a channel dimension.",
      randomize: false,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What happens when you index a 3D array with a single integer: a[0]?",
      options: ["Returns a scalar", "Returns a 2D sub-array — the first 'slice' along axis 0, reducing ndim by 1", "Error", "Returns a 1D array"],
      correctIndex: 1,
      explanation: "a[0] selects along axis 0, reducing dimensionality by 1. Shape (2,3,4)[0] -> (3,4). Each integer index removes one axis.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Why can two arrays with the same shape have different strides?",
      options: ["They can't", "One might be a transposed view, a slice with step, or Fortran-ordered — all share the same shape but traverse memory differently", "Different dtypes", "Different sizes"],
      correctIndex: 1,
      explanation: "A (3,4) C-order array has strides (32,8). Its transpose .T also has shape... wait, .T has shape (4,3). But a[::2, ::2] could have shape (2,2) with doubled strides. Same shape, different memory access patterns affect cache performance.",
      randomize: true,
    }
  ]}
/>
