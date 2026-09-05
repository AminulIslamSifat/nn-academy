---
title: "Reshape & Transpose"
slug: "009-array-reshape"
description: "Change array shapes without copying data. Master reshape, ravel, flatten, transpose, and the -1 trick."
track: "numpy-foundations"
order: 6
read_time: 12
code_time: 10
execution_timeout: 5
prerequisites: ["008-array-shape"]
---

# Reshape & Transpose

Reshaping reinterprets the same memory with a different shape — no data movement, instant operation.

## Basic Reshape

<PyRunner
  cellId="009-cell-1"
  defaultCode={`import numpy as np

a = np.arange(12)
print(f"Original: shape={a.shape}, values={a}")

b = a.reshape(3, 4)
print(f"Reshaped (3,4):
{b}")

c = a.reshape(2, 2, 3)
print(f"Reshaped (2,2,3): shape={c.shape}")

# Total elements must match!
try:
    a.reshape(5, 3)  # 15 ≠ 12
except ValueError as e:
    print(f"
Error: {e}")
`}
/>

## The -1 Trick

Use `-1` to let NumPy infer one dimension:

<PyRunner
  cellId="009-cell-2"
  defaultCode={`import numpy as np

a = np.arange(24)

# Infer the last dimension
b = a.reshape(4, -1)    # 24 / 4 = 6 → (4, 6)
print(f"reshape(4, -1):  {b.shape}")

c = a.reshape(-1, 8)    # 24 / 8 = 3 → (3, 8)
print(f"reshape(-1, 8):  {c.shape}")

d = a.reshape(2, 3, -1) # 24 / (2*3) = 4 → (2, 3, 4)
print(f"reshape(2,3,-1): {d.shape}")

# Common pattern: flatten batch for processing, then restore
batch = np.random.randn(32, 28, 28)  # 32 images of 28×28
flat = batch.reshape(32, -1)          # (32, 784)
restored = flat.reshape(32, 28, 28)   # back to images
print(f"
Batch: {batch.shape} → Flat: {flat.shape} → Restored: {restored.shape}")
`}
/>

## Reshape Returns a View (Usually)

<PyRunner
  cellId="009-cell-3"
  defaultCode={`import numpy as np

a = np.arange(12).reshape(3, 4)
b = a.reshape(4, 3)

print(f"Shares memory: {np.shares_memory(a, b)}")
print(f"b.base is a: {b.base is a}")

# Modify b → a changes
b[0, 0] = 999
print(f"
a[0,0] after b[0,0]=999: {a[0,0]}")
`}
/>

## Transpose

Transpose swaps axes — also a view, no data copied:

<PyRunner
  cellId="009-cell-4"
  defaultCode={`import numpy as np

a = np.arange(6).reshape(2, 3)
print(f"Original:
{a}  shape={a.shape}")

print(f"
.T (transpose):
{a.T}  shape={a.T.shape}")

# For higher dimensions, specify axis order
t = np.zeros((2, 3, 4))
print(f"
3D tensor: {t.shape}")
print(f"transpose(1,0,2): {t.transpose(1,0,2).shape}")
print(f"transpose(2,1,0): {t.transpose(2,1,0).shape}")

# swapaxes for swapping two specific axes
print(f"swapaxes(0,2): {t.swapaxes(0,2).shape}")
`}
/>

## ravel vs flatten

<PyRunner
  cellId="009-cell-5"
  defaultCode={`import numpy as np

a = np.arange(6).reshape(2, 3)

r = a.ravel()    # view when possible
f = a.flatten()  # always a copy

print(f"ravel:   {r}, shares_memory={np.shares_memory(a, r)}")
print(f"flatten: {f}, shares_memory={np.shares_memory(a, f)}")

# Order matters: 'C' (row-major) vs 'F' (column-major)
print(f"
ravel('C'): {a.ravel('C')}")
print(f"ravel('F'): {a.ravel('F')}")
`}
/>

<Quiz
  chapterSlug="009-array-reshape"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does reshape do?",
      options: ["Changes data values", "Returns an array with different dimensions but the same underlying data", "Sorts the array", "Changes dtype"],
      correctIndex: 1,
      explanation: "reshape reinterprets the same flat memory under a new shape. No data moves (when possible). Total elements must remain the same.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(12)\nb = a.reshape(3, 4)\nprint(b[1, 2])",
      options: ["6", "5", "7", "10"],
      correctIndex: 0,
      explanation: "Row-major order fills row by row: [0,1,2,3], [4,5,6,7], [8,9,10,11]. Row 1, col 2 = 6.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does -1 mean in reshape?",
      options: ["Reverse the array", "Infer this dimension automatically from the total size and other specified dimensions", "Remove this dimension", "Error value"],
      correctIndex: 1,
      explanation: "reshape(2, -1) on 12 elements -> (2, 6). NumPy calculates -1 = 12/2 = 6. Only one dimension can be -1.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(8).reshape(2, -1)\nprint(a.shape)",
      options: ["(2, 4)", "(4, 2)", "(2, 2)", "(8,)"],
      correctIndex: 0,
      explanation: "8 elements / 2 rows = 4 columns. -1 is inferred as 4. Shape: (2, 4).",
      randomize: false,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Does reshape always return a view?",
      options: ["Yes, always", "Usually yes, but returns a copy if the data isn't contiguous or the new shape is incompatible with the stride pattern", "Never", "Only for 2D arrays"],
      correctIndex: 1,
      explanation: "reshape returns a view when the memory layout supports it. After transpose or non-contiguous slicing, the data may need rearranging, forcing a copy.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nb = a.reshape(6)\nb[0] = 99\nprint(a[0, 0])",
      options: ["99", "0", "Error", "None"],
      correctIndex: 0,
      explanation: "reshape returned a view. b and a share memory. Modifying b[0] modifies a[0,0].",
      randomize: false,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is the difference between reshape and resize?",
      options: ["No difference", "reshape returns a new array (view or copy) with same total elements; resize modifies in-place and CAN change total element count", "reshape is faster", "resize returns a view"],
      correctIndex: 1,
      explanation: "resize can pad with zeros or truncate to change total size. reshape requires prod(new_shape) == size, otherwise ValueError.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nprint(a.reshape(4, 3)[0])",
      options: ["[0 1 2]", "[0 3 6 9]", "[0 1 2 3]", "Error"],
      correctIndex: 0,
      explanation: "Reshape to (4,3) in row-major order: first row is [0, 1, 2]. The data is read sequentially regardless of original shape.",
      randomize: false,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why does reshape sometimes fail with ValueError?",
      options: ["Array too large", "New shape must have the same total number of elements as the original array", "Wrong dtype", "Array not contiguous"],
      correctIndex: 1,
      explanation: "prod(new_shape) must equal array.size. You can't reshape 12 elements into (3, 5) because 3*5=15 != 12.",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(24).reshape(2, 3, 4)\nprint(a.reshape(-1).shape)",
      options: ["(24,)", "(2, 3, 4)", "(12,)", "(6, 4)"],
      correctIndex: 0,
      explanation: "reshape(-1) flattens everything to 1D. 2*3*4 = 24 elements -> shape (24,).",
      randomize: false,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What order does reshape use by default?",
      options: ["Column-major (Fortran)", "Row-major (C order): last index changes fastest, elements fill row by row", "Random", "Diagonal"],
      correctIndex: 1,
      explanation: "C order means elements fill row by row. reshape(2,3) on [0..5] gives [[0,1,2],[3,4,5]]. This matches how most people think about matrices.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3, order='F')\nprint(a)",
      options: ["[[0 2 4]\n [1 3 5]]", "[[0 1 2]\n [3 4 5]]", "[[0 3]\n [1 4]\n [2 5]]", "Error"],
      correctIndex: 0,
      explanation: "Fortran order fills column by column: col0=[0,1], col1=[2,3], col2=[4,5]. Result: [[0,2,4],[1,3,5]].",
      randomize: false,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "When would you use order='F' in reshape?",
      options: ["Never", "When interfacing with Fortran libraries or when data is stored column-major in memory", "For speed", "For GPU compatibility"],
      correctIndex: 1,
      explanation: "Some scientific computing libraries (MATLAB, R) use column-major storage. Matching the order avoids unnecessary copies and incorrect data interpretation.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nb = a.ravel()\nprint(b.shape, np.shares_memory(a, b))",
      options: ["(12,) True", "(12,) False", "(3, 4) True", "(4, 3) False"],
      correctIndex: 0,
      explanation: "ravel() returns a flattened view when possible. Shape becomes (12,), and memory is shared with the original.",
      randomize: false,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What's the difference between flatten() and ravel()?",
      options: ["No difference", "flatten() always returns a copy; ravel() returns a view when possible", "flatten is faster", "ravel always copies"],
      correctIndex: 1,
      explanation: "Use ravel() for efficiency (avoids copy). Use flatten() when you need guaranteed independence from the original array.",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.arange(8).reshape(2, 2, 2)\nprint(a[1, 0, 1])",
      options: ["5", "4", "6", "3"],
      correctIndex: 0,
      explanation: "Flat: [0,1,2,3,4,5,6,7]. Reshape (2,2,2): a[1]=[[4,5],[6,7]]. a[1,0]=[4,5]. a[1,0,1]=5.",
      randomize: false,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why is understanding reshape order important for CNNs?",
      options: ["It isn't", "Image data may be stored in different memory layouts (NCHW vs NHWC); wrong reshape order corrupts spatial structure and scrambles pixel positions", "GPUs require specific order", "Reshape order affects dtype"],
      correctIndex: 1,
      explanation: "Reshaping feature maps with wrong order scrambles which pixels go where. A (C,H,W) tensor reshaped assuming (H,W,C) produces meaningless garbage.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nprint(a.T.shape)",
      options: ["(3, 2)", "(2, 3)", "(6,)", "(1, 6)"],
      correctIndex: 0,
      explanation: "Transpose swaps all axes: (2,3).T -> (3,2). Rows become columns, columns become rows. Returns a view.",
      randomize: false,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Can you reshape a non-contiguous array?",
      options: ["Never", "Yes, but NumPy may need to make a copy first to create a contiguous layout for the new shape", "Only with transpose", "Only 1D arrays"],
      correctIndex: 1,
      explanation: "Non-contiguous arrays (e.g., after transpose or strided slicing) can be reshaped, but the result will be a copy since the data can't be reinterpreted as a simple strided view.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In neural networks, why is flatten() used before the classifier head in a CNN?",
      options: ["To sort data", "CNN feature maps are multi-dimensional (N,C,H,W); flatten converts them to 2D (N, features) that dense layers expect", "To reduce memory", "To normalize values"],
      correctIndex: 1,
      explanation: "Conv layers output spatial feature maps. Dense layers expect flat feature vectors. Flatten bridges the gap: (N,64,7,7) -> (N,3136).",
      randomize: true,
    }
  ]}
/>
