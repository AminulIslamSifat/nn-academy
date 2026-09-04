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
      type: "shape-prediction",
      prompt: "What is the shape of np.arange(24).reshape(2, -1, 4)?",
      options: ["(2, 3, 4)", "(2, 4, 3)", "(2, 12)", "Error"],
      correctIndex: 0,
      explanation: "24 / (2 × 4) = 3, so -1 resolves to 3. Shape is (2, 3, 4).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Does reshape always return a view?",
      options: ["Usually yes, but may copy if memory isn't contiguous", "Always yes", "Always no", "Only for 2D arrays"],
      correctIndex: 0,
      explanation: "Reshape returns a view when the memory layout allows it. After transpose or non-contiguous slicing, it may need to copy.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What's the difference between ravel() and flatten()?",
      options: ["ravel returns a view when possible; flatten always copies", "They are identical", "flatten is faster", "ravel only works on 2D"],
      correctIndex: 0,
      explanation: "ravel() tries to return a view (zero-copy), flatten() always allocates a new array.",
      randomize: true,
    }
  ]}
/>
