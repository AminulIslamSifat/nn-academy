#!/usr/bin/env python3
"""
Generate all missing MDX chapters for NumPy NN Academy.
Creates comprehensive, interactive chapters with PyRunner cells and quizzes.
"""
import os

CONTENT_DIR = "/home/sifat/hdd/projects/numpy-nn-academy/src/content/chapters"

def write_chapter(slug, content):
    path = os.path.join(CONTENT_DIR, f"{slug}.mdx")
    if os.path.exists(path):
        print(f"  SKIP {slug} (exists)")
        return
    with open(path, 'w') as f:
        f.write(content)
    print(f"  ✓ {slug}")

# ═══════════════════════════════════════════════════════════════════════════
# TRACK 1: NumPy Foundations (chapters 006-015)
# ═══════════════════════════════════════════════════════════════════════════

write_chapter("006-data-types", """---
title: "Data Types"
slug: "006-data-types"
description: "NumPy's type system — from bool to complex128. Understand dtype, type promotion, and why choosing the right type matters for memory and speed."
track: "numpy-foundations"
order: 3
read_time: 12
code_time: 8
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes"]
---

# Data Types

Every NumPy array has a **dtype** — a fixed data type that every element shares. This is what makes NumPy fast: homogeneous, contiguous memory.

## The Type Hierarchy

NumPy supports a rich set of types beyond Python's built-ins:

| Category | Types | Bytes |
|----------|-------|-------|
| Boolean | `bool_` | 1 |
| Integer | `int8`, `int16`, `int32`, `int64` | 1–8 |
| Unsigned | `uint8`, `uint16`, `uint32`, `uint64` | 1–8 |
| Float | `float16`, `float32`, `float64` | 2–8 |
| Complex | `complex64`, `complex128` | 8–16 |
| String | `str_`, `bytes_` | variable |

<PyRunner
  cellId="006-cell-1"
  defaultCode={`import numpy as np

# Check default types
a = np.array([1, 2, 3])
b = np.array([1.0, 2.0, 3.0])
c = np.array([True, False, True])

print(f"Integer array dtype: {a.dtype}")
print(f"Float array dtype:   {b.dtype}")
print(f"Boolean array dtype: {c.dtype}")

# Explicit dtype
small = np.array([1, 2, 3], dtype=np.int8)
print(f"\nExplicit int8: {small.dtype}, itemsize: {small.itemsize} byte")

big = np.array([1, 2, 3], dtype=np.float64)
print(f"Explicit float64: {big.dtype}, itemsize: {big.itemsize} bytes")
`}
/>

## Type Promotion

When you combine arrays of different types, NumPy promotes to the "safer" type:

<PyRunner
  cellId="006-cell-2"
  defaultCode={`import numpy as np

int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.5], dtype=np.float32)

result = int_arr[:2] + float_arr
print(f"int32 + float32 → {result.dtype}")

# The promotion rules: bool < int < uint < float < complex
print(f"\nnp.result_type(np.int8, np.float32) = {np.result_type(np.int8, np.float32)}")
print(f"np.result_type(np.float32, np.complex64) = {np.result_type(np.float32, np.complex64)}")
`}
/>

<Callout type="warning" title="Silent Overflow">
Integer types silently overflow! `np.uint8(255) + 1` gives `0`, not `256`. Use `np.int64` or `np.float64` when in doubt.
</Callout>

## Memory Impact

Choosing the right dtype can save massive amounts of memory:

<PyRunner
  cellId="006-cell-3"
  defaultCode={`import numpy as np

n = 1_000_000

arr_f64 = np.ones(n, dtype=np.float64)
arr_f32 = np.ones(n, dtype=np.float32)
arr_f16 = np.ones(n, dtype=np.float16)
arr_i8  = np.ones(n, dtype=np.int8)

print(f"float64: {arr_f64.nbytes / 1e6:.1f} MB")
print(f"float32: {arr_f32.nbytes / 1e6:.1f} MB")
print(f"float16: {arr_f16.nbytes / 1e6:.1f} MB")
print(f"int8:    {arr_i8.nbytes / 1e6:.1f} MB")
print(f"\nSavings: float64→float32 = 2x, float64→int8 = 8x")
`}
/>

## Converting Types

Use `.astype()` to convert — it always creates a copy:

<PyRunner
  cellId="006-cell-4"
  defaultCode={`import numpy as np

a = np.array([1.7, 2.9, 3.1])
print(f"Original: {a} (dtype={a.dtype})")

as_int = a.astype(np.int32)
print(f"As int32: {as_int} (truncated, not rounded!)")

as_str = a.astype(str)
print(f"As string: {as_str} (dtype={as_str.dtype})")

# Round-trip danger
original = np.array([1, 2, 3], dtype=np.int64)
small = original.astype(np.int8)
back = small.astype(np.int64)
print(f"\nRound-trip int64→int8→int64: {back} (safe for small values)")
`}
/>

<Quiz
  chapterSlug="006-data-types"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the default dtype of np.array([1, 2, 3]) on a 64-bit system?",
      options: ["int64", "int32", "float64", "object"],
      correctIndex: 0,
      explanation: "On 64-bit systems, NumPy defaults to int64 for integer literals.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does np.uint8(255) + np.uint8(1) return?",
      code: "import numpy as np\nresult = np.uint8(255) + np.uint8(1)\nprint(result)",
      options: ["0", "256", "OverflowError", "-1"],
      correctIndex: 0,
      explanation: "uint8 wraps around: 255 + 1 = 256 mod 256 = 0. Silent overflow!",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "How much memory does np.ones(1_000_000, dtype=np.float32) use?",
      options: ["4 MB", "8 MB", "2 MB", "1 MB"],
      correctIndex: 0,
      explanation: "float32 = 4 bytes per element × 1,000,000 = 4,000,000 bytes = 4 MB.",
      randomize: true
    }
  ]}
/>
""")

write_chapter("007-copy-vs-view", """---
title: "Copy vs View"
slug: "007-copy-vs-view"
description: "The critical distinction between views (shared memory) and copies (independent data). Master .copy(), .view(), and base attribute to avoid subtle bugs."
track: "numpy-foundations"
order: 4
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["002-indexing-and-slicing"]
---

# Copy vs View

This is the #1 source of subtle bugs in NumPy. Understanding when you're looking at the same data vs a separate copy is essential.

## Views Share Memory

A **view** is a new array object that looks at the same data buffer:

<PyRunner
  cellId="007-cell-1"
  defaultCode={`import numpy as np

original = np.array([10, 20, 30, 40, 50])
view = original[1:4]  # This is a VIEW, not a copy!

print(f"Original: {original}")
print(f"View:     {view}")
print(f"Shares memory: {np.shares_memory(original, view)}")

# Modify the view → original changes!
view[0] = 999
print(f"\nAfter view[0] = 999:")
print(f"Original: {original}")  # [10, 999, 30, 40, 50]
print(f"View:     {view}")
`}
/>

## Copies Are Independent

Use `.copy()` to get an independent array:

<PyRunner
  cellId="007-cell-2"
  defaultCode={`import numpy as np

original = np.array([10, 20, 30, 40, 50])
copy = original[1:4].copy()  # Explicit copy

print(f"Shares memory: {np.shares_memory(original, copy)}")

copy[0] = 999
print(f"Original: {original}")  # Unchanged!
print(f"Copy:     {copy}")
`}
/>

## The .base Attribute

Every view has a `.base` pointing to the original array that owns the memory:

<PyRunner
  cellId="007-cell-3"
  defaultCode={`import numpy as np

a = np.arange(12).reshape(3, 4)
b = a[:, 1]      # view of column 1
c = a.copy()     # independent copy

print(f"b.base is a: {b.base is a}")
print(f"c.base is None: {c.base is None}")
print(f"b.flags.owndata: {b.flags.owndata}")
print(f"c.flags.owndata: {c.flags.owndata}")
`}
/>

<Callout type="danger" title="Common Pitfall">
Fancy indexing (with lists/arrays) always returns a **copy**, but basic slicing always returns a **view**. This asymmetry catches everyone.
</Callout>

## When Views Matter in Practice

<PyRunner
  cellId="007-cell-4"
  defaultCode={`import numpy as np
import time

# Large array
big = np.random.randn(10_000_000)

# View: instant, zero memory
start = time.perf_counter()
view = big[::2]  # every other element
view_time = time.perf_counter() - start

# Copy: allocates new memory
start = time.perf_counter()
copy = big[::2].copy()
copy_time = time.perf_counter() - start

print(f"View creation: {view_time*1e6:.1f} μs")
print(f"Copy creation: {copy_time*1e6:.1f} μs")
print(f"View memory: {view.nbytes / 1e6:.1f} MB (shared)")
print(f"Copy memory: {copy.nbytes / 1e6:.1f} MB (new allocation)")
`}
/>

<Quiz
  chapterSlug="007-copy-vs-view"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Which operation returns a VIEW (not a copy)?",
      options: ["a[1:5]", "a[[1,3,5]]", "a[a > 3]", "a.flatten()"],
      correctIndex: 0,
      explanation: "Basic slicing returns a view. Fancy indexing (lists), boolean masks, and flatten() all return copies.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is x.base after: x = np.arange(10)[2:5].copy()?",
      code: "import numpy as np\nx = np.arange(10)[2:5].copy()\nprint(x.base)",
      options: ["None", "The original arange array", "The slice [2:5]", "An error"],
      correctIndex: 0,
      explanation: ".copy() creates an independent array that owns its own data, so .base is None.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "np.shares_memory(a, b) returns True when:",
      options: ["b is a view of a (or vice versa)", "a and b have the same shape", "a and b have the same values", "b was created with .copy()"],
      correctIndex: 0,
      explanation: "shares_memory checks if two arrays share any underlying data buffer.",
      randomize: true
    }
  ]}
/>
""")

write_chapter("008-array-shape", """---
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
print(f"\nrow + bias = {row + bias}")         # (3,) + (3,) → (3,)
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
print(f"\nexpand_dims(a, 0): {np.expand_dims(a, 0).shape}")
print(f"expand_dims(a, 1): {np.expand_dims(a, 1).shape}")

# Squeeze removes size-1 dimensions
padded = np.zeros((1, 3, 1, 5))
print(f"\nPadded: {padded.shape}")
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
print(f"\nMemory layout (row-major / C-order):")
print(a)
print(f"\nTo go from a[i,j] to a[i,j+1]: step {a.strides[1]} bytes (1 element)")
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
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a[np.newaxis, :] do to a 1D array of shape (5,)?",
      options: ["Adds a leading dimension → (1, 5)", "Adds a trailing dimension → (5, 1)", "Does nothing", "Flattens the array"],
      correctIndex: 0,
      explanation: "np.newaxis inserts a new axis at that position. [newaxis, :] adds dim at axis 0.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is arr.strides for arr = np.arange(6, dtype=np.float64).reshape(2, 3)?",
      code: "import numpy as np\narr = np.arange(6, dtype=np.float64).reshape(2, 3)\nprint(arr.strides)",
      options: ["(24, 8)", "(8, 24)", "(16, 8)", "(8, 16)"],
      correctIndex: 0,
      explanation: "Row-major: stride[0] = 3 elements × 8 bytes = 24, stride[1] = 1 element × 8 bytes = 8.",
      randomize: true
    }
  ]}
/>
""")

write_chapter("009-array-reshape", """---
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
print(f"Reshaped (3,4):\n{b}")

c = a.reshape(2, 2, 3)
print(f"Reshaped (2,2,3): shape={c.shape}")

# Total elements must match!
try:
    a.reshape(5, 3)  # 15 ≠ 12
except ValueError as e:
    print(f"\nError: {e}")
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
print(f"\nBatch: {batch.shape} → Flat: {flat.shape} → Restored: {restored.shape}")
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
print(f"\na[0,0] after b[0,0]=999: {a[0,0]}")
`}
/>

## Transpose

Transpose swaps axes — also a view, no data copied:

<PyRunner
  cellId="009-cell-4"
  defaultCode={`import numpy as np

a = np.arange(6).reshape(2, 3)
print(f"Original:\n{a}  shape={a.shape}")

print(f"\n.T (transpose):\n{a.T}  shape={a.T.shape}")

# For higher dimensions, specify axis order
t = np.zeros((2, 3, 4))
print(f"\n3D tensor: {t.shape}")
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
print(f"\nravel('C'): {a.ravel('C')}")
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
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Does reshape always return a view?",
      options: ["Usually yes, but may copy if memory isn't contiguous", "Always yes", "Always no", "Only for 2D arrays"],
      correctIndex: 0,
      explanation: "Reshape returns a view when the memory layout allows it. After transpose or non-contiguous slicing, it may need to copy.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What's the difference between ravel() and flatten()?",
      options: ["ravel returns a view when possible; flatten always copies", "They are identical", "flatten is faster", "ravel only works on 2D"],
      correctIndex: 0,
      explanation: "ravel() tries to return a view (zero-copy), flatten() always allocates a new array.",
      randomize: true
    }
  ]}
/>
""")

write_chapter("010-array-iterating", """---
title: "Array Iteration"
slug: "010-array-iterating"
description: "How to iterate over NumPy arrays — and why you almost never should. nditer, flat, and the vectorization alternative."
track: "numpy-foundations"
order: 7
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["003-vectorization"]
---

# Array Iteration

You *can* iterate over NumPy arrays. You almost never *should*. But knowing how helps you understand why vectorization is better.

## Basic Iteration

<PyRunner
  cellId="010-cell-1"
  defaultCode={`import numpy as np

a = np.arange(6).reshape(2, 3)

# Iterating a 2D array iterates over rows
for row in a:
    print(f"Row: {row}")

# Iterate every element with .flat
print("\nAll elements:")
for val in a.flat:
    print(val, end=" ")
print()
`}
/>

## Why Iteration Is Slow

<PyRunner
  cellId="010-cell-2"
  defaultCode={`import numpy as np
import time

n = 1_000_000
arr = np.random.randn(n)

# Python loop
start = time.perf_counter()
result_loop = 0.0
for x in arr:
    result_loop += x * x
loop_time = time.perf_counter() - start

# Vectorized
start = time.perf_counter()
result_vec = np.sum(arr * arr)
vec_time = time.perf_counter() - start

print(f"Python loop: {loop_time*1000:.1f} ms")
print(f"Vectorized:  {vec_time*1000:.3f} ms")
print(f"Speedup:     {loop_time/vec_time:.0f}x")
print(f"Results match: {np.isclose(result_loop, result_vec)}")
`}
/>

<Callout type="tip" title="Rule of Thumb">
If you see a `for` loop over a NumPy array, ask yourself: can this be a single array operation? The answer is almost always yes.
</Callout>

## nditer: When You Must Iterate

For cases where you genuinely need element-wise iteration (e.g., custom logic that can't be vectorized):

<PyRunner
  cellId="010-cell-3"
  defaultCode={`import numpy as np

a = np.arange(6).reshape(2, 3)
b = np.arange(6, 12).reshape(2, 3)

# nditer can iterate multiple arrays simultaneously
for x, y in np.nditer([a, b]):
    print(f"{x} + {y} = {x + y}", end="  ")
print()

# With write access
result = np.zeros_like(a)
for x, y, r in np.nditer([a, b, result], op_flags=[['readonly'], ['readonly'], ['writeonly']]):
    r[...] = x + y
print(f"\nResult via nditer:\n{result}")
`}
/>

## enumerate and zip with Arrays

<PyRunner
  cellId="010-cell-4"
  defaultCode={`import numpy as np

names = np.array(["Alice", "Bob", "Charlie"])
scores = np.array([95, 87, 92])

# zip works naturally
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# enumerate for index
print()
for i, val in enumerate(scores):
    print(f"Student {i}: {val}")
`}
/>

<Quiz
  chapterSlug="010-array-iterating"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Iterating a 2D NumPy array with 'for row in arr' iterates over:",
      options: ["Rows (first axis)", "Individual elements", "Columns", "Diagonals"],
      correctIndex: 0,
      explanation: "Iteration over an ndarray iterates along the first axis. For 2D, that means rows.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why is vectorized code faster than Python loops over NumPy arrays?",
      options: ["C-level loops avoid Python interpreter overhead per element", "NumPy uses GPU automatically", "Python loops have a bug", "It's not actually faster"],
      correctIndex: 0,
      explanation: "Vectorized ops run tight C loops without Python's per-element type checking and GIL overhead.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does list(np.arange(6).reshape(2,3).flat) return?",
      code: "import numpy as np\nprint(list(np.arange(6).reshape(2,3).flat))",
      options: ["[0, 1, 2, 3, 4, 5]", "[[0,1,2],[3,4,5]]", "[0, 3, 1, 4, 2, 5]", "Error"],
      correctIndex: 0,
      explanation: ".flat gives a 1D iterator over all elements in row-major (C) order.",
      randomize: true
    }
  ]}
/>
""")

write_chapter("011-array-join", """---
title: "Joining Arrays"
slug: "011-array-join"
description: "Combine arrays with concatenate, stack, hstack, vstack, dstack. Know which function to reach for."
track: "numpy-foundations"
order: 8
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["008-array-shape"]
---

# Joining Arrays

NumPy gives you multiple ways to combine arrays. Each serves a different purpose.

## concatenate: The General Tool

<PyRunner
  cellId="011-cell-1"
  defaultCode={`import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Along axis 0 (rows)
print("axis=0:")
print(np.concatenate([a, b], axis=0))

# Along axis 1 (columns)
print("\naxis=1:")
print(np.concatenate([a, b], axis=1))

# 1D arrays
c = np.array([1, 2, 3])
d = np.array([4, 5, 6])
print(f"\n1D concat: {np.concatenate([c, d])}")
`}
/>

## Stack: New Dimension

Unlike concatenate, `stack` creates a **new axis**:

<PyRunner
  cellId="011-cell-2"
  defaultCode={`import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a: {a.shape}, b: {b.shape}")
print(f"stack:    {np.stack([a, b]).shape}")      # (2, 3) — new axis 0
print(f"stack ax1: {np.stack([a, b], axis=1).shape}") # (3, 2) — new axis 1

# Compare with concatenate
print(f"concat:   {np.concatenate([a, b]).shape}")  # (6,) — no new axis
`}
/>

## Convenience Functions

<PyRunner
  cellId="011-cell-3"
  defaultCode={`import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# hstack = horizontal (axis=1 for 2D, axis=0 for 1D)
print(f"hstack: {np.hstack([a, b])}")

# vstack = vertical (always adds row dimension)
print(f"vstack:\n{np.vstack([a, b])}")

# dstack = depth (axis=2)
print(f"dstack shape: {np.dstack([a, b]).shape}")

# column_stack treats 1D as columns
print(f"column_stack:\n{np.column_stack([a, b])}")
`}
/>

## Practical: Building Feature Matrices

<PyRunner
  cellId="011-cell-4"
  defaultCode={`import numpy as np

# Simulate combining feature vectors into a dataset
feature1 = np.random.randn(100, 3)   # 100 samples, 3 features
feature2 = np.random.randn(100, 5)   # 100 samples, 5 features
feature3 = np.random.randn(100, 2)   # 100 samples, 2 features

# Combine all features
dataset = np.concatenate([feature1, feature2, feature3], axis=1)
print(f"Combined dataset: {dataset.shape}")  # (100, 10)

# Add bias column (column of 1s)
bias = np.ones((100, 1))
dataset_with_bias = np.hstack([bias, dataset])
print(f"With bias: {dataset_with_bias.shape}")  # (100, 11)
`}
/>

<Quiz
  chapterSlug="011-array-join"
  questions={[
    {
      id: "q1",
      type: "shape-prediction",
      prompt: "What is np.stack([np.ones(3), np.zeros(3)]).shape?",
      options: ["(2, 3)", "(6,)", "(3, 2)", "(2, 1, 3)"],
      correctIndex: 0,
      explanation: "stack creates a new axis at position 0 by default. Two (3,) arrays → (2, 3).",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What's the difference between concatenate and stack?",
      options: ["concatenate joins along existing axis; stack creates a new axis", "They are identical", "stack is faster", "concatenate only works on 1D"],
      correctIndex: 0,
      explanation: "concatenate joins along an existing axis (no new dim). stack adds a brand new axis.",
      randomize: true
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "np.vstack([np.array([1,2,3]), np.array([4,5,6])]).shape is:",
      options: ["(2, 3)", "(6,)", "(3, 2)", "(1, 6)"],
      correctIndex: 0,
      explanation: "vstack stacks vertically — each 1D array becomes a row. Result is (2, 3).",
      randomize: true
    }
  ]}
/>
""")

write_chapter("012-array-split", """---
title: "Splitting Arrays"
slug: "012-array-split"
description: "Break arrays apart with split, hsplit, vsplit, dsplit, and array_split for uneven divisions."
track: "numpy-foundations"
order: 9
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["011-array-join"]
---

# Splitting Arrays

The inverse of joining — break one array into multiple sub-arrays.

## Basic Split

<PyRunner
  cellId="012-cell-1"
  defaultCode={`import numpy as np

a = np.arange(12).reshape(3, 4)
print(f"Original:\n{a}\n")

# Split into 2 along axis 1 (columns)
left, right = np.split(a, 2, axis=1)
print(f"Left:\n{left}")
print(f"Right:\n{right}")

# Split into 3 along axis 0 (rows)
parts = np.split(a, 3, axis=0)
for i, p in enumerate(parts):
    print(f"Part {i}: {p.flatten()}")
`}
/>

## Split at Specific Indices

<PyRunner
  cellId="012-cell-2"
  defaultCode={`import numpy as np

a = np.arange(10)

# Split at indices [3, 7]
p1, p2, p3 = np.split(a, [3, 7])
print(f"[0:3]  = {p1}")
print(f"[3:7]  = {p2}")
print(f"[7:10] = {p3}")
`}
/>

## array_split: Uneven Divisions

`np.split` requires even division. `np.array_split` handles remainders:

<PyRunner
  cellId="012-cell-3"
  defaultCode={`import numpy as np

a = np.arange(10)

# 10 elements into 3 parts → uneven!
try:
    np.split(a, 3)
except ValueError as e:
    print(f"split error: {e}")

parts = np.array_split(a, 3)
for i, p in enumerate(parts):
    print(f"Part {i}: {p} (len={len(p)})")
`}
/>

## hsplit, vsplit, dsplit

<PyRunner
  cellId="012-cell-4"
  defaultCode={`import numpy as np

a = np.arange(12).reshape(3, 4)

# hsplit = split along axis 1
left, right = np.hsplit(a, 2)
print(f"hsplit: left={left.shape}, right={right.shape}")

# vsplit = split along axis 0
top, mid, bot = np.vsplit(a, 3)
print(f"vsplit: top={top.shape}, mid={mid.shape}, bot={bot.shape}")

# dsplit for 3D
b = np.arange(24).reshape(2, 3, 4)
parts = np.dsplit(b, 2)
print(f"dsplit: {[p.shape for p in parts]}")
`}
/>

## Practical: Train/Test Split

<PyRunner
  cellId="012-cell-5"
  defaultCode={`import numpy as np

# Simple train/test split without sklearn
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

split_idx = 80
X_train, X_test = np.split(X, [split_idx])
y_train, y_test = np.split(y, [split_idx])

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")
`}
/>

<Quiz
  chapterSlug="012-array-split"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What happens when you np.split an array of 10 elements into 3 parts?",
      options: ["ValueError — uneven split", "Returns 3 parts of sizes 4,3,3", "Returns 3 parts of sizes 3,3,4", "Pads with zeros"],
      correctIndex: 0,
      explanation: "np.split requires even division. Use np.array_split for uneven splits.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does np.split(np.arange(6), [2, 4]) return?",
      code: "import numpy as np\nparts = np.split(np.arange(6), [2, 4])\nprint([list(p) for p in parts])",
      options: ["[[0,1], [2,3], [4,5]]", "[[0,1,2], [3,4,5]]", "[[0,1], [2,3,4,5]]", "Error"],
      correctIndex: 0,
      explanation: "Split at indices 2 and 4: [0:2], [2:4], [4:6] → three equal parts.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "np.hsplit is equivalent to np.split with axis=?",
      options: ["axis=1", "axis=0", "axis=2", "axis=-1"],
      correctIndex: 0,
      explanation: "hsplit splits horizontally, which means along axis 1 (columns).",
      randomize: true
    }
  ]}
/>
""")

write_chapter("013-array-search", """---
title: "Searching Arrays"
slug: "013-array-search"
description: "Find elements with where, argwhere, nonzero, searchsorted, and extract. Locate data efficiently."
track: "numpy-foundations"
order: 10
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["002-indexing-and-slicing"]
---

# Searching Arrays

Finding elements that meet conditions is fundamental to data processing and neural network logic.

## np.where: The Workhorse

<PyRunner
  cellId="013-cell-1"
  defaultCode={`import numpy as np

a = np.array([10, 25, 3, 47, 8, 32])

# Find indices where condition is true
indices = np.where(a > 20)
print(f"Indices where a > 20: {indices[0]}")
print(f"Values: {a[indices]}")

# Three-argument form: conditional replacement
result = np.where(a > 20, a, 0)  # keep if >20, else 0
print(f"\nConditional: {result}")

# 2D example
matrix = np.random.randint(0, 10, (3, 4))
print(f"\nMatrix:\n{matrix}")
rows, cols = np.where(matrix > 5)
print(f"Elements > 5 at: {list(zip(rows, cols))}")
`}
/>

## argwhere and nonzero

<PyRunner
  cellId="013-cell-2"
  defaultCode={`import numpy as np

a = np.array([[0, 1, 0], [2, 0, 3], [0, 4, 0]])

# argwhere returns coordinates as (N, ndim) array
coords = np.argwhere(a > 0)
print(f"argwhere(a > 0):\n{coords}")

# nonzero returns tuple of arrays (one per axis)
rows, cols = np.nonzero(a)
print(f"\nnonzero rows: {rows}")
print(f"nonzero cols: {cols}")

# Count non-zero elements
print(f"\nCount non-zero: {np.count_nonzero(a)}")
`}
/>

## searchsorted: Binary Search

<PyRunner
  cellId="013-cell-3"
  defaultCode={`import numpy as np

sorted_arr = np.array([1, 3, 5, 7, 9, 11])

# Where would 6 be inserted to maintain order?
idx = np.searchsorted(sorted_arr, 6)
print(f"Insert 6 at index: {idx}")
print(f"Result: {np.insert(sorted_arr, idx, 6)}")

# Multiple values
indices = np.searchsorted(sorted_arr, [0, 4, 8, 12])
print(f"\nInsert [0,4,8,12] at: {indices}")

# side='right' for insertion after existing values
print(f"searchsorted(5, 'left'):  {np.searchsorted(sorted_arr, 5, side='left')}")
print(f"searchsorted(5, 'right'): {np.searchsorted(sorted_arr, 5, side='right')}")
`}
/>

## extract and place

<PyRunner
  cellId="013-cell-4"
  defaultCode={`import numpy as np

a = np.arange(10)
condition = a % 3 == 0

# Extract elements matching condition
extracted = np.extract(condition, a)
print(f"Multiples of 3: {extracted}")

# place: inverse of extract (modify in-place)
b = np.arange(10)
np.place(b, b % 3 == 0, [99, 88, 77, 66])
print(f"After place: {b}")
`}
/>

<Quiz
  chapterSlug="013-array-search"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "What does np.where(np.array([1,0,3,0,5]) > 0, 1, -1) return?",
      code: "import numpy as np\nprint(np.where(np.array([1,0,3,0,5]) > 0, 1, -1))",
      options: ["[ 1 -1  1 -1  1]", "[0, 2, 4]", "[1, 3, 5]", "Error"],
      correctIndex: 0,
      explanation: "Three-arg where: condition true → 1, false → -1. Elements 1,3,5 are >0.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "np.searchsorted([1,3,5,7], 4) returns:",
      options: ["2", "1", "3", "4"],
      correctIndex: 0,
      explanation: "4 would be inserted at index 2 to maintain sorted order: [1,3,4,5,7].",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does np.count_nonzero([[0,1],[2,0]]) return?",
      options: ["2", "4", "1", "3"],
      correctIndex: 0,
      explanation: "Two non-zero elements: 1 and 2.",
      randomize: true
    }
  ]}
/>
""")

write_chapter("014-array-sort", """---
title: "Sorting Arrays"
slug: "014-array-sort"
description: "Sort with sort, argsort, partition, and lexsort. In-place vs out-of-place, stable sorts, and partial sorting."
track: "numpy-foundations"
order: 11
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes"]
---

# Sorting Arrays

NumPy provides efficient sorting algorithms optimized for numerical data.

## sort vs argsort

<PyRunner
  cellId="014-cell-1"
  defaultCode={`import numpy as np

a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# sort returns sorted copy
sorted_a = np.sort(a)
print(f"Original: {a}")
print(f"Sorted:   {sorted_a}")

# argsort returns indices that would sort the array
indices = np.argsort(a)
print(f"Argsort:  {indices}")
print(f"Verify:   {a[indices]}")

# In-place sort
a.sort()
print(f"In-place: {a}")
`}
/>

## Sorting Along Axes

<PyRunner
  cellId="014-cell-2"
  defaultCode={`import numpy as np

matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
print(f"Original:\n{matrix}\n")

# Sort each row (axis=1)
print(f"Sort rows:\n{np.sort(matrix, axis=1)}\n")

# Sort each column (axis=0)
print(f"Sort cols:\n{np.sort(matrix, axis=0)}\n")

# Flatten and sort
print(f"Sort all: {np.sort(matrix, axis=None)}")
`}
/>

## argsort: The Power Move

argsort is incredibly useful for ranking and reordering:

<PyRunner
  cellId="014-cell-3"
  defaultCode={`import numpy as np

names = np.array(["Charlie", "Alice", "Bob", "Diana"])
scores = np.array([72, 95, 88, 91])

# Sort by scores (descending)
order = np.argsort(scores)[::-1]
print("Ranking by score:")
for rank, idx in enumerate(order):
    print(f"  {rank+1}. {names[idx]}: {scores[idx]}")

# Sort multiple arrays together
print(f"\nNames sorted by score: {names[order]}")
print(f"Scores sorted:         {scores[order]}")
`}
/>

## partition: Partial Sort (Top-K)

When you only need the top/bottom K elements, partition is O(n) instead of O(n log n):

<PyRunner
  cellId="014-cell-4"
  defaultCode={`import numpy as np
import time

a = np.random.randn(1_000_000)

# Full sort
start = time.perf_counter()
top10_sort = np.sort(a)[-10:]
sort_time = time.perf_counter() - start

# Partition: only guarantee top 10 are correct
start = time.perf_counter()
top10_part = np.partition(a, -10)[-10:]
part_time = time.perf_counter() - start

print(f"Full sort top-10:  {sort_time*1000:.1f} ms")
print(f"Partition top-10:  {part_time*1000:.1f} ms")
print(f"Speedup: {sort_time/part_time:.1f}x")
print(f"Same values: {np.allclose(np.sort(top10_sort), np.sort(top10_part))}")
`}
/>

<Quiz
  chapterSlug="014-array-sort"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does np.argsort([3, 1, 2]) return?",
      options: ["[1, 2, 0]", "[1, 2, 3]", "[3, 1, 2]", "[0, 1, 2]"],
      correctIndex: 0,
      explanation: "argsort returns indices: element at index 1 (value 1) is smallest, then index 2 (value 2), then index 0 (value 3).",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why use np.partition instead of np.sort for top-K?",
      options: ["partition is O(n) vs sort's O(n log n)", "partition uses less memory", "partition is more accurate", "They are identical"],
      correctIndex: 0,
      explanation: "partition only partially sorts — guarantees the K-th element is in place, smaller before, larger after. O(n) average.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does np.sort([[3,1],[4,2]], axis=0) return?",
      code: "import numpy as np\nprint(np.sort([[3,1],[4,2]], axis=0))",
      options: ["[[3 1] [4 2]]", "[[1 3] [2 4]]", "[[3 4] [1 2]]", "[[4 2] [3 1]]"],
      correctIndex: 0,
      explanation: "axis=0 sorts each column: col0 [3,4]→[3,4], col1 [1,2]→[1,2]. Already sorted!",
      randomize: true
    }
  ]}
/>
""")

write_chapter("015-array-filter", """---
title: "Filtering Arrays"
slug: "015-array-filter"
description: "Boolean masking, fancy indexing, and np.take/put for selective element access and modification."
track: "numpy-foundations"
order: 12
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["002-indexing-and-slicing", "013-array-search"]
---

# Filtering Arrays

Boolean masks are NumPy's most powerful filtering tool — expressive, fast, and composable.

## Boolean Masking

<PyRunner
  cellId="015-cell-1"
  defaultCode={`import numpy as np

a = np.array([15, 3, 28, 7, 42, 11, 33])

# Create a boolean mask
mask = a > 20
print(f"Array: {a}")
print(f"Mask:  {mask}")
print(f"Filtered: {a[mask]}")

# Combine conditions with & | ~
print(f"\nBetween 10 and 30: {a[(a > 10) & (a < 30)]}")
print(f"Not divisible by 3: {a[~(a % 3 == 0)]}")
print(f"Either < 5 or > 30: {a[(a < 5) | (a > 30)]}")
`}
/>

<Callout type="warning" title="Use & | ~ not and or not">
Python's `and`/`or`/`not` don't work element-wise on arrays. Always use `&` (and), `|` (or), `~` (not) with parentheses around each condition.
</Callout>

## Modifying with Masks

<PyRunner
  cellId="015-cell-2"
  defaultCode={`import numpy as np

a = np.arange(10)
print(f"Before: {a}")

# Set all even numbers to -1
a[a % 2 == 0] = -1
print(f"After masking evens: {a}")

# Clip values to a range
b = np.random.randn(8) * 10
print(f"\nRandom: {b.round(2)}")
b[np.abs(b) > 5] = np.sign(b[np.abs(b) > 5]) * 5
print(f"Clipped to ±5: {b.round(2)}")
`}
/>

## Fancy Indexing

Pass arrays of indices to select arbitrary elements:

<PyRunner
  cellId="015-cell-3"
  defaultCode={`import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])

# Select specific indices
indices = [0, 2, 5]
print(f"Fancy index [0,2,5]: {a[indices]}")

# 2D fancy indexing
matrix = np.arange(12).reshape(3, 4)
rows = [0, 1, 2]
cols = [1, 2, 3]
print(f"\nMatrix:\n{matrix}")
print(f"matrix[[0,1,2], [1,2,3]] = {matrix[rows, cols]}")

# Reorder rows
print(f"\nReversed rows:\n{matrix[[2, 1, 0]]}")
`}
/>

## take and put

<PyRunner
  cellId="015-cell-4"
  defaultCode={`import numpy as np

a = np.array([10, 20, 30, 40, 50])

# take: like fancy indexing but as a method
print(f"take([1,3,4]): {a.take([1, 3, 4])}")

# put: modify specific indices in-place
a.put([0, 2, 4], [99, 88, 77])
print(f"after put: {a}")

# take along axis
matrix = np.arange(12).reshape(3, 4)
print(f"\ntake along axis=1: {matrix.take([0, 2], axis=1)}")
`}
/>

<Quiz
  chapterSlug="015-array-filter"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "What does np.array([1,2,3,4,5])[(np.array([1,2,3,4,5]) % 2 == 0)] return?",
      code: "import numpy as np\na = np.array([1,2,3,4,5])\nprint(a[a % 2 == 0])",
      options: ["[2 4]", "[1 3 5]", "[True False True False True]", "Error"],
      correctIndex: 0,
      explanation: "Boolean mask selects elements where condition is True: 2 and 4 are even.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why must you use & instead of 'and' for combining NumPy boolean conditions?",
      options: ["'and' operates on whole arrays as single booleans; & is element-wise", "They are interchangeable", "& is faster", "'and' causes a syntax error"],
      correctIndex: 0,
      explanation: "Python's 'and' returns one of its operands (truthiness of whole array). & performs element-wise bitwise AND.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Does fancy indexing (a[[1,3,5]]) return a view or copy?",
      options: ["Copy", "View", "Depends on the array", "Neither"],
      correctIndex: 0,
      explanation: "Fancy indexing always returns a copy, unlike basic slicing which returns a view.",
      randomize: true
    }
  ]}
/>
""")

print("\n✅ Track 1 (NumPy Foundations) complete: chapters 006-015")
