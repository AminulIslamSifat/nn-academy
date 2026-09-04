---
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
print("
All elements:")
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
print(f"
Result via nditer:
{result}")
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
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why is vectorized code faster than Python loops over NumPy arrays?",
      options: ["C-level loops avoid Python interpreter overhead per element", "NumPy uses GPU automatically", "Python loops have a bug", "It's not actually faster"],
      correctIndex: 0,
      explanation: "Vectorized ops run tight C loops without Python's per-element type checking and GIL overhead.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does list(np.arange(6).reshape(2,3).flat) return?",
      code: "import numpy as np\nprint(list(np.arange(6).reshape(2,3).flat))",
      options: ["[0, 1, 2, 3, 4, 5]", "[[0,1,2],[3,4,5]]", "[0, 3, 1, 4, 2, 5]", "Error"],
      correctIndex: 0,
      explanation: ".flat gives a 1D iterator over all elements in row-major (C) order.",
      randomize: true,
    }
  ]}
/>
