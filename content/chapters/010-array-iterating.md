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
      prompt: "When you write `for row in arr` on a 2D NumPy array, what does each iteration yield?",
      options: ["Individual scalar elements", "Rows (1D sub-arrays along the first axis)", "Columns", "The entire array"],
      correctIndex: 1,
      explanation: "Iteration over an ndarray always goes along the first axis. For a (3,4) array, you get 3 iterations, each yielding a shape-(4,) row.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why is iterating over NumPy arrays with Python for-loops so much slower than vectorized operations?",
      options: ["NumPy arrays are stored differently", "Each loop iteration pays Python interpreter overhead (type checking, object dispatch); vectorized ops run tight C loops with no per-element Python cost", "Python loops have a bug with NumPy", "They're actually the same speed"],
      correctIndex: 1,
      explanation: "A Python for-loop over 1M elements invokes the interpreter 1M times. A vectorized operation processes all 1M elements in a single compiled C call. Typical speedup: 50-100x.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does this print?",
      code: "import numpy as np\nprint(list(np.arange(6).reshape(2, 3).flat))",
      options: ["[0, 1, 2, 3, 4, 5]", "[[0, 1, 2], [3, 4, 5]]", "[0, 3, 1, 4, 2, 5]", "Error"],
      correctIndex: 0,
      explanation: ".flat provides a 1D iterator over all elements in row-major (C) order, regardless of the array's shape.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is .flat?",
      options: ["A method that flattens the array into a copy", "A 1D iterator over all elements of the array, without creating a copy", "A property that returns the flattened shape", "An alias for reshape(-1)"],
      correctIndex: 1,
      explanation: ".flat is an iterator object, not an array. It yields elements one at a time in C order without allocating new memory. Useful when you must iterate but want to avoid flatten()'s copy.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nfor row in a:\n    print(row.shape)",
      options: ["(4,) printed 3 times", "(3,) printed 4 times", "(12,) printed once", "Error"],
      correctIndex: 0,
      explanation: "Iterating a (3,4) array yields 3 rows, each with shape (4,). First axis is consumed by the loop.",
      randomize: false,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What is np.nditer used for?",
      options: ["Faster vectorized computation", "Efficient multi-array iteration when you genuinely need element-wise access that can't be vectorized", "Sorting arrays", "Reshaping"],
      correctIndex: 1,
      explanation: "nditer handles broadcasting, multiple arrays, and write access in a single loop. But it's still slower than vectorized ops — use only when vectorization isn't possible.",
      randomize: true,
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "How many iterations does this loop execute?",
      code: "import numpy as np\na = np.zeros((3, 4, 5))\ncount = 0\nfor x in a.flat:\n    count += 1\nprint(count)",
      options: ["60", "12", "15", "3"],
      correctIndex: 0,
      explanation: ".flat iterates over ALL elements: 3 * 4 * 5 = 60 total elements.",
      randomize: false,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "If you see a for-loop over a NumPy array in production ML code, what should your first instinct be?",
      options: ["It's fine, loops are normal", "Ask whether this can be replaced with a vectorized array operation — the answer is almost always yes", "Add more loops for parallelism", "Convert to PyTorch"],
      correctIndex: 1,
      explanation: "Loops over arrays are a code smell in NumPy. Most element-wise operations have vectorized equivalents. The exceptions are genuinely sequential algorithms like custom recurrence relations.",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nfor i, row in enumerate(a):\n    print(i, row.sum())",
      options: ["0 3 and 1 12", "0 0 and 1 1", "0 3 and 1 3", "Error"],
      correctIndex: 0,
      explanation: "enumerate yields (index, row). Row 0: [0,1,2] sum=3. Row 1: [3,4,5] sum=12.",
      randomize: false,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Can you modify array elements while iterating with .flat?",
      options: ["No, .flat is read-only", "Yes, .flat supports assignment: a.flat[i] = value modifies the original array in place", "Only with nditer", "Only for 1D arrays"],
      correctIndex: 1,
      explanation: ".flat supports both reading and writing. a.flat[5] = 99 modifies the 6th element in C-order. This is useful for conditional element-wise updates.",
      randomize: true,
    },
    {
      id: "q11",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([[1, 2], [3, 4]])\nresult = []\nfor row in a:\n    result.append(row[0])\nprint(result)",
      options: ["[1, 3]", "[1, 2, 3, 4]", "[2, 4]", "[[1, 2], [3, 4]]"],
      correctIndex: 0,
      explanation: "Loop yields rows [1,2] and [3,4]. row[0] extracts first element of each: 1 and 3.",
      randomize: false,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What is the performance difference between `for x in arr` and `np.sum(arr)`?",
      options: ["Negligible", "np.sum is typically 50-100x faster because it runs a compiled C loop instead of invoking the Python interpreter per element", "for-loop is faster", "They're identical"],
      correctIndex: 1,
      explanation: "The gap widens with array size. For 1M elements, Python loop: ~100ms. np.sum: ~1ms. The ratio stays roughly constant because both are O(n) but with vastly different constants.",
      randomize: true,
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(8).reshape(2, 2, 2)\nfor slab in a:\n    print(slab.shape)",
      options: ["(2, 2) printed twice", "(2,) printed four times", "(8,) printed once", "Error"],
      correctIndex: 0,
      explanation: "Iterating a 3D array yields 2D slabs along axis 0. Shape (2,2,2) -> two iterations of shape (2,2).",
      randomize: false,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "When IS it acceptable to use a Python loop over a NumPy array?",
      options: ["Always, for readability", "When the operation is inherently sequential (each element depends on the previous), involves complex conditional logic that can't be vectorized, or the array is very small", "Never", "Only for 1D arrays"],
      correctIndex: 1,
      explanation: "Some algorithms are fundamentally sequential (recurrence relations, state machines). For tiny arrays (<100 elements), loop overhead is negligible. But for bulk numerical work, always prefer vectorization.",
      randomize: true,
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\nnames = np.array(['Alice', 'Bob', 'Charlie'])\nscores = np.array([95, 87, 92])\nfor n, s in zip(names, scores):\n    print(f'{n}:{s}', end=' ')",
      options: ["Alice:95 Bob:87 Charlie:92", "Alice Bob Charlie 95 87 92", "Error", "Alice:Bob:Charlie"],
      correctIndex: 0,
      explanation: "zip pairs corresponding elements from both arrays. Works naturally with NumPy arrays just like Python lists.",
      randomize: false,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What does np.nditer's op_flags parameter control?",
      options: ["Iteration speed", "Read/write access for each operand: 'readonly', 'writeonly', or 'readwrite'", "Array dtype", "Memory layout"],
      correctIndex: 1,
      explanation: "op_flags tells nditer how each array will be used. Without 'writeonly' or 'readwrite', you can't modify elements during iteration.",
      randomize: true,
    },
    {
      id: "q17",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nprint(sum(a.flat))",
      options: ["15", "6", "21", "Error"],
      correctIndex: 0,
      explanation: ".flat iterates all elements: 0+1+2+3+4+5 = 15. Python's sum() works on any iterable.",
      randomize: false,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Why doesn't NumPy just make iteration fast automatically?",
      options: ["It could but chose not to", "Python's for-loop protocol requires calling __next__() on a Python object for each element — there's no way to bypass the interpreter without leaving Python entirely", "GPUs don't support iteration", "It's a design limitation"],
      correctIndex: 1,
      explanation: "Python iteration is fundamentally per-object. Each yield creates a Python object, invokes the interpreter, and does type dispatch. The only way to avoid this is to push the entire loop into compiled code (vectorization).",
      randomize: true,
    },
    {
      id: "q19",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nprint(a.flat[5])",
      options: ["5", "9", "1", "6"],
      correctIndex: 0,
      explanation: ".flat indexes in C (row-major) order. Element 5 in [0,1,2,3,4,5,...] is 5.",
      randomize: false,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In deep learning training loops, why do we iterate over batches but NOT over individual samples within a batch?",
      options: ["Convention", "Batch-level iteration is necessary (data loading, shuffling); sample-level iteration defeats GPU parallelism — the whole point of batching is to process all samples simultaneously via matrix multiply", "Samples are too small", "GPU memory limits"],
      correctIndex: 1,
      explanation: "Outer loop over batches: unavoidable (data I/O). Inner loop over samples: defeats the purpose of batching. Y = X @ W + b processes the entire batch in one GPU kernel launch.",
      randomize: true,
    }
  ]}
/>
