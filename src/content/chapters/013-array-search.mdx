---
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
print(f"
Conditional: {result}")

# 2D example
matrix = np.random.randint(0, 10, (3, 4))
print(f"
Matrix:
{matrix}")
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
print(f"argwhere(a > 0):
{coords}")

# nonzero returns tuple of arrays (one per axis)
rows, cols = np.nonzero(a)
print(f"
nonzero rows: {rows}")
print(f"nonzero cols: {cols}")

# Count non-zero elements
print(f"
Count non-zero: {np.count_nonzero(a)}")
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
print(f"
Insert [0,4,8,12] at: {indices}")

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
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "np.searchsorted([1,3,5,7], 4) returns:",
      options: ["2", "1", "3", "4"],
      correctIndex: 0,
      explanation: "4 would be inserted at index 2 to maintain sorted order: [1,3,4,5,7].",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does np.count_nonzero([[0,1],[2,0]]) return?",
      options: ["2", "4", "1", "3"],
      correctIndex: 0,
      explanation: "Two non-zero elements: 1 and 2.",
      randomize: true,
    }
  ]}
/>
