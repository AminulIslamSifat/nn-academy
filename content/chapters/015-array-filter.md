---
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
print(f"
Between 10 and 30: {a[(a > 10) & (a < 30)]}")
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
print(f"
Random: {b.round(2)}")
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
print(f"
Matrix:
{matrix}")
print(f"matrix[[0,1,2], [1,2,3]] = {matrix[rows, cols]}")

# Reorder rows
print(f"
Reversed rows:
{matrix[[2, 1, 0]]}")
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
print(f"
take along axis=1: {matrix.take([0, 2], axis=1)}")
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
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why must you use & instead of 'and' for combining NumPy boolean conditions?",
      options: ["'and' operates on whole arrays as single booleans; & is element-wise", "They are interchangeable", "& is faster", "'and' causes a syntax error"],
      correctIndex: 0,
      explanation: "Python's 'and' returns one of its operands (truthiness of whole array). & performs element-wise bitwise AND.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Does fancy indexing (a[[1,3,5]]) return a view or copy?",
      options: ["Copy", "View", "Depends on the array", "Neither"],
      correctIndex: 0,
      explanation: "Fancy indexing always returns a copy, unlike basic slicing which returns a view.",
      randomize: true,
    }
  ]}
/>
