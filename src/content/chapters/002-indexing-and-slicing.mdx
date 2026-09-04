---
title: "Indexing & Slicing"
slug: "002-indexing-and-slicing"
description: "Master array access — basic indexing, fancy indexing, boolean masks, and the critical view-vs-copy distinction."
track: "numpy-foundations"
order: 2
read_time: 14
code_time: 10
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes"]
---

# Indexing & Slicing

Accessing elements efficiently is the bread and butter of tensor manipulation. NumPy gives you multiple indexing strategies, each with different performance and memory implications.

## Basic Slicing: Views, Not Copies

Slicing returns a *view* into the original array. No data is copied — you're just looking at the same memory differently.

```python
import numpy as np

a = np.arange(10)
slice_view = a[2:7]    # shape (5,), shares memory with a
slice_view[0] = 999    # modifies a as well!
```

<PyRunner
  cellId="002-cell-1"
  defaultCode={`import numpy as np

a = np.arange(10)
print("Original:", a)

view = a[2:7]
print("Slice [2:7]:", view)

view[0] = 999
print("After modifying view:", a)  # a changed too!
print("Shares memory:", np.shares_memory(a, view))
`}
/>

<Callout type="warning" title="View vs Copy">
Modifying a view modifies the original. Use `.copy()` when you need independent data. This is the #1 source of subtle bugs in NumPy code.
</Callout>

## Multi-dimensional Indexing

For 2D+ arrays, you index each axis separated by commas:

```python
mat = np.arange(12).reshape(3, 4)

mat[0, :]     # first row      → shape (4,)
mat[:, 2]     # third column   → shape (3,)
mat[1:3, 1:3] # sub-matrix     → shape (2, 2)
```

<PyRunner
  cellId="002-cell-2"
  defaultCode={`import numpy as np

mat = np.arange(12).reshape(3, 4)
print("Matrix:")
print(mat)
print("\nRow 0:", mat[0, :])
print("Col 2:", mat[:, 2])
print("\nSub-matrix [1:3, 1:3]:")
print(mat[1:3, 1:3])
`}
/>

## Fancy Indexing: Always Copies

When you index with an array of indices, NumPy *copies* the data:

```python
indices = np.array([0, 2, 4])
selected = mat[indices]  # new array, independent copy
```

## Boolean Masking

The most powerful indexing tool for filtering:

```python
data = np.random.randn(100)
positive = data[data > 0]       # only positive values
clipped = np.where(data > 0, data, 0)  # ReLU by hand!
```

<PyRunner
  cellId="002-cell-3"
  defaultCode={`import numpy as np

data = np.random.randn(10)
print("Data:", data.round(3))

mask = data > 0
print("Mask:", mask)
print("Positives only:", data[mask])

# Manual ReLU
relu = np.where(data > 0, data, 0)
print("ReLU:", relu.round(3))
`}
/>

<Quiz
  chapterSlug="002-indexing-and-slicing"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "What is the output of the following code?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nprint(a[:, 1])",
      options: ["[1 4]", "[3 4 5]", "[0 3]", "[1 2]"],
      correctIndex: 0,
      explanation: "a[:, 1] selects column index 1 from all rows. Row 0 has [0,1,2] → element 1. Row 1 has [3,4,5] → element 4. Result: [1, 4].",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Does a[1:3] return a view or a copy of array a?",
      options: ["View", "Copy", "Depends on dtype", "Depends on array size"],
      correctIndex: 0,
      explanation: "Basic slicing always returns a view in NumPy. The data is shared. Only fancy indexing (array of indices) or .copy() creates independent copies.",
      randomize: false,
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "What is the shape of mat[1:3, 1:3] where mat has shape (4, 5)?",
      options: ["(2, 2)", "(3, 3)", "(2, 3)", "(1, 1)"],
      correctIndex: 0,
      explanation: "Rows 1 and 2 (2 elements), columns 1 and 2 (2 elements) → shape (2, 2).",
      randomize: true,
    }
  ]}
/>
