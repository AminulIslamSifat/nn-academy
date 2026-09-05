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
      type: "multiple-choice",
      prompt: "What is the key difference between basic slicing and fancy indexing in NumPy?",
      options: ["Basic slicing is faster", "Basic slicing returns a view (shared memory); fancy indexing always returns a copy (independent data)", "Fancy indexing only works on 1D arrays", "There is no difference"],
      correctIndex: 1,
      explanation: "Basic slicing (a[1:3]) shares memory with the original. Fancy indexing (a[[0,2,4]]) allocates new memory. Modifying a view changes the original; modifying a fancy-indexed result does not.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nprint(a[:, 1])",
      options: ["[1 4]", "[3 4 5]", "[0 3]", "[1 2]"],
      correctIndex: 0,
      explanation: "a[:, 1] selects column index 1 from all rows. Row 0→1, Row 1→4. Result: [1, 4].",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Does a[1:3] return a view or a copy of array a?",
      options: ["View", "Copy", "Depends on dtype", "Depends on array size"],
      correctIndex: 0,
      explanation: "Basic slicing always returns a view in NumPy. The data is shared with the original array.",
      randomize: false,
    },
    {
      id: "q4",
      type: "shape-prediction",
      prompt: "What is the shape of mat[1:3, 1:3] where mat has shape (4, 5)?",
      options: ["(2, 2)", "(3, 3)", "(2, 3)", "(1, 1)"],
      correctIndex: 0,
      explanation: "Rows 1 and 2 (2 elements), columns 1 and 2 (2 elements) → shape (2, 2).",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(10)\nb = a[3:7]\nb[0] = 99\nprint(a[3])",
      options: ["99", "3", "0", "Error"],
      correctIndex: 0,
      explanation: "b is a view of a. Modifying b[0] modifies a[3] because they share memory.",
      randomize: false,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "How do you create an independent copy of a slice so modifications don't affect the original?",
      options: ["Use a[1:3].copy()", "Use a[1:3][:]", "Slices are always independent", "Use np.view(a[1:3])"],
      correctIndex: 0,
      explanation: ".copy() explicitly allocates new memory and copies data. Without it, slices share memory with the source.",
      randomize: true,
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nprint(a[::2, ::2])",
      options: ["[[0 2]\n [8 10]]", "[[0 1]\n [4 5]]", "[[0 3]\n [8 11]]", "[[0 2 4]\n [8 10 12]]"],
      correctIndex: 0,
      explanation: "::2 takes every other row (0,2) and every other column (0,2). Elements: a[0,0]=0, a[0,2]=2, a[2,0]=8, a[2,2]=10.",
      randomize: false,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does boolean masking return: a view or a copy?",
      options: ["A view", "A copy — boolean indexing always creates a new array", "Depends on the mask", "A reference"],
      correctIndex: 1,
      explanation: "Boolean indexing (a[a > 0]) returns a 1D copy of matching elements. Unlike basic slicing, it cannot be represented as a strided view.",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([10, 20, 30, 40, 50])\nmask = a > 25\nprint(a[mask])",
      options: ["[30 40 50]", "[False False True True True]", "[10 20]", "[25 30 40 50]"],
      correctIndex: 0,
      explanation: "Boolean mask selects elements where condition is True. 30, 40, 50 are all > 25.",
      randomize: false,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What does np.where(condition, x, y) do?",
      options: ["Returns indices where condition is True", "Element-wise selection: picks from x where True, from y where False", "Filters array x", "Replaces values in place"],
      correctIndex: 1,
      explanation: "np.where acts as vectorized if-else. np.where(a>0, a, 0) implements ReLU: keep positive values, replace negatives with 0.",
      randomize: true,
    },
    {
      id: "q11",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.arange(9).reshape(3, 3)\nprint(a[np.array([0, 2]), np.array([1, 2])])",
      options: ["[1 8]", "[[0 1]\n [6 7]]", "[1 7]", "Error"],
      correctIndex: 0,
      explanation: "Fancy indexing with two arrays selects paired elements: a[0,1]=1 and a[2,2]=8. Result: [1, 8].",
      randomize: false,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why can't you use a list of booleans directly for indexing in some contexts?",
      options: ["Lists are slower", "Python lists aren't NumPy arrays; convert with np.array(mask) to ensure proper boolean indexing behavior", "It always works", "NumPy doesn't support boolean indexing"],
      correctIndex: 1,
      explanation: "While Python lists of bools often work, explicit np.array conversion ensures consistent behavior and avoids ambiguity with integer indexing.",
      randomize: true,
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(5)\nprint(a[-2:])",
      options: ["[3 4]", "[0 1]", "[2 3 4]", "[4 3]"],
      correctIndex: 0,
      explanation: "Negative indexing counts from the end. -2: means from second-to-last to end: [3, 4].",
      randomize: false,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What does a[::-1] do to a 1D array?",
      options: ["Selects every other element", "Reverses the array", "Returns the first element", "Creates a copy"],
      correctIndex: 1,
      explanation: "Step -1 traverses the array backwards, producing a reversed view.",
      randomize: true,
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nprint(a[1:, :2].shape)",
      options: ["(2, 2)", "(1, 2)", "(3, 2)", "(2, 4)"],
      correctIndex: 0,
      explanation: "1: selects rows 1,2 (2 rows). :2 selects columns 0,1 (2 cols). Shape: (2, 2).",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "When using fancy indexing with duplicate indices like a[[0,0,1]], what happens?",
      options: ["Error", "Duplicates are allowed; each occurrence produces a separate copy of that element in the output", "Duplicates are removed", "Only the last occurrence is kept"],
      correctIndex: 1,
      explanation: "Fancy indexing allows duplicates. a[[0,0,1]] returns [a[0], a[0], a[1]] — three elements with a[0] appearing twice.",
      randomize: true,
    },
    {
      id: "q17",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([[1,2],[3,4],[5,6]])\nprint(a[a[:,0] > 2])",
      options: ["[[3 4]\n [5 6]]", "[3 5]", "[[1 2]]", "Error"],
      correctIndex: 0,
      explanation: "a[:,0] > 2 gives [False, True, True]. Boolean indexing selects rows 1 and 2: [[3,4],[5,6]].",
      randomize: false,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "What is the performance implication of fancy indexing vs basic slicing?",
      options: ["No difference", "Fancy indexing copies data (slower, more memory); basic slicing creates views (faster, no extra memory)", "Basic slicing is slower", "Fancy indexing uses less memory"],
      correctIndex: 1,
      explanation: "Views are zero-cost references to existing memory. Copies allocate new memory and transfer data. For large arrays in loops, this difference is significant.",
      randomize: true,
    },
    {
      id: "q19",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nprint(a.flat[4])",
      options: ["4", "3", "5", "Error"],
      correctIndex: 0,
      explanation: "a.flat provides a 1D iterator over the array in row-major order. Index 4 in [0,1,2,3,4,5] is 4.",
      randomize: false,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In deep learning, why is understanding view vs copy critical?",
      options: ["It affects print formatting", "Accidentally modifying training data through a view corrupts the dataset; unintended copies waste memory during batch processing", "Views are deprecated", "Copies are always safer"],
      correctIndex: 1,
      explanation: "Data augmentation applied to a view modifies the original dataset. Large unnecessary copies during batching can exhaust GPU memory. Understanding the distinction prevents subtle bugs.",
      randomize: true,
    }
  ]}
/>
