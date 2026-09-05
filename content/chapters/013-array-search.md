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
      type: "multiple-choice",
      prompt: "What does np.where(condition) with a single argument return?",
      options: ["A tuple of arrays containing indices where condition is True", "The values where condition is True", "A boolean array", "The count of True elements"],
      correctIndex: 0,
      explanation: "Single-argument where returns indices (as a tuple of arrays, one per dimension) where the condition holds.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does this output?",
      code: "import numpy as np\nprint(np.where(np.array([1,0,3,0,5]) > 0, 1, -1))",
      options: ["[ 1 -1  1 -1  1]", "[0, 2, 4]", "[1, 3, 5]", "Error"],
      correctIndex: 0,
      explanation: "Three-argument where: where condition is True pick 1, otherwise pick -1. Positions 0,2,4 have values >0.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In the three-argument form np.where(cond, x, y), what happens when cond is False?",
      options: ["The corresponding element from y is selected", "The corresponding element from x is selected", "It returns 0", "It raises an error"],
      correctIndex: 0,
      explanation: "where(cond, x, y) picks from x where cond is True and from y where cond is False.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "np.searchsorted([1,3,5,7], 4) returns:",
      options: ["2", "1", "3", "4"],
      correctIndex: 0,
      explanation: "4 would be inserted at index 2 to maintain sorted order: [1, 3, 4, 5, 7].",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does np.count_nonzero([[0,1],[2,0]]) return?",
      options: ["2", "4", "1", "3"],
      correctIndex: 0,
      explanation: "Two non-zero elements: 1 and 2.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([10, 25, 3, 47, 8, 32])\nindices = np.where(a > 20)\nprint(indices[0])",
      options: ["[1 3 5]", "[25 47 32]", "[0 2 4]", "[True False False True False True]"],
      correctIndex: 0,
      explanation: "Elements >20 are at indices 1 (25), 3 (47), and 5 (32). where returns a tuple; [0] extracts the index array.",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is the difference between np.where and np.argwhere?",
      options: ["where returns a tuple of index arrays; argwhere returns an (N, ndim) array of coordinates", "They are identical", "argwhere is faster", "where only works on 1D arrays"],
      correctIndex: 0,
      explanation: "where(cond) returns a tuple (one array per axis). argwhere(cond) returns a 2D array where each row is a coordinate.",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does np.nonzero return?",
      options: ["A tuple of arrays, one per dimension, containing indices of non-zero elements", "A flat array of non-zero values", "The count of non-zero elements", "A boolean mask"],
      correctIndex: 0,
      explanation: "nonzero is equivalent to where(array != 0). It returns a tuple of index arrays for each dimension.",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nsorted_arr = np.array([1, 3, 5, 7, 9, 11])\nprint(np.searchsorted(sorted_arr, [0, 4, 8, 12]))",
      options: ["[0 2 4 6]", "[1 2 3 5]", "[0 1 3 5]", "Error"],
      correctIndex: 0,
      explanation: "0→index 0, 4→index 2 (between 3 and 5), 8→index 4 (between 7 and 9), 12→index 6 (past end).",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What is the difference between side='left' and side='right' in searchsorted?",
      options: ["left inserts before existing equal values; right inserts after them", "left searches from the start; right searches from the end", "They produce identical results", "right is faster"],
      correctIndex: 0,
      explanation: "For value 5 in [1,3,5,7]: left returns index 2 (before the 5), right returns index 3 (after the 5).",
      randomize: true,
    },
    {
      id: "q11",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(10)\nextracted = np.extract(a % 3 == 0, a)\nprint(extracted)",
      options: ["[0 3 6 9]", "[1 2 4 5 7 8]", "[0 1 2 3]", "Error"],
      correctIndex: 0,
      explanation: "extract picks elements where the condition is True. Multiples of 3 in range(10): 0, 3, 6, 9.",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Does searchsorted require the input array to be sorted?",
      options: ["Yes — it uses binary search and gives wrong results on unsorted arrays", "No — it sorts internally", "Only for side='right'", "Only for 1D arrays"],
      correctIndex: 0,
      explanation: "searchsorted assumes sorted input. On unsorted data, the returned indices are meaningless.",
      randomize: true,
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([[0, 1, 0], [2, 0, 3]])\ncoords = np.argwhere(a > 0)\nprint(coords.shape)",
      options: ["(3, 2)", "(2, 3)", "(3,)", "(6,)"],
      correctIndex: 0,
      explanation: "Three elements are >0 (at positions (0,1), (1,0), (1,2)). argwhere returns shape (num_matches, ndim) = (3, 2).",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "How does np.place differ from np.where's three-argument form?",
      options: ["place modifies the array in-place; where returns a new array", "place is faster", "where modifies in-place; place returns new", "They are identical"],
      correctIndex: 0,
      explanation: "place(arr, mask, vals) writes vals into arr where mask is True, modifying arr directly. where always returns a new array.",
      randomize: true,
    },
    {
      id: "q15",
      type: "shape-prediction",
      prompt: "a has shape (4, 5). You call rows, cols = np.where(a > 0). What is len(rows)?",
      options: ["Equal to the number of elements > 0 in a", "Always 4", "Always 5", "Equal to np.count_nonzero(a) + 1"],
      correctIndex: 0,
      explanation: "rows and cols each contain one entry per matching element. Their length equals the count of True positions.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "You want to clip negative values in an array to zero. Which expression works?",
      options: ["np.where(a < 0, 0, a)", "np.where(a < 0, a, 0)", "np.extract(a < 0, 0)", "np.searchsorted(a, 0)"],
      correctIndex: 0,
      explanation: "where(a < 0, 0, a): where negative, use 0; otherwise keep original value. This is ReLU.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is the time complexity of np.searchsorted on a sorted array of size N?",
      options: ["O(log N) — binary search", "O(N) — linear scan", "O(1) — direct lookup", "O(N log N) — sort then search"],
      correctIndex: 0,
      explanation: "searchsorted uses binary search, which halves the search space each step → O(log N).",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([5, 3, 8, 1, 9])\nmask = a > 4\nprint(np.count_nonzero(mask))",
      options: ["3", "2", "5", "4"],
      correctIndex: 0,
      explanation: "Elements >4: 5, 8, 9 → three True values in the mask.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Can np.where's three-argument form work with arrays of different shapes?",
      options: ["Yes — broadcasting rules apply to cond, x, and y", "No — all three must have identical shapes", "Only if x and y are scalars", "Only for 1D arrays"],
      correctIndex: 0,
      explanation: "Broadcasting applies: cond, x, and y are broadcast together. A scalar x or y broadcasts to match cond's shape.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In neural networks, np.where is commonly used to implement which operation?",
      options: ["ReLU activation (max(0, x)) via where(x > 0, x, 0)", "Matrix multiplication", "Weight initialization", "Gradient descent updates"],
      correctIndex: 0,
      explanation: "ReLU zeroes out negatives. np.where(x > 0, x, 0) is a vectorized way to implement it without Python loops.",
      randomize: true,
    }
  ]}
/>
