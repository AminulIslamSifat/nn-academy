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
      type: "multiple-choice",
      prompt: "What is a boolean mask in NumPy?",
      options: ["An array of True/False values used to select elements from another array", "A function that hides certain array elements", "A dtype for storing binary data", "A way to encrypt array contents"],
      correctIndex: 0,
      explanation: "A boolean mask is an array of the same shape with True/False values. Indexing with it selects only the True positions.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([1,2,3,4,5])\nprint(a[a % 2 == 0])",
      options: ["[2 4]", "[1 3 5]", "[True False True False True]", "Error"],
      correctIndex: 0,
      explanation: "The mask a % 2 == 0 is [False, True, False, True, False]. Selecting with it gives [2, 4].",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why must you use & instead of 'and' for combining NumPy boolean conditions?",
      options: ["'and' evaluates truthiness of the whole array; & operates element-wise", "They are interchangeable", "& is simply faster", "'and' causes a syntax error"],
      correctIndex: 0,
      explanation: "Python's 'and' tries to evaluate the entire array as a single boolean (which raises ValueError). & performs element-wise logical AND.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which operator is the element-wise NOT for boolean arrays?",
      options: ["~ (tilde)", "not", "!", "neg()"],
      correctIndex: 0,
      explanation: "~ is the bitwise NOT operator that works element-wise on boolean arrays. Python's 'not' doesn't work on arrays.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([15, 3, 28, 7, 42])\nprint(a[(a > 10) & (a < 30)])",
      options: ["[15 28]", "[15 3 28 7 42]", "[3 7]", "[42]"],
      correctIndex: 0,
      explanation: "Elements >10 AND <30: 15 and 28 qualify. Each condition must be parenthesized when combined with &.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Does fancy indexing (a[[1,3,5]]) return a view or a copy?",
      options: ["Copy — modifying the result won't change the original", "View — modifying the result changes the original", "Depends on the array size", "Neither"],
      correctIndex: 0,
      explanation: "Fancy indexing always returns a copy. Basic slicing (a[1:4]) returns a view, but index arrays create independent copies.",
      randomize: true,
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(10)\na[a % 2 == 0] = -1\nprint(a)",
      options: ["[-1  1 -1  3 -1  5 -1  7 -1  9]", "[-1 -1 -1 -1 -1  5  6  7  8  9]", "[0 1 2 3 4 5 6 7 8 9]", "Error"],
      correctIndex: 0,
      explanation: "The mask selects even indices (0,2,4,6,8). Those positions are set to -1 in-place.",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does a[[0, 2, 5]] do?",
      options: ["Selects elements at indices 0, 2, and 5 (fancy indexing)", "Selects a slice from 0 to 5 with step 2", "Creates a 3D array", "Raises an error"],
      correctIndex: 0,
      explanation: "Passing a list/array of integers triggers fancy indexing, selecting exactly those positions.",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nmatrix = np.arange(12).reshape(3, 4)\nrows = [0, 1, 2]\ncols = [1, 2, 3]\nprint(matrix[rows, cols])",
      options: ["[1 6 11]", "[[1 2 3]\n [5 6 7]\n [9 10 11]]", "[0 5 10]", "Error"],
      correctIndex: 0,
      explanation: "Paired fancy indexing: matrix[0,1]=1, matrix[1,2]=6, matrix[2,3]=11 → [1, 6, 11].",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What does np.take(a, [1, 3, 4]) do compared to a[[1, 3, 4]]?",
      options: ["Same result — take is a function form of fancy indexing", "take is faster", "take returns a view while fancy indexing returns a copy", "They produce different results"],
      correctIndex: 0,
      explanation: "take and fancy indexing produce identical results. take also supports an axis parameter for multi-dimensional selection.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What does np.put(a, [0, 2, 4], [99, 88, 77]) do?",
      options: ["Sets a[0]=99, a[2]=88, a[4]=77 in-place", "Returns a new array with those values replaced", "Appends values to the array", "Extracts values at those indices"],
      correctIndex: 0,
      explanation: "put modifies the array in-place, assigning the given values at the specified indices.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([10, 20, 30, 40, 50])\nprint(a.take([1, 3, 4]))",
      options: ["[20 40 50]", "[10 30 50]", "[1 3 4]", "Error"],
      correctIndex: 0,
      explanation: "take selects elements at indices 1, 3, 4: values 20, 40, 50.",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Can boolean masks be used to assign values, not just read them?",
      options: ["Yes — a[mask] = value sets all masked positions", "No — masks are read-only", "Only with np.put", "Only for 1D arrays"],
      correctIndex: 0,
      explanation: "Boolean indexing works on both sides of assignment. a[a < 0] = 0 clips negatives to zero in-place.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What shape must a boolean mask have to index an array?",
      options: ["The same shape as the array (or broadcastable to it)", "Always 1D", "Always 2D", "Any shape works"],
      correctIndex: 0,
      explanation: "The mask must match the array's shape or be broadcastable to it. Mismatched shapes raise IndexError.",
      randomize: true,
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([5, 3, 8, 1, 9])\nmask = (a > 2) & (a < 8)\nprint(mask.sum())",
      options: ["3", "2", "5", "0"],
      correctIndex: 0,
      explanation: "Elements >2 AND <8: 5✓, 3✓, 8✗(not <8), 1✗(not >2), 9✗(not <8). Two elements match, so sum = 2.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "How do you select all rows of a matrix where column 0 is greater than 5?",
      options: ["matrix[matrix[:, 0] > 5]", "matrix[matrix > 5]", "matrix[:, matrix[0] > 5]", "np.where(matrix[:, 0] > 5)"],
      correctIndex: 0,
      explanation: "matrix[:, 0] > 5 creates a 1D boolean mask over rows. Indexing with it selects matching rows entirely.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What happens if you use Python's 'or' instead of '|' in a boolean mask expression?",
      options: ["ValueError: truth value of an array is ambiguous", "It works but slower", "It silently gives wrong results", "SyntaxError"],
      correctIndex: 0,
      explanation: "Python's 'or' tries to evaluate the array as a single boolean, which raises ValueError because arrays don't have a single truth value.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nmatrix = np.arange(12).reshape(3, 4)\nprint(matrix.take([0, 2], axis=1))",
      options: ["[[0 2]\n [4 6]\n [8 10]]", "[[0 1]\n [8 9]]", "[[0 2]\n [4 6]]", "Error"],
      correctIndex: 0,
      explanation: "take along axis=1 selects columns 0 and 2 from every row. Row 0: [0,2], Row 1: [4,6], Row 2: [8,10].",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "In ML preprocessing, how would you normalize only positive values in an array?",
      options: ["mask = a > 0; a[mask] = a[mask] / a[mask].max()", "a / a.max()", "np.where(a > 0, a / a.max(), a)", "Both A and C work"],
      correctIndex: 0,
      explanation: "Both approaches work. Boolean mask assignment modifies in-place. np.where returns a new array. Both correctly target only positive values.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You have a predictions array and want to compute accuracy. Which expression gives the fraction correct?",
      options: ["(predictions == labels).mean()", "np.sum(predictions == labels)", "np.count_nonzero(predictions == labels)", "(predictions == labels).sum() / len(predictions.shape)"],
      correctIndex: 0,
      explanation: "== produces a boolean array. .mean() treats True as 1 and False as 0, giving the fraction of correct predictions directly.",
      randomize: true,
    }
  ]}
/>
