---
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
print("
axis=1:")
print(np.concatenate([a, b], axis=1))

# 1D arrays
c = np.array([1, 2, 3])
d = np.array([4, 5, 6])
print(f"
1D concat: {np.concatenate([c, d])}")
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
print(f"vstack:
{np.vstack([a, b])}")

# dstack = depth (axis=2)
print(f"dstack shape: {np.dstack([a, b]).shape}")

# column_stack treats 1D as columns
print(f"column_stack:
{np.column_stack([a, b])}")
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
      type: "multiple-choice",
      prompt: "What does np.concatenate do?",
      options: ["Joins arrays along an existing axis", "Creates a new axis and joins arrays", "Splits an array into parts", "Sorts arrays before joining"],
      correctIndex: 0,
      explanation: "concatenate joins arrays along an existing axis without adding any new dimensions.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the key difference between concatenate and stack?",
      options: ["concatenate joins along an existing axis; stack creates a new axis", "They are identical", "stack is faster than concatenate", "concatenate only works on 1D arrays"],
      correctIndex: 0,
      explanation: "concatenate merges along an existing dimension. stack adds a brand-new dimension and places arrays along it.",
      randomize: true,
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "What is np.concatenate([np.array([1,2,3]), np.array([4,5,6])]).shape?",
      options: ["(6,)", "(2, 3)", "(3, 2)", "(1, 6)"],
      correctIndex: 0,
      explanation: "Concatenating two 1D arrays along axis 0 (default) produces a single longer 1D array of shape (6,).",
      randomize: true,
    },
    {
      id: "q4",
      type: "shape-prediction",
      prompt: "What is np.stack([np.ones(3), np.zeros(3)]).shape?",
      options: ["(2, 3)", "(6,)", "(3, 2)", "(2, 1, 3)"],
      correctIndex: 0,
      explanation: "stack creates a new axis at position 0 by default. Two (3,) arrays become rows in a (2, 3) result.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([[1, 2], [3, 4]])\nb = np.array([[5, 6], [7, 8]])\nprint(np.concatenate([a, b], axis=0))",
      options: ["[[1 2]\n [3 4]\n [5 6]\n [7 8]]", "[[1 2 5 6]\n [3 4 7 8]]", "Error", "[[1 2]\n [5 6]]"],
      correctIndex: 0,
      explanation: "axis=0 concatenates along rows. The second array's rows are appended below the first array's rows.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([[1, 2], [3, 4]])\nb = np.array([[5, 6], [7, 8]])\nprint(np.concatenate([a, b], axis=1))",
      options: ["[[1 2 5 6]\n [3 4 7 8]]", "[[1 2]\n [3 4]\n [5 6]\n [7 8]]", "Error", "[[5 6]\n [7 8]]"],
      correctIndex: 0,
      explanation: "axis=1 concatenates along columns. Each row gets wider: [1,2] + [5,6] → [1,2,5,6].",
      randomize: true,
    },
    {
      id: "q7",
      type: "shape-prediction",
      prompt: "np.vstack([np.array([1,2,3]), np.array([4,5,6])]).shape is:",
      options: ["(2, 3)", "(6,)", "(3, 2)", "(1, 6)"],
      correctIndex: 0,
      explanation: "vstack treats each 1D array as a row and stacks them vertically, producing shape (2, 3).",
      randomize: true,
    },
    {
      id: "q8",
      type: "shape-prediction",
      prompt: "np.hstack([np.array([1,2,3]), np.array([4,5,6])]).shape is:",
      options: ["(6,)", "(2, 3)", "(3, 2)", "(1, 6)"],
      correctIndex: 0,
      explanation: "For 1D arrays, hstack concatenates along axis 0 (the only axis), producing (6,). For 2D it would join columns.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "When would you use np.column_stack instead of np.hstack?",
      options: ["When you want 1D arrays treated as columns in a 2D result", "When working with 3D arrays", "column_stack is always faster", "There is no difference"],
      correctIndex: 0,
      explanation: "column_stack converts 1D arrays to column vectors before stacking, giving a 2D result. hstack on 1D arrays just concatenates them into a longer 1D array.",
      randomize: true,
    },
    {
      id: "q10",
      type: "shape-prediction",
      prompt: "What is np.dstack([np.ones((2,3)), np.zeros((2,3))]).shape?",
      options: ["(2, 3, 2)", "(4, 3)", "(2, 6)", "(2, 3, 1)"],
      correctIndex: 0,
      explanation: "dstack stacks along axis 2 (depth). Two (2,3) arrays stacked along depth → (2, 3, 2).",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "You have feature1 with shape (100, 3) and feature2 with shape (100, 5). How do you combine them into (100, 8)?",
      options: ["np.concatenate([feature1, feature2], axis=1)", "np.concatenate([feature1, feature2], axis=0)", "np.stack([feature1, feature2])", "np.vstack([feature1, feature2])"],
      correctIndex: 0,
      explanation: "To merge features (columns), concatenate along axis=1. Both arrays must have the same number of rows (100).",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What happens if you try np.concatenate([a, b], axis=1) where a has shape (3, 4) and b has shape (2, 4)?",
      options: ["ValueError: shapes don't match on non-concatenation axes", "It works and produces (5, 4)", "It broadcasts b to match a", "It truncates a to (2, 4)"],
      correctIndex: 0,
      explanation: "When concatenating along axis=1, all other axes must match. Here axis 0 differs (3 vs 2), so NumPy raises ValueError.",
      randomize: true,
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nprint(np.column_stack([a, b]))",
      options: ["[[1 4]\n [2 5]\n [3 6]]", "[1 2 3 4 5 6]", "[[1 2 3]\n [4 5 6]]", "Error"],
      correctIndex: 0,
      explanation: "column_stack treats each 1D array as a column. a becomes column [1,2,3]ᵀ and b becomes [4,5,6]ᵀ, producing a (3, 2) matrix.",
      randomize: true,
    },
    {
      id: "q14",
      type: "shape-prediction",
      prompt: "What is np.stack([np.ones((2,3)), np.zeros((2,3))], axis=1).shape?",
      options: ["(2, 2, 3)", "(2, 3, 2)", "(4, 3)", "(2, 6)"],
      correctIndex: 0,
      explanation: "stack with axis=1 inserts the new dimension at position 1. Two (2,3) arrays → (2, 2, 3).",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "In ML, why do we often add a bias column of ones using hstack?",
      options: ["So the linear layer can learn an offset without requiring the input to encode it", "To make the matrix square", "Bias columns speed up matrix multiplication", "It prevents overfitting"],
      correctIndex: 0,
      explanation: "A bias term lets the model shift its decision boundary. By prepending a column of 1s, the weight vector's first element acts as the bias.",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([[1, 2], [3, 4]])\nb = np.array([[5, 6]])\ntry:\n    print(np.concatenate([a, b], axis=0))\nexcept Exception as e:\n    print(type(e).__name__)",
      options: ["[[1 2]\n [3 4]\n [5 6]]", "ValueError", "TypeError", "[[1 2 5]\n [3 4 6]]"],
      correctIndex: 0,
      explanation: "b has shape (1, 2) and a has shape (2, 2). Along axis=0, the column count matches (both 2), so concatenation succeeds → (3, 2).",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "How many arrays can you pass to np.concatenate?",
      options: ["Any number (as a list or tuple)", "Exactly 2", "At most 3", "Depends on the axis"],
      correctIndex: 0,
      explanation: "concatenate accepts a sequence (list/tuple) of arrays. You can join 2, 3, or 100 arrays in one call.",
      randomize: true,
    },
    {
      id: "q18",
      type: "shape-prediction",
      prompt: "You have three arrays each with shape (50, 4). What is np.stack(them, axis=0).shape?",
      options: ["(3, 50, 4)", "(150, 4)", "(50, 12)", "(50, 4, 3)"],
      correctIndex: 0,
      explanation: "stack with axis=0 creates a new leading dimension. Three (50, 4) arrays → (3, 50, 4).",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Which function would you use to combine a batch of grayscale images (each shape (28, 28)) into a single 3D tensor for processing?",
      options: ["np.stack(images, axis=0)", "np.concatenate(images, axis=0)", "np.hstack(images)", "np.column_stack(images)"],
      correctIndex: 0,
      explanation: "stack creates a new batch dimension. N images of (28, 28) → (N, 28, 28). concatenate would flatten them into one giant 2D array.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You concatenated two arrays along axis=0 and got shape (7, 4). One input had shape (3, 4). What was the other input's shape?",
      options: ["(4, 4)", "(3, 4)", "(7, 0)", "Cannot determine from this information"],
      correctIndex: 0,
      explanation: "Along axis=0, rows add up: 3 + ? = 7, so the other array had 4 rows. Columns must match (4). Shape was (4, 4).",
      randomize: true,
    }
  ]}
/>
