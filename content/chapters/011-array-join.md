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
      type: "shape-prediction",
      prompt: "What is np.stack([np.ones(3), np.zeros(3)]).shape?",
      options: ["(2, 3)", "(6,)", "(3, 2)", "(2, 1, 3)"],
      correctIndex: 0,
      explanation: "stack creates a new axis at position 0 by default. Two (3,) arrays → (2, 3).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What's the difference between concatenate and stack?",
      options: ["concatenate joins along existing axis; stack creates a new axis", "They are identical", "stack is faster", "concatenate only works on 1D"],
      correctIndex: 0,
      explanation: "concatenate joins along an existing axis (no new dim). stack adds a brand new axis.",
      randomize: true,
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "np.vstack([np.array([1,2,3]), np.array([4,5,6])]).shape is:",
      options: ["(2, 3)", "(6,)", "(3, 2)", "(1, 6)"],
      correctIndex: 0,
      explanation: "vstack stacks vertically — each 1D array becomes a row. Result is (2, 3).",
      randomize: true,
    }
  ]}
/>
