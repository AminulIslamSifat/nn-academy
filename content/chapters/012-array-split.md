---
title: "Splitting Arrays"
slug: "012-array-split"
description: "Break arrays apart with split, hsplit, vsplit, dsplit, and array_split for uneven divisions."
track: "numpy-foundations"
order: 9
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["011-array-join"]
---

# Splitting Arrays

The inverse of joining — break one array into multiple sub-arrays.

## Basic Split

<PyRunner
  cellId="012-cell-1"
  defaultCode={`import numpy as np

a = np.arange(12).reshape(3, 4)
print(f"Original:
{a}
")

# Split into 2 along axis 1 (columns)
left, right = np.split(a, 2, axis=1)
print(f"Left:
{left}")
print(f"Right:
{right}")

# Split into 3 along axis 0 (rows)
parts = np.split(a, 3, axis=0)
for i, p in enumerate(parts):
    print(f"Part {i}: {p.flatten()}")
`}
/>

## Split at Specific Indices

<PyRunner
  cellId="012-cell-2"
  defaultCode={`import numpy as np

a = np.arange(10)

# Split at indices [3, 7]
p1, p2, p3 = np.split(a, [3, 7])
print(f"[0:3]  = {p1}")
print(f"[3:7]  = {p2}")
print(f"[7:10] = {p3}")
`}
/>

## array_split: Uneven Divisions

`np.split` requires even division. `np.array_split` handles remainders:

<PyRunner
  cellId="012-cell-3"
  defaultCode={`import numpy as np

a = np.arange(10)

# 10 elements into 3 parts → uneven!
try:
    np.split(a, 3)
except ValueError as e:
    print(f"split error: {e}")

parts = np.array_split(a, 3)
for i, p in enumerate(parts):
    print(f"Part {i}: {p} (len={len(p)})")
`}
/>

## hsplit, vsplit, dsplit

<PyRunner
  cellId="012-cell-4"
  defaultCode={`import numpy as np

a = np.arange(12).reshape(3, 4)

# hsplit = split along axis 1
left, right = np.hsplit(a, 2)
print(f"hsplit: left={left.shape}, right={right.shape}")

# vsplit = split along axis 0
top, mid, bot = np.vsplit(a, 3)
print(f"vsplit: top={top.shape}, mid={mid.shape}, bot={bot.shape}")

# dsplit for 3D
b = np.arange(24).reshape(2, 3, 4)
parts = np.dsplit(b, 2)
print(f"dsplit: {[p.shape for p in parts]}")
`}
/>

## Practical: Train/Test Split

<PyRunner
  cellId="012-cell-5"
  defaultCode={`import numpy as np

# Simple train/test split without sklearn
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

split_idx = 80
X_train, X_test = np.split(X, [split_idx])
y_train, y_test = np.split(y, [split_idx])

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")
`}
/>

<Quiz
  chapterSlug="012-array-split"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What happens when you np.split an array of 10 elements into 3 parts?",
      options: ["ValueError — uneven split", "Returns 3 parts of sizes 4,3,3", "Returns 3 parts of sizes 3,3,4", "Pads with zeros"],
      correctIndex: 0,
      explanation: "np.split requires even division. Use np.array_split for uneven splits.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does np.split(np.arange(6), [2, 4]) return?",
      code: "import numpy as np\nparts = np.split(np.arange(6), [2, 4])\nprint([list(p) for p in parts])",
      options: ["[[0,1], [2,3], [4,5]]", "[[0,1,2], [3,4,5]]", "[[0,1], [2,3,4,5]]", "Error"],
      correctIndex: 0,
      explanation: "Split at indices 2 and 4: [0:2], [2:4], [4:6] → three equal parts.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "np.hsplit is equivalent to np.split with axis=?",
      options: ["axis=1", "axis=0", "axis=2", "axis=-1"],
      correctIndex: 0,
      explanation: "hsplit splits horizontally, which means along axis 1 (columns).",
      randomize: true,
    }
  ]}
/>
