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
      prompt: "What does np.split do?",
      options: ["Breaks an array into multiple sub-arrays along a specified axis", "Joins multiple arrays into one", "Sorts an array and splits at median", "Reshapes an array into smaller dimensions"],
      correctIndex: 0,
      explanation: "split divides an array into sub-arrays along a given axis. It's the inverse of concatenate.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What happens when you call np.split on an array of 10 elements with n=3?",
      options: ["ValueError — 10 is not evenly divisible by 3", "Returns 3 parts of sizes 4, 3, 3", "Returns 3 parts of sizes 3, 3, 4", "Pads with zeros to make it divisible"],
      correctIndex: 0,
      explanation: "np.split requires the array to be evenly divisible. For uneven splits, use np.array_split instead.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Which function handles uneven splits without raising an error?",
      options: ["np.array_split", "np.split", "np.hsplit", "np.vsplit"],
      correctIndex: 0,
      explanation: "array_split distributes remainder elements across the first few sub-arrays, so no ValueError is raised.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What does this output?",
      code: "import numpy as np\nparts = np.split(np.arange(6), [2, 4])\nprint([list(p) for p in parts])",
      options: ["[[0, 1], [2, 3], [4, 5]]", "[[0, 1, 2], [3, 4, 5]]", "[[0, 1], [2, 3, 4, 5]]", "Error"],
      correctIndex: 0,
      explanation: "Split at indices 2 and 4 creates three slices: [0:2], [2:4], [4:6].",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "np.hsplit is equivalent to np.split with which axis?",
      options: ["axis=1", "axis=0", "axis=2", "axis=-1"],
      correctIndex: 0,
      explanation: "hsplit splits horizontally (along columns), which is axis=1 for 2D arrays.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "np.vsplit is equivalent to np.split with which axis?",
      options: ["axis=0", "axis=1", "axis=2", "axis=-1"],
      correctIndex: 0,
      explanation: "vsplit splits vertically (along rows), which is axis=0 for 2D arrays.",
      randomize: true,
    },
    {
      id: "q7",
      type: "shape-prediction",
      prompt: "If a has shape (6, 8), what is the shape of each part after np.hsplit(a, 4)?",
      options: ["(6, 2)", "(2, 8)", "(6, 4)", "(3, 8)"],
      correctIndex: 0,
      explanation: "hsplit splits along axis=1. 8 columns ÷ 4 parts = 2 columns each. Rows stay at 6 → (6, 2).",
      randomize: true,
    },
    {
      id: "q8",
      type: "shape-prediction",
      prompt: "If a has shape (9, 4), what is the shape of each part after np.vsplit(a, 3)?",
      options: ["(3, 4)", "(9, 1)", "(3, 1)", "(9, 4)"],
      correctIndex: 0,
      explanation: "vsplit splits along axis=0. 9 rows ÷ 3 parts = 3 rows each. Columns stay at 4 → (3, 4).",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What are the lengths of the parts?",
      code: "import numpy as np\nparts = np.array_split(np.arange(10), 3)\nprint([len(p) for p in parts])",
      options: ["[4, 3, 3]", "[3, 3, 4]", "[3, 4, 3]", "ValueError"],
      correctIndex: 0,
      explanation: "10 ÷ 3 = 3 remainder 1. array_split gives the extra element to the first part: sizes 4, 3, 3.",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "When splitting at specific indices like np.split(a, [3, 7]), how many sub-arrays are produced?",
      options: ["3 (one more than the number of split points)", "2", "4", "Depends on array length"],
      correctIndex: 0,
      explanation: "N split indices create N+1 sub-arrays: [0:3], [3:7], [7:end].",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "In a train/test split, if you have 100 samples and want 80% train, what index do you split at?",
      options: ["80", "20", "0.8", "100"],
      correctIndex: 0,
      explanation: "Split at index 80: samples [0:80] go to train (80 samples), [80:100] go to test (20 samples).",
      randomize: true,
    },
    {
      id: "q12",
      type: "shape-prediction",
      prompt: "b has shape (2, 3, 4). After np.dsplit(b, 2), what is each part's shape?",
      options: ["(2, 3, 2)", "(1, 3, 4)", "(2, 1, 4)", "(2, 3, 4)"],
      correctIndex: 0,
      explanation: "dsplit splits along axis=2 (depth). 4 ÷ 2 = 2 depth slices each → (2, 3, 2).",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Why should you shuffle data before doing a train/test split with np.split?",
      options: ["Because split takes contiguous slices — unshuffled data may have class ordering bias", "Shuffling makes split faster", "NumPy requires sorted data before splitting", "It doesn't matter"],
      correctIndex: 0,
      explanation: "np.split takes contiguous slices. If data is ordered by class, your train set might contain only certain classes. Shuffle first.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nleft, right = np.hsplit(a, 2)\nprint(left.shape, right.shape)",
      options: ["(3, 2) (3, 2)", "(2, 4) (1, 4)", "(3, 1) (3, 3)", "Error"],
      correctIndex: 0,
      explanation: "hsplit(a, 2) splits 4 columns into 2 equal halves. Each half has shape (3, 2).",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Can np.split accept a list of indices instead of an integer count?",
      options: ["Yes — it splits at those exact positions", "No — only integer counts are accepted", "Only for 1D arrays", "Only with axis=0"],
      correctIndex: 0,
      explanation: "When you pass a list like [3, 7], split cuts at those indices, producing len(indices)+1 sub-arrays.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What does np.array_split(np.arange(7), 3) produce in terms of sizes?",
      options: ["[3, 2, 2]", "[2, 2, 3]", "[3, 3, 1]", "ValueError"],
      correctIndex: 0,
      explanation: "7 ÷ 3 = 2 remainder 1. The first sub-array gets the extra element: sizes 3, 2, 2.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "After splitting, are the resulting sub-arrays views or copies of the original?",
      options: ["Views — modifying them changes the original", "Copies — they are independent", "Depends on the split function used", "Depends on whether the split is even"],
      correctIndex: 0,
      explanation: "split returns views into the original array. Changes to a sub-array will modify the original data.",
      randomize: true,
    },
    {
      id: "q18",
      type: "shape-prediction",
      prompt: "You have X with shape (200, 10). You do np.split(X, [160]). What are the shapes of the two parts?",
      options: ["(160, 10) and (40, 10)", "(160, 5) and (40, 5)", "(200, 8) and (200, 2)", "Error"],
      correctIndex: 0,
      explanation: "Splitting at index 160 along default axis=0: first 160 rows → (160, 10), remaining 40 rows → (40, 10).",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Which split function would you use to divide a batch of RGB images (shape (N, H, W, 3)) into separate channel arrays?",
      options: ["np.dsplit(images, 3) or np.split(images, 3, axis=3)", "np.hsplit(images, 3)", "np.vsplit(images, 3)", "np.split(images, 3, axis=0)"],
      correctIndex: 0,
      explanation: "Channels are along axis=3. dsplit operates on axis=2, but for 4D arrays you'd use split with axis=3. However, dsplit on the last axis of a 4D array also works if reshaped appropriately. The most direct answer is split(axis=3).",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You split an array into 5 equal parts, process each independently, then want to recombine. Which function reverses np.split?",
      options: ["np.concatenate(parts, axis=same_axis)", "np.stack(parts)", "np.merge(parts)", "np.join(parts)"],
      correctIndex: 0,
      explanation: "concatenate is the inverse of split. Pass the list of sub-arrays and the same axis to reconstruct the original.",
      randomize: true,
    }
  ]}
/>
