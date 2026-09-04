---
title: "Sorting Arrays"
slug: "014-array-sort"
description: "Sort with sort, argsort, partition, and lexsort. In-place vs out-of-place, stable sorts, and partial sorting."
track: "numpy-foundations"
order: 11
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes"]
---

# Sorting Arrays

NumPy provides efficient sorting algorithms optimized for numerical data.

## sort vs argsort

<PyRunner
  cellId="014-cell-1"
  defaultCode={`import numpy as np

a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# sort returns sorted copy
sorted_a = np.sort(a)
print(f"Original: {a}")
print(f"Sorted:   {sorted_a}")

# argsort returns indices that would sort the array
indices = np.argsort(a)
print(f"Argsort:  {indices}")
print(f"Verify:   {a[indices]}")

# In-place sort
a.sort()
print(f"In-place: {a}")
`}
/>

## Sorting Along Axes

<PyRunner
  cellId="014-cell-2"
  defaultCode={`import numpy as np

matrix = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
print(f"Original:
{matrix}
")

# Sort each row (axis=1)
print(f"Sort rows:
{np.sort(matrix, axis=1)}
")

# Sort each column (axis=0)
print(f"Sort cols:
{np.sort(matrix, axis=0)}
")

# Flatten and sort
print(f"Sort all: {np.sort(matrix, axis=None)}")
`}
/>

## argsort: The Power Move

argsort is incredibly useful for ranking and reordering:

<PyRunner
  cellId="014-cell-3"
  defaultCode={`import numpy as np

names = np.array(["Charlie", "Alice", "Bob", "Diana"])
scores = np.array([72, 95, 88, 91])

# Sort by scores (descending)
order = np.argsort(scores)[::-1]
print("Ranking by score:")
for rank, idx in enumerate(order):
    print(f"  {rank+1}. {names[idx]}: {scores[idx]}")

# Sort multiple arrays together
print(f"
Names sorted by score: {names[order]}")
print(f"Scores sorted:         {scores[order]}")
`}
/>

## partition: Partial Sort (Top-K)

When you only need the top/bottom K elements, partition is O(n) instead of O(n log n):

<PyRunner
  cellId="014-cell-4"
  defaultCode={`import numpy as np
import time

a = np.random.randn(1_000_000)

# Full sort
start = time.perf_counter()
top10_sort = np.sort(a)[-10:]
sort_time = time.perf_counter() - start

# Partition: only guarantee top 10 are correct
start = time.perf_counter()
top10_part = np.partition(a, -10)[-10:]
part_time = time.perf_counter() - start

print(f"Full sort top-10:  {sort_time*1000:.1f} ms")
print(f"Partition top-10:  {part_time*1000:.1f} ms")
print(f"Speedup: {sort_time/part_time:.1f}x")
print(f"Same values: {np.allclose(np.sort(top10_sort), np.sort(top10_part))}")
`}
/>

<Quiz
  chapterSlug="014-array-sort"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does np.argsort([3, 1, 2]) return?",
      options: ["[1, 2, 0]", "[1, 2, 3]", "[3, 1, 2]", "[0, 1, 2]"],
      correctIndex: 0,
      explanation: "argsort returns indices: element at index 1 (value 1) is smallest, then index 2 (value 2), then index 0 (value 3).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why use np.partition instead of np.sort for top-K?",
      options: ["partition is O(n) vs sort's O(n log n)", "partition uses less memory", "partition is more accurate", "They are identical"],
      correctIndex: 0,
      explanation: "partition only partially sorts — guarantees the K-th element is in place, smaller before, larger after. O(n) average.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does np.sort([[3,1],[4,2]], axis=0) return?",
      code: "import numpy as np\nprint(np.sort([[3,1],[4,2]], axis=0))",
      options: ["[[3 1] [4 2]]", "[[1 3] [2 4]]", "[[3 4] [1 2]]", "[[4 2] [3 1]]"],
      correctIndex: 0,
      explanation: "axis=0 sorts each column: col0 [3,4]→[3,4], col1 [1,2]→[1,2]. Already sorted!",
      randomize: true,
    }
  ]}
/>
