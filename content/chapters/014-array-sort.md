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
      prompt: "What does np.sort return?",
      options: ["A new sorted copy of the array", "The original array modified in-place", "Indices that would sort the array", "A boolean mask"],
      correctIndex: 0,
      explanation: "np.sort always returns a new sorted array. The original is unchanged. For in-place sorting, use arr.sort().",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does np.argsort([3, 1, 2]) return?",
      options: ["[1, 2, 0]", "[1, 2, 3]", "[3, 1, 2]", "[0, 1, 2]"],
      correctIndex: 0,
      explanation: "argsort returns indices: index 1 has value 1 (smallest), index 2 has value 2, index 0 has value 3 (largest).",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "How do you use argsort to get the sorted version of an array?",
      options: ["a[np.argsort(a)]", "np.argsort(a)[a]", "np.sort(np.argsort(a))", "a.argsort()"],
      correctIndex: 0,
      explanation: "argsort gives the indices; fancy-indexing with those indices reorders the array: a[argsort(a)] equals sort(a).",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([3, 1, 4, 1, 5])\na.sort()\nprint(a)",
      options: ["[1 1 3 4 5]", "[3 1 4 1 5]", "[5 4 3 1 1]", "Error"],
      correctIndex: 0,
      explanation: "arr.sort() sorts in-place, modifying the original array. The result is ascending order.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.sort([[3,1],[4,2]], axis=0))",
      options: ["[[3 1]\n [4 2]]", "[[1 3]\n [2 4]]", "[[3 4]\n [1 2]]", "[[4 2]\n [3 1]]"],
      correctIndex: 0,
      explanation: "axis=0 sorts each column independently: col0 [3,4] stays [3,4], col1 [1,2] stays [1,2]. Already sorted.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.sort([[3,1],[4,2]], axis=1))",
      options: ["[[1 3]\n [2 4]]", "[[3 1]\n [4 2]]", "[[1 2]\n [3 4]]", "[[4 3]\n [2 1]]"],
      correctIndex: 0,
      explanation: "axis=1 sorts each row independently: row0 [3,1]→[1,3], row1 [4,2]→[2,4].",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What does np.sort(matrix, axis=None) do?",
      options: ["Flattens the matrix and returns a sorted 1D array", "Sorts along axis 0", "Sorts along axis 1", "Returns None"],
      correctIndex: 0,
      explanation: "axis=None flattens the array first, then sorts all elements into a single 1D sorted result.",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "Why use np.partition instead of np.sort for finding top-K elements?",
      options: ["partition is O(n) average vs sort's O(n log n)", "partition uses less memory", "partition produces a fully sorted result", "They are identical in performance"],
      correctIndex: 0,
      explanation: "partition only guarantees the K-th element is correctly placed with smaller elements before and larger after, without fully sorting. This is O(n) on average.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "After np.partition(a, k), what is guaranteed about element at index k?",
      options: ["All elements before index k are ≤ a[k], all after are ≥ a[k]", "The array is fully sorted", "a[k] is the maximum element", "Elements before k are sorted in ascending order"],
      correctIndex: 0,
      explanation: "partition places the k-th smallest element at position k, with smaller elements before and larger after (but not sorted within those groups).",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([5, 2, 8, 1, 9, 3])\nprint(np.partition(a, -2)[-2:])",
      options: ["The two largest values (8 and 9), but possibly unsorted", "[8, 9] always sorted", "[1, 2]", "Error"],
      correctIndex: 0,
      explanation: "partition(a, -2) ensures the 2nd-from-end position holds the correct value with larger elements after. The last 2 elements are the top-2, but may not be internally sorted.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "How do you sort names by scores in descending order using argsort?",
      options: ["names[np.argsort(scores)[::-1]]", "names[np.argsort(scores)]", "np.sort(names)", "names.sort(scores)"],
      correctIndex: 0,
      explanation: "argsort(scores) gives ascending order indices. [::-1] reverses them for descending. Fancy-index names with those indices.",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Does np.sort modify the original array?",
      options: ["No — it returns a new sorted copy", "Yes — it sorts in-place", "Only for 1D arrays", "Depends on the dtype"],
      correctIndex: 0,
      explanation: "np.sort(arr) always returns a new array. Use arr.sort() for in-place modification.",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What sorting algorithm does NumPy use by default?",
      options: ["Quicksort (introsort variant)", "Merge sort", "Bubble sort", "Heap sort"],
      correctIndex: 0,
      explanation: "NumPy's default kind='quicksort' uses introsort (quicksort + heapsort fallback) for O(n log n) worst case.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "When would you choose kind='stable' in np.sort?",
      options: ["When equal elements must maintain their original relative order", "When you need faster sorting", "When sorting floats", "Never — quicksort is always better"],
      correctIndex: 0,
      explanation: "Stable sort preserves the relative order of equal elements. Important when sorting by multiple keys sequentially.",
      randomize: true,
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([3, 1, 2])\nidx = np.argsort(a)\nprint(idx)",
      options: ["[1 2 0]", "[0 1 2]", "[1 2 3]", "[3 1 2]"],
      correctIndex: 0,
      explanation: "Smallest value (1) is at index 1, next (2) at index 2, largest (3) at index 0 → [1, 2, 0].",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "You have predictions and labels. How do you find the indices of the worst 5 predictions (highest error)?",
      options: ["np.argsort(np.abs(predictions - labels))[-5:]", "np.sort(predictions)[-5:]", "np.partition(predictions, 5)", "np.argmax(predictions)"],
      correctIndex: 0,
      explanation: "Compute absolute errors, argsort gives ascending order of error, [-5:] takes the 5 highest-error indices.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is the key difference between np.sort and arr.sort()?",
      options: ["np.sort returns a new array; arr.sort() modifies in-place and returns None", "They are identical", "arr.sort() is slower", "np.sort only works on 1D arrays"],
      correctIndex: 0,
      explanation: "np.sort(arr) creates and returns a sorted copy. arr.sort() sorts the array in-place and returns None.",
      randomize: true,
    },
    {
      id: "q18",
      type: "shape-prediction",
      prompt: "matrix has shape (4, 6). What is np.sort(matrix, axis=1).shape?",
      options: ["(4, 6)", "(6, 4)", "(24,)", "(4,)"],
      correctIndex: 0,
      explanation: "Sorting along an axis doesn't change the shape. Each row is sorted independently, preserving dimensions.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "For finding the median of a large array, which is most efficient?",
      options: ["np.partition(a, len(a)//2) then read the middle element", "np.sort(a) then read the middle element", "np.median(a) which uses sort internally", "Loop through and count"],
      correctIndex: 0,
      explanation: "partition is O(n) vs sort's O(n log n). Note: np.median actually uses partition internally for efficiency.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You want to sort rows of a 2D array by the values in column 2. Which approach works?",
      options: ["matrix[matrix[:, 2].argsort()]", "np.sort(matrix, axis=2)", "matrix.sort(axis=2)", "np.argsort(matrix, axis=2)"],
      correctIndex: 0,
      explanation: "matrix[:, 2] extracts column 2, argsort gives the row ordering by that column, then fancy-index reorders the rows.",
      randomize: true,
    }
  ]}
/>
