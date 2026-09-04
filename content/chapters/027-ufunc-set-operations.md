---
title: "Set Operation ufuncs"
slug: "027-ufunc-set-operations"
description: "unique, intersect1d, union1d, setdiff1d, isin — set theory on arrays."
track: "ufuncs"
order: 12
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["014-array-sort"]
---

# Set Operation ufuncs

## unique

<PyRunner cellId="027-cell-1" defaultCode={`import numpy as np
a = np.array([3, 1, 2, 3, 1, 4, 2, 5, 3])
print(f"unique:       {np.unique(a)}")
print(f"with counts:  {np.unique(a, return_counts=True)}")
vals, inv = np.unique(a, return_inverse=True)
print(f"reconstruct:  {vals[inv]}")
print(f"match: {np.array_equal(vals[inv], a)}")`}/>

## Set Operations

<PyRunner cellId="027-cell-2" defaultCode={`import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
print(f"intersect: {np.intersect1d(a, b)}")
print(f"union:     {np.union1d(a, b)}")
print(f"setdiff:   {np.setdiff1d(a, b)}")
print(f"setxor:    {np.setxor1d(a, b)}")`}/>

## Membership Testing

<PyRunner cellId="027-cell-3" defaultCode={`import numpy as np
a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
test = np.array([2, 4, 9, 1])
mask = np.isin(a, test)
print(f"isin mask: {mask}")
print(f"matching:  {a[mask]}")
print(f"not in:    {a[~mask]}")`}/>

<Quiz
  chapterSlug="027-ufunc-set-operations"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "np.intersect1d([1,2,3],[2,3,4]) returns:",
      code: "import numpy as np\nprint(np.intersect1d([1,2,3],[2,3,4]))",
      options: ["[2 3]","[1 2 3 4]","[1 4]","[2]"],
      correctIndex: 0,
      explanation: "Intersection: elements in both.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "np.unique(a, return_inverse=True) gives:",
      options: ["Indices to reconstruct original","Counts","First occurrence","Nothing"],
      correctIndex: 0,
      explanation: "inverse indices: unique[inverse]==original.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "np.setdiff1d([1,2,3,4],[3,4,5]) returns:",
      code: "import numpy as np\nprint(np.setdiff1d([1,2,3,4],[3,4,5]))",
      options: ["[1 2]","[3 4]","[5]","[1 2 5]"],
      correctIndex: 0,
      explanation: "In first but not second: [1,2].",
      randomize: true,
    }
  ]}
/>
