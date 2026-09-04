---
title: "Summation ufuncs"
slug: "021-ufunc-summations"
description: "sum, nansum, cumsum — axis-based aggregation for loss computation and statistics."
track: "ufuncs"
order: 6
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Summation ufuncs

## sum Along Axes

<PyRunner cellId="021-cell-1" defaultCode={`import numpy as np
m = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"Total:   {m.sum()}")
print(f"axis=0:  {m.sum(axis=0)}")
print(f"axis=1:  {m.sum(axis=1)}")
print(f"keepdims: {m.sum(axis=1, keepdims=True).shape}")`}/>

## nansum

<PyRunner cellId="021-cell-2" defaultCode={`import numpy as np
data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
print(f"sum:    {np.sum(data)}")
print(f"nansum: {np.nansum(data)}")`}/>

## cumsum

<PyRunner cellId="021-cell-3" defaultCode={`import numpy as np
print(f"cumsum: {np.cumsum([1,2,3,4,5])}")
m = np.array([[1,2],[3,4],[5,6]])
print(f"cumsum axis=0:
{np.cumsum(m, axis=0)}")`}/>

## Batch MSE Loss

<PyRunner cellId="021-cell-4" defaultCode={`import numpy as np
pred = np.random.randn(32, 10)
target = np.random.randn(32, 10)
per_sample = np.sum((pred-target)**2, axis=1)
print(f"Mean batch loss: {per_sample.mean():.4f}")`}/>

<Quiz
  chapterSlug="021-ufunc-summations"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "np.sum([[1,2],[3,4]], axis=0) returns:",
      code: "import numpy as np\nprint(np.sum([[1,2],[3,4]],axis=0))",
      options: ["[4 6]","[3 7]","10","[[1 2]]"],
      correctIndex: 0,
      explanation: "axis=0 sums columns: [1+3, 2+4]=[4,6].",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "np.sum([1, np.nan, 3]) returns:",
      options: ["nan","4","0","Error"],
      correctIndex: 0,
      explanation: "NaN propagates. Use nansum to ignore.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "np.cumsum([1,2,3,4]) returns:",
      code: "import numpy as np\nprint(np.cumsum([1,2,3,4]))",
      options: ["[ 1  3  6 10]","[1 2 3 4]","10","[1 3 6]"],
      correctIndex: 0,
      explanation: "Running sum: [1, 1+2, 1+2+3, 1+2+3+4].",
      randomize: true,
    }
  ]}
/>
