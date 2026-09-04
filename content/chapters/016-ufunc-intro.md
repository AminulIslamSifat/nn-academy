---
title: "Universal Functions (ufunc) Intro"
slug: "016-ufunc-intro"
description: "What ufuncs are, why they're fast, and how NumPy's element-wise engine works."
track: "ufuncs"
order: 1
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["003-vectorization"]
---

# Universal Functions (ufunc) Intro

A **ufunc** operates on ndarrays element-by-element in compiled C loops.

## What Makes a ufunc Special

<PyRunner cellId="016-cell-1" defaultCode={`import numpy as np, time
a = np.arange(1_000_000, dtype=np.float64)

start = time.perf_counter()
r1 = [x**2 for x in a]
t1 = time.perf_counter() - start

start = time.perf_counter()
r2 = np.square(a)
t2 = time.perf_counter() - start

print(f"Python loop: {t1*1000:.1f} ms")
print(f"np.square:   {t2*1000:.3f} ms")
print(f"Speedup:     {t1/t2:.0f}x")`}/>

## Inspecting ufuncs

<PyRunner cellId="016-cell-2" defaultCode={`import numpy as np
print(f"np.add.nin:      {np.add.nin}")
print(f"np.add.nout:     {np.add.nout}")
print(f"np.add.identity: {np.add.identity}")
ufuncs = [n for n in dir(np) if isinstance(getattr(np,n,None), np.ufunc)]
print(f"Total ufuncs: {len(ufuncs)}")`}/>

## The out Parameter

<PyRunner cellId="016-cell-3" defaultCode={`import numpy as np
a = np.arange(5, dtype=np.float64)
result = np.empty_like(a)
np.multiply(a, 2, out=result)
print(f"Zero-alloc result: {result}")
np.add(a, 10, out=a)
print(f"In-place add: {a}")`}/>

<Quiz
  chapterSlug="016-ufunc-intro"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is np.add.identity?",
      options: ["0","1","None","-1"],
      correctIndex: 0,
      explanation: "Identity for addition is 0: x+0=x.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why are ufuncs faster than Python loops?",
      options: ["Pre-compiled C loops without per-element Python overhead","They use GPU","They skip errors","Less memory"],
      correctIndex: 0,
      explanation: "ufuncs run tight C loops over contiguous memory.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does np.add.nin return?",
      code: "import numpy as np\nprint(np.add.nin)",
      options: ["2","1","0","Error"],
      correctIndex: 0,
      explanation: "nin=number of inputs. add takes 2.",
      randomize: true,
    }
  ]}
/>
