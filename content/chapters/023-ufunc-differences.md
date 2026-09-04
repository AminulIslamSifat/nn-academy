---
title: "Difference ufuncs"
slug: "023-ufunc-differences"
description: "diff, gradient — discrete derivatives for numerical analysis and gradient checking."
track: "ufuncs"
order: 8
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["021-ufunc-summations"]
---

# Difference ufuncs

## np.diff

<PyRunner cellId="023-cell-1" defaultCode={`import numpy as np
a = np.array([1, 3, 6, 10, 15])
print(f"diff:     {np.diff(a)}")
print(f"diff n=2: {np.diff(a, n=2)}")`}/>

## np.gradient: Central Differences

<PyRunner cellId="023-cell-2" defaultCode={`import numpy as np
x = np.linspace(0, 2*np.pi, 100)
dy_num = np.gradient(np.sin(x), x)
dy_exact = np.cos(x)
print(f"Max error vs cos(x): {np.abs(dy_num-dy_exact).max():.6f}")`}/>

## Numerical Gradient Check

<PyRunner cellId="023-cell-3" defaultCode={`import numpy as np
def f(x): return x**2
def num_grad(f, x, eps=1e-7):
    return (f(x+eps) - f(x-eps)) / (2*eps)
x = 3.0
print(f"Numerical: {num_grad(f,x):.8f}")
print(f"Analytic:  {2*x:.8f}")
print(f"Match: {np.isclose(num_grad(f,x), 2*x)}")`}/>

<Quiz
  chapterSlug="023-ufunc-differences"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "np.diff([1,4,9,16]) returns:",
      code: "import numpy as np\nprint(np.diff([1,4,9,16]))",
      options: ["[3 5 7]","[1 4 9 16]","[4 9 16]","[1 2 3 4]"],
      correctIndex: 0,
      explanation: "Consecutive diffs: 4-1=3, 9-4=5, 16-9=7.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "np.gradient uses:",
      options: ["Central differences","Forward only","Backward only","Symbolic"],
      correctIndex: 0,
      explanation: "Central differences for interior, forward/backward at edges.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Output length of np.diff on array of length N?",
      options: ["N-1","N","N+1","Depends"],
      correctIndex: 0,
      explanation: "N-1 consecutive differences from N elements.",
      randomize: true,
    }
  ]}
/>
