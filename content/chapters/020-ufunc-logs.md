---
title: "Logarithmic ufuncs"
slug: "020-ufunc-logs"
description: "log, log2, log10, log1p, exp, expm1 — essential for softmax and numerical stability."
track: "ufuncs"
order: 5
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Logarithmic ufuncs

## Basic Logs

<PyRunner cellId="020-cell-1" defaultCode={`import numpy as np
a = np.array([1, 2, 4, 8, 16, 100], dtype=float)
print(f"ln:    {np.log(a).round(4)}")
print(f"log2:  {np.log2(a).round(4)}")
print(f"log10: {np.log10(a).round(4)}")
print(f"exp(ln(a))==a: {np.allclose(np.exp(np.log(a)), a)}")`}/>

## Numerical Stability

<PyRunner cellId="020-cell-2" defaultCode={`import numpy as np
tiny = 1e-15
print(f"log(1+tiny): {np.log(1+tiny)}")   # loses precision
print(f"log1p(tiny): {np.log1p(tiny)}")     # accurate
print(f"exp(tiny)-1: {np.exp(tiny)-1}")     # loses precision
print(f"expm1(tiny): {np.expm1(tiny)}")     # accurate`}/>

## Stable Softmax

<PyRunner cellId="020-cell-3" defaultCode={`import numpy as np
logits = np.array([1000, 1001, 1002])
print(f"Naive:    {np.exp(logits)/np.exp(logits).sum()}")  # nan!
shifted = logits - logits.max()
e = np.exp(shifted)
print(f"Stable:   {e/e.sum()}")`}/>

<Callout type="danger" title="Cross-Entropy Safety">
log(0) = -inf crashes training. Always clip probabilities: np.clip(probs, 1e-12, 1.0).
</Callout>

<Quiz
  chapterSlug="020-ufunc-logs"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "np.log(0) returns:",
      code: "import numpy as np\nprint(np.log(0))",
      options: ["-inf","0","nan","Error"],
      correctIndex: 0,
      explanation: "log(0)=-infinity.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why use log1p(x) instead of log(1+x)?",
      options: ["Maintains precision for tiny x","Faster","Identical","Handles negatives"],
      correctIndex: 0,
      explanation: "When x~1e-15, 1+x rounds to 1.0. log1p computes accurately.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why subtract max in stable softmax?",
      options: ["Prevents exp overflow, same ratios","Faster","Changes distribution","Optional"],
      correctIndex: 0,
      explanation: "exp(x-max)/sum = exp(x)/sum. Same result, no overflow.",
      randomize: true,
    }
  ]}
/>
