---
title: "Rounding ufuncs"
slug: "019-ufunc-rounding"
description: "around, floor, ceil, trunc, fix — five rounding modes with different negative behavior."
track: "ufuncs"
order: 4
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["018-ufunc-arithmetic"]
---

# Rounding ufuncs

## Five Rounding Modes

<PyRunner cellId="019-cell-1" defaultCode={`import numpy as np
vals = np.array([2.7, 2.3, -2.7, -2.3, 2.5, -2.5])
print(f"{'Val':>5s} {'around':>7s} {'floor':>6s} {'ceil':>6s} {'trunc':>6s}")
for v in vals:
    print(f"{v:5.1f} {np.around(v):7.0f} {np.floor(v):6.0f} {np.ceil(v):6.0f} {np.trunc(v):6.0f}")`}/>

<Callout type="info" title="Banker's Rounding">
np.around uses round-half-to-even: 2.5→2, 3.5→4. Reduces statistical bias.
</Callout>

## around with Decimals

<PyRunner cellId="019-cell-2" defaultCode={`import numpy as np
a = np.array([3.14159, 2.71828, 1.41421])
print(f"dec=2: {np.around(a, 2)}")
print(f"dec=1: {np.around(a, 1)}")
print(f"dec=-1: {np.around(np.array([15,25,35]), -1)}")`}/>

<Quiz
  chapterSlug="019-ufunc-rounding"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "np.around(2.5) returns:",
      code: "import numpy as np\nprint(np.around(2.5))",
      options: ["2.0","3.0","2","Error"],
      correctIndex: 0,
      explanation: "Banker's rounding: 2.5→nearest even=2.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "np.floor(-2.3) returns:",
      options: ["-3.0","-2.0","-2","-3"],
      correctIndex: 0,
      explanation: "Floor rounds toward -inf: -2.3→-3.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Which rounds toward zero?",
      options: ["trunc","floor","ceil","around"],
      correctIndex: 0,
      explanation: "trunc truncates toward zero: -2.7→-2.",
      randomize: true,
    }
  ]}
/>
