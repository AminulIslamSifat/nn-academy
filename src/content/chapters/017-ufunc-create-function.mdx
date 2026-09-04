---
title: "Creating Custom ufuncs"
slug: "017-ufunc-create-function"
description: "Build your own ufuncs with frompyfunc and vectorize. Understand the tradeoffs."
track: "ufuncs"
order: 2
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Creating Custom ufuncs

## np.frompyfunc

<PyRunner cellId="017-cell-1" defaultCode={`import numpy as np
my_ufunc = np.frompyfunc(lambda x: x**2 + 2*x + 1, 1, 1)
a = np.array([1, 2, 3, 4, 5])
result = my_ufunc(a)
print(f"Result: {result}")
print(f"dtype: {result.dtype}")  # object!
print(f"As float: {result.astype(np.float64)}")`}/>

## np.vectorize

<PyRunner cellId="017-cell-2" defaultCode={`import numpy as np
@np.vectorize
def classify(x):
    if x < 0: return -1
    elif x < 5: return 0
    else: return 1
print(classify(np.array([-3, 0, 2, 5, 8])))`}/>

<Callout type="warning" title="Not Actually Faster">
vectorize and frompyfunc still call Python per element. For real speed, use native ufuncs or numba.
</Callout>

## Speed Comparison

<PyRunner cellId="017-cell-3" defaultCode={`import numpy as np, time
n = 100_000
a = np.random.randn(n)
vec = np.vectorize(lambda x: x**2 if x > 0 else 0)
for name, fn in [("loop", lambda a: [x**2 if x>0 else 0 for x in a]),
                 ("vectorize", vec),
                 ("native", lambda a: np.where(a>0, a**2, 0))]:
    t = time.perf_counter(); fn(a); print(f"{name:12s}: {(time.perf_counter()-t)*1000:.1f} ms")`}/>

<Quiz
  chapterSlug="017-ufunc-create-function"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Does np.vectorize make functions faster?",
      options: ["No — convenience wrapper only","Yes significantly","Only small arrays","Only with numba"],
      correctIndex: 0,
      explanation: "vectorize is a fancy for-loop, not a speed optimization.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What dtype does frompyfunc return?",
      code: "import numpy as np\nf=np.frompyfunc(lambda x:x*2,1,1)\nprint(f(np.array([1,2,3])).dtype)",
      options: ["object","float64","int64","Error"],
      correctIndex: 0,
      explanation: "frompyfunc always returns object dtype.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "For real performance with custom ops use:",
      options: ["Native ufuncs or numba","np.vectorize","frompyfunc","List comprehensions"],
      correctIndex: 0,
      explanation: "Native ufuncs run in C. Numba JIT-compiles to machine code.",
      randomize: true,
    }
  ]}
/>
