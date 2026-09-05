---
title: "Vectorization: Kill Your Loops"
slug: "003-vectorization"
description: "Replace Python loops with array operations. Understand why vectorized code is 100x faster and how to think in batches."
track: "vectorization"
order: 1
read_time: 15
code_time: 12
execution_timeout: 10
prerequisites: ["001-arrays-and-shapes", "002-indexing-and-slicing"]
---

# Vectorization: Kill Your Loops

The single most important mindset shift in NumPy: **never loop over elements when an array operation can do it for you.**

## The Problem with Loops

Python loops are slow because each iteration involves type checking, object allocation, and interpreter overhead. NumPy pushes the loop down to optimized C/Fortran.

```python
import numpy as np

# ❌ Slow: Python loop
result = np.zeros(1000000)
for i in range(1000000):
    result[i] = np.sin(i) * np.cos(i)

# ✅ Fast: Vectorized
result = np.sin(np.arange(1000000)) * np.cos(np.arange(1000000))
```

<PyRunner
  cellId="003-cell-1"
  timeout={10000}
  defaultCode={`import numpy as np
import time

n = 1_000_000
x = np.arange(n, dtype=np.float64)

# Vectorized
start = time.perf_counter()
result = np.sin(x) * np.cos(x)
vec_time = time.perf_counter() - start

print(f"Vectorized: {vec_time*1000:.2f}ms")
print(f"Result shape: {result.shape}")
print(f"First 5: {result[:5]}")
`}
/>

## Element-wise Operations

All arithmetic operators work element-wise on arrays:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b    # [5, 7, 9]
a * b    # [4, 10, 18]  (element-wise, NOT dot product!)
a ** 2   # [1, 4, 9]
np.exp(a)  # [e, e², e³]
```

<Callout type="danger" title="* is NOT matrix multiplication">
`a * b` multiplies element-by-element. For dot products, use `a @ b` or `np.dot(a, b)`. This trips up everyone at first.
</Callout>

## The Dot Product

The fundamental operation in neural networks:

```python
weights = np.random.randn(784, 128)  # layer weights
x = np.random.randn(784)             # single input

output = x @ weights  # shape (128,) — one forward pass
```

<PyRunner
  cellId="003-cell-2"
  defaultCode={`import numpy as np

# Simulate a single linear layer
np.random.seed(42)
weights = np.random.randn(4, 3)  # 4 inputs → 3 outputs
bias = np.zeros(3)
x = np.array([1.0, 2.0, 3.0, 4.0])

output = x @ weights + bias
print("Input shape:", x.shape)
print("Weights shape:", weights.shape)
print("Output shape:", output.shape)
print("Output:", output.round(4))
`}
/>

## Batched Operations

Instead of looping over samples, stack them into a matrix:

```python
# ❌ Loop over batch
for i in range(batch_size):
    outputs[i] = X[i] @ W + b

# ✅ Single matrix multiply
outputs = X @ W + b  # X: (batch, features), W: (features, out)
```

<PyRunner
  cellId="003-cell-3"
  timeout={10000}
  defaultCode={`import numpy as np
import time

np.random.seed(0)
batch_size = 256
features = 512
out_features = 128

X = np.random.randn(batch_size, features)
W = np.random.randn(features, out_features)
b = np.random.randn(out_features)

# Vectorized forward pass
start = time.perf_counter()
Y = X @ W + b
elapsed = time.perf_counter() - start

print(f"Batch forward pass: {elapsed*1000:.3f}ms")
print(f"X: {X.shape} @ W: {W.shape} → Y: {Y.shape}")
`}
/>

<Visualizer mode="matmul" defaultShapeA={[4, 3]} defaultShapeB={[3, 2]} />

<Quiz
  chapterSlug="003-vectorization"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is vectorized NumPy code faster than an equivalent Python for-loop?",
      options: ["Python is inherently slow at everything", "NumPy pushes the loop into optimized compiled C/Fortran code, eliminating per-element interpreter overhead like type checking and object allocation", "NumPy automatically uses the GPU", "Vectorized code uses less RAM"],
      correctIndex: 1,
      explanation: "Every iteration of a Python loop goes through the interpreter: type dispatch, reference counting, memory allocation. NumPy's inner loops run in compiled C with SIMD instructions on contiguous memory — no interpreter involved.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does the * operator do when applied to two NumPy arrays?",
      options: ["Matrix multiplication", "Element-wise multiplication", "Dot product", "Outer product"],
      correctIndex: 1,
      explanation: "In NumPy, * is ALWAYS element-wise. This is different from math notation where * often means matrix multiply. Use @ or np.dot() for matrix multiplication.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([2, 3, 4])\nprint(a ** 2)",
      options: ["[4 9 16]", "[2 3 4 2 3 4]", "8", "Error"],
      correctIndex: 0,
      explanation: "** is element-wise exponentiation. Each element is squared independently: 2²=4, 3²=9, 4²=16.",
      randomize: true,
    },
    {
      id: "q4",
      type: "shape-prediction",
      prompt: "If X has shape (32, 784) and W has shape (784, 256), what is the shape of X @ W?",
      options: ["(32, 256)", "(784, 784)", "(32, 784)", "(256, 32)"],
      correctIndex: 0,
      explanation: "Matrix multiply rule: (m, k) @ (k, n) → (m, n). Inner dimensions must match (784=784). Result takes outer dimensions: (32, 256).",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nprint(a * b)",
      options: ["[4 10 18]", "32", "[5 7 9]", "Error"],
      correctIndex: 0,
      explanation: "Element-wise multiplication: 1×4=4, 2×5=10, 3×6=18. NOT a dot product.",
      randomize: false,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nprint(np.dot(a, b))",
      options: ["32", "[4 10 18]", "21", "Error"],
      correctIndex: 0,
      explanation: "np.dot on 1D arrays computes the dot product (sum of element-wise products): 1×4 + 2×5 + 3×6 = 4+10+18 = 32.",
      randomize: false,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "Which expression correctly computes a batched forward pass for a linear layer?",
      options: ["for i in range(N): Y[i] = X[i] @ W + b", "Y = X @ W + b where X is (batch, features)", "Y = X * W + b", "Y = np.dot(X, W) * b"],
      correctIndex: 1,
      explanation: "X @ W + b processes the entire batch in one matrix multiply. The bias broadcasts across all samples. No loop needed.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the shape of the result?",
      code: "import numpy as np\nX = np.ones((3, 4))\nw = np.ones(4)\nprint((X @ w).shape)",
      options: ["(3,)", "(4,)", "(3, 4)", "()"],
      correctIndex: 0,
      explanation: "(3,4) @ (4,) → (3,). Matrix times vector produces a vector with one entry per row of the matrix.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why should you never write `for i in range(len(a)): result[i] = np.sin(a[i])`?",
      options: ["It's ugly", "Each iteration calls the Python interpreter and invokes np.sin on a scalar; np.sin(a) processes the entire array in one optimized C call", "Loops don't work with NumPy arrays", "It produces wrong results"],
      correctIndex: 1,
      explanation: "Calling np.sin on individual scalars N times has N× interpreter overhead. Calling it once on the whole array runs a single optimized loop in C. Typical speedup: 50-100×.",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nb = np.arange(6).reshape(3, 2)\nprint((a @ b).shape)",
      options: ["(2, 2)", "(3, 3)", "(2, 3)", "(6,)"],
      correctIndex: 0,
      explanation: "(2,3) @ (3,2) → (2,2). Inner dimensions match (3=3), outer dimensions form the result shape.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What happens if you try (3, 4) @ (3, 2)?",
      options: ["Returns (3, 2)", "ValueError: inner dimensions don't match (4 ≠ 3)", "Returns (4, 2)", "Broadcasts automatically"],
      correctIndex: 1,
      explanation: "Matrix multiply requires the last dimension of the left operand to equal the first dimension of the right. 4 ≠ 3, so it fails. No broadcasting in matmul.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([[1, 2], [3, 4]])\nprint(a.sum())",
      options: ["10", "[4 6]", "[3 7]", "[[1 2]\n [3 4]]"],
      correctIndex: 0,
      explanation: "sum() with no axis argument sums ALL elements into a scalar: 1+2+3+4 = 10.",
      randomize: false,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "In neural networks, why is the forward pass written as Y = X @ W + b instead of looping over samples?",
      options: ["It's shorter to write", "A single matrix multiply maps directly to highly optimized BLAS routines (cuBLAS on GPU); a Python loop would be 100-1000× slower", "Loops can't compute dot products", "NumPy doesn't support loops"],
      correctIndex: 1,
      explanation: "BLAS libraries are hand-tuned for specific hardware with cache-aware blocking, SIMD, and GPU kernels. No Python loop can approach this performance.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.eye(3)\nb = np.array([1.0, 2.0, 3.0])\nprint(a @ b)",
      options: ["[1. 2. 3.]", "[6.]", "[[1 0 0]\n [0 2 0]\n [0 0 3]]", "Error"],
      correctIndex: 0,
      explanation: "Identity matrix times any vector returns the same vector unchanged. I @ x = x. This is the defining property of the identity matrix.",
      randomize: false,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What typical speedup do you get by replacing a Python loop with a vectorized NumPy operation?",
      options: ["2-3×", "10-100× for numerical operations on large arrays", "No measurable difference", "1000× always"],
      correctIndex: 1,
      explanation: "Speedup depends on array size and operation complexity. Small arrays: 10×. Large arrays with complex math: 100× or more. The gap widens because vectorized code also benefits from better cache utilization.",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nprint(a + 10)",
      options: ["[11 12 13]", "[1 2 3 10]", "Error", "10"],
      correctIndex: 0,
      explanation: "Scalar-to-array operations broadcast automatically. 10 is added to every element: [1+10, 2+10, 3+10] = [11, 12, 13].",
      randomize: false,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What does 'thinking in batches' mean for deep learning code?",
      options: ["Process data in fixed-size chunks", "Stack multiple samples into matrices and compute all outputs simultaneously with one matrix multiply, instead of processing samples one at a time", "Use batch normalization", "Store data in batches on disk"],
      correctIndex: 1,
      explanation: "Batching turns N separate forward passes into ONE matrix multiply. This maximizes GPU utilization and amortizes kernel launch overhead. It's why we use mini-batches instead of single samples.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What is the shape of the result?",
      code: "import numpy as np\nX = np.random.randn(64, 128)\nW = np.random.randn(128, 10)\nb = np.random.randn(10)\nY = X @ W + b\nprint(Y.shape)",
      options: ["(64, 10)", "(128, 10)", "(64, 128)", "(10,)"],
      correctIndex: 0,
      explanation: "This is a complete batched linear layer. (64,128) @ (128,10) → (64,10). Bias (10,) broadcasts across the batch dimension via broadcasting rules.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Why does np.exp(a) vastly outperform [math.exp(x) for x in a]?",
      options: ["Shorter syntax only", "np.exp processes the entire array in a single compiled C call with SIMD; the list comprehension invokes Python's interpreter and math.exp separately for each element", "math.exp doesn't accept floats", "They perform identically"],
      correctIndex: 1,
      explanation: "Vectorized ufuncs like np.exp execute a tight C loop over contiguous memory. List comprehensions pay interpreter overhead per element. For 1M elements, this is ~100× slower.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You have X with shape (batch, features) and want to normalize each sample independently. Which approach is correct and vectorized?",
      options: ["for i in range(batch): X[i] = X[i] / X[i].max()", "X / X.max(axis=1, keepdims=True)", "X / X.max()", "X / np.linalg.norm(X)"],
      correctIndex: 1,
      explanation: "max(axis=1, keepdims=True) computes per-sample max with shape (batch, 1), which broadcasts correctly against (batch, features). No loop needed. Without keepdims, shape (batch,) won't broadcast properly against (batch, features).",
      randomize: true,
    }
  ]}
/>
