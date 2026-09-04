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
      type: "shape-prediction",
      prompt: "If X has shape (32, 784) and W has shape (784, 256), what is the shape of X @ W?",
      options: ["(32, 256)", "(784, 784)", "(32, 784)", "(256, 32)"],
      correctIndex: 0,
      explanation: "Matrix multiply: (m, k) @ (k, n) → (m, n). So (32, 784) @ (784, 256) → (32, 256).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a * b compute for two 1D arrays of the same length?",
      options: ["Element-wise multiplication", "Dot product", "Cross product", "Outer product"],
      correctIndex: 0,
      explanation: "The * operator is always element-wise in NumPy. For dot product, use @ or np.dot().",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([2, 3, 4])\nprint(a ** 2)",
      options: ["[4 9 16]", "[2 3 4 2 3 4]", "8", "Error"],
      correctIndex: 0,
      explanation: "** is element-wise exponentiation. Each element is squared: 2²=4, 3²=9, 4²=16.",
      randomize: true,
    }
  ]}
/>
