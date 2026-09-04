---
title: "The Linear Layer"
slug: "005-linear-layer"
description: "Build the fundamental building block of neural networks: y = xW + b. Understand weight matrices, bias vectors, and batch processing."
track: "nn-primitives"
order: 1
read_time: 18
code_time: 15
execution_timeout: 10
prerequisites: ["003-vectorization", "004-broadcasting"]
---

# The Linear Layer

Every neural network, no matter how complex, is built from linear transformations. The linear (or "dense"/"fully connected") layer computes:

<BlockMath latex="y = xW + b" />

where <InlineMath latex="x" /> is the input, <InlineMath latex="W" /> is the weight matrix, and <InlineMath latex="b" /> is the bias vector.

## Single Sample

For one input vector of size <InlineMath latex="n" /> mapping to <InlineMath latex="m" /> outputs:
- <InlineMath latex="x" />: shape <InlineMath latex="(n,)" />
- <InlineMath latex="W" />: shape <InlineMath latex="(n, m)" />
- <InlineMath latex="b" />: shape <InlineMath latex="(m,)" />
- <InlineMath latex="y" />: shape <InlineMath latex="(m,)" />

```python
import numpy as np

def linear_forward(x, W, b):
    """Single sample forward pass."""
    return x @ W + b
```

<PyRunner
  cellId="005-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)

# 4 features → 3 outputs
n_in, n_out = 4, 3
x = np.random.randn(n_in)
W = np.random.randn(n_in, n_out) * 0.1  # small init
b = np.zeros(n_out)

y = x @ W + b

print(f"x: {x.shape} → W: {W.shape} + b: {b.shape}")
print(f"y: {y.shape}")
print(f"y = {y.round(4)}")
`}
/>

## Batched Forward Pass

In practice, we process batches. For a batch of <InlineMath latex="B" /> samples:
- <InlineMath latex="X" />: shape <InlineMath latex="(B, n)" />
- <InlineMath latex="W" />: shape <InlineMath latex="(n, m)" />
- <InlineMath latex="b" />: shape <InlineMath latex="(m,)" /> ← broadcasts over batch!
- <InlineMath latex="Y" />: shape <InlineMath latex="(B, m)" />

```python
def linear_forward_batch(X, W, b):
    """Batched forward pass."""
    return X @ W + b  # b broadcasts from (m,) to (B, m)
```

<PyRunner
  cellId="005-cell-2"
  defaultCode={`import numpy as np

np.random.seed(0)

batch_size = 8
n_in, n_hidden = 784, 256  # like MNIST → hidden layer

X = np.random.randn(batch_size, n_in)
W = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)  # He init
b = np.zeros(n_hidden)

Y = X @ W + b

print(f"X: ({batch_size}, {n_in})")
print(f"W: ({n_in}, {n_hidden})")
print(f"Y: {Y.shape}")
print(f"Y mean: {Y.mean():.6f} (should be ≈ 0)")
print(f"Y std: {Y.std():.4f}")
`}
/>

## Weight Initialization Matters

Random weights that are too large or too small cause training to fail. The standard approach (He initialization):

<BlockMath latex="W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)" />

<Callout type="info" title="Why He initialization?">
It keeps the variance of activations roughly constant across layers, preventing gradients from vanishing or exploding in deep networks.
</Callout>

## Building a Reusable Layer

```python
class LinearLayer:
    def __init__(self, n_in: int, n_out: int):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
    
    def forward(self, X):
        return X @ self.W + self.b
```

<PyRunner
  cellId="005-cell-3"
  defaultCode={`import numpy as np

class LinearLayer:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
    
    def forward(self, X):
        return X @ self.W + self.b

# Stack two layers
np.random.seed(42)
layer1 = LinearLayer(4, 8)
layer2 = LinearLayer(8, 2)

x = np.random.randn(3, 4)  # batch of 3
h = layer1.forward(x)
y = layer2.forward(h)

print(f"Input: {x.shape}")
print(f"Hidden: {h.shape}")
print(f"Output: {y.shape}")
print(f"Output values:\n{y.round(4)}")
`}
/>

<Quiz
  chapterSlug="005-linear-layer"
  questions={[
    {
      id: "q1",
      type: "shape-prediction",
      prompt: "A linear layer maps from 512 features to 128 outputs. What is the shape of W?",
      options: ["(512, 128)", "(128, 512)", "(512,)", "(128,)"],
      correctIndex: 0,
      explanation: "W maps input features (columns of X) to output features. Shape is (n_in, n_out) = (512, 128).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does the bias b of shape (128,) work when added to output of shape (32, 128)?",
      options: ["Broadcasting: (128,) aligns with the last dim", "It gets reshaped automatically", "NumPy pads it with zeros", "It only works if batch_size=1"],
      correctIndex: 0,
      explanation: "Broadcasting aligns shapes from the right. (32, 128) + (128,) → the bias is treated as (1, 128) and broadcast across all 32 rows.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is the shape of Y = X @ W if X is (16, 64) and W is (64, 10)?",
      code: "X = np.random.randn(16, 64)\nW = np.random.randn(64, 10)\nY = X @ W",
      options: ["(16, 10)", "(64, 64)", "(16, 64)", "(10, 16)"],
      correctIndex: 0,
      explanation: "(16, 64) @ (64, 10) → inner dimensions match (64=64), result is (16, 10).",
      randomize: true,
    }
  ]}
/>
