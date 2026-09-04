---
title: "The Chain Rule"
slug: "061-chain-rule"
description: "The mathematical foundation of backpropagation. Understand how derivatives compose through nested functions and why this enables efficient gradient computation."
track: "calculus-autodiff"
order: 2
read_time: 22
code_time: 18
execution_timeout: 10
prerequisites: ["060-derivatives-and-gradients"]
---

# The Chain Rule

Backpropagation IS the chain rule applied systematically. ==If you understand the chain rule, you understand backprop.==

## Single-Variable Chain Rule

For composed functions <InlineMath latex="y = f(g(x))" />:

<BlockMath latex="\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}" />

<PyRunner
  cellId="061-cell-1"
  defaultCode={`import numpy as np

# y = sigmoid(x² + 1)
# dy/dx = sigmoid'(x²+1) · 2x

def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_deriv(out): return out * (1 - out)

x = 2.0

# Forward
g = x**2 + 1          # g(x) = x² + 1 = 5
y = sigmoid(g)         # y = σ(5)

# Backward (chain rule)
dy_dg = sigmoid_deriv(y)   # dσ/dg at g=5
dg_dx = 2 * x              # dg/dx = 2x = 4
dy_dx = dy_dg * dg_dx      # chain rule!

# Verify numerically
h = 1e-7
numerical = (sigmoid((x+h)**2+1) - sigmoid((x-h)**2+1)) / (2*h)

print(f"y = sigmoid(x² + 1) at x = {x}")
print(f"  Forward:  g = {g}, y = {y:.6f}")
print(f"  Chain rule: dy/dx = {dy_dg:.6f} × {dg_dx:.1f} = {dy_dx:.6f}")
print(f"  Numerical:  dy/dx = {numerical:.6f}")
print(f"  Match: {np.isclose(dy_dx, numerical)}")
`}
/>

## Multi-Layer Chain Rule

For a deep network <InlineMath latex="y = f_n(f_{n-1}(...f_1(x)...))" />:

<BlockMath latex="\frac{dy}{dx} = \frac{\partial f_n}{\partial f_{n-1}} \cdot \frac{\partial f_{n-1}}{\partial f_{n-2}} \cdots \frac{\partial f_1}{\partial x}" />

This product of matrices is exactly what backpropagation computes — but in reverse order, reusing intermediate results.

<PyRunner
  cellId="061-cell-2"
  defaultCode={`import numpy as np

# 3-layer network: y = σ(W3 · σ(W2 · σ(W1 · x)))
np.random.seed(42)
x = np.random.randn(3)
W1 = np.random.randn(3, 4) * 0.5
W2 = np.random.randn(4, 4) * 0.5
W3 = np.random.randn(4, 1) * 0.5

def sigmoid(z): return 1/(1+np.exp(-z))

# Forward pass (cache everything)
z1 = W1.T @ x;  a1 = sigmoid(z1)
z2 = W2.T @ a1; a2 = sigmoid(z2)
z3 = W3.T @ a2; y = sigmoid(z3)

# Backward pass (chain rule in reverse)
dy_dz3 = y * (1 - y)
dz3_da2 = W3
da2_dz2 = a2 * (1 - a2)
dz2_da1 = W2
da1_dz1 = a1 * (1 - a1)
dz1_dx = W1

# Full gradient dy/dx
dy_dx = dz1_dx @ (da1_dz1 * (dz2_da1 @ (da2_dz2 * (dz3_da2 @ dy_dz3))))

# Verify numerically
h = 1e-5
num_grad = np.zeros(3)
for i in range(3):
    xp = x.copy(); xp[i] += h
    xm = x.copy(); xm[i] -= h
    yp = sigmoid(W3.T @ sigmoid(W2.T @ sigmoid(W1.T @ xp)))
    ym = sigmoid(W3.T @ sigmoid(W2.T @ sigmoid(W1.T @ xm)))
    num_grad[i] = (yp - ym) / (2*h)

print(f"Chain rule through 3 layers:")
print(f"  Analytical gradient: {dy_dx.flatten().round(6)}")
print(f"  Numerical gradient:  {num_grad.round(6)}")
print(f"  Match: {np.allclose(dy_dx.flatten(), num_grad, atol=1e-4)}")
print(f"\n✅ This IS backpropagation — just the chain rule applied layer by layer")
`}
/>

## Why Reverse Mode?

Forward mode computes gradients one input at a time. Reverse mode (backprop) computes gradients for ALL parameters in one pass. For a network with millions of parameters and one scalar loss, reverse mode is millions of times more efficient.

> [!IMPORTANT] The Fundamental Insight
> Forward pass: compute and cache intermediates. Backward pass: multiply local gradients in reverse, reusing cached values. Total cost: ~2× forward pass. This is why deep learning is computationally feasible.

<Quiz
  chapterSlug="061-chain-rule"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is reverse-mode differentiation (backprop) preferred over forward-mode for neural networks?",
      options: ["It's easier to implement", "Networks have millions of parameters but one scalar loss; reverse mode computes all parameter gradients in one pass, while forward mode needs one pass per parameter", "Forward mode doesn't work with ReLU", "Reverse mode uses less memory"],
      correctIndex: 1,
      explanation: "Forward mode: O(parameters) passes. Reverse mode: O(1) passes (one backward pass). For a network with 100M parameters, that's 100M× speedup. This is why backprop made deep learning practical.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does backpropagation cache during the forward pass?",
      options: ["Nothing", "Intermediate activations and pre-activation values needed to compute local gradients during the backward pass", "Only the final loss", "The input data"],
      correctIndex: 1,
      explanation: "Each layer's backward pass needs the forward pass intermediates (e.g., sigmoid needs its output to compute the derivative). These are cached during forward and consumed during backward.",
      randomize: false,
    }
  ]}
/>
