---
title: "Activation Functions"
slug: "028-activation-functions"
description: "Why linear layers alone can't learn anything useful. Implement sigmoid, ReLU, tanh, and softmax from scratch in NumPy and understand when to use each."
track: "nn-beginner"
order: 1
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["005-linear-layer"]
---

# Activation Functions

A stack of linear layers is still just one linear layer. No matter how many you stack:

<BlockMath latex="y = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'" />

The whole thing collapses into a single affine transform. ==Non-linearity is what gives depth its power.== Activation functions break linearity so networks can learn curves, boundaries, and complex patterns.

## The Sigmoid

The classic squashing function. Maps any real number to <InlineMath latex="(0, 1)" />:

<BlockMath latex="\sigma(z) = \frac{1}{1 + e^{-z}}" />

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
```

<PyRunner
  cellId="028-cell-1"
  defaultCode={`import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.linspace(-10, 10, 9)
out = sigmoid(z)

for zi, oi in zip(z, out):
    bar = '█' * int(oi * 30)
    print(f"z={zi:+6.1f} → σ(z)={oi:.4f} |{bar}")
`}
/>

<Callout type="warning" title="Numerical Stability">
For large negative `z`, `np.exp(-z)` overflows. Use `np.where` or `scipy.special.expit` in production. We'll handle this properly in later chapters.
</Callout>

### Sigmoid Derivative

The derivative has an elegant form in terms of the output itself:

<BlockMath latex="\sigma'(z) = \sigma(z)(1 - \sigma(z))" />

This means if you've already computed the forward pass, the gradient is essentially free:

```python
def sigmoid_derivative(sigmoid_output):
    """Takes the OUTPUT of sigmoid, not the input."""
    return sigmoid_output * (1.0 - sigmoid_output)
```

<PyRunner
  cellId="028-cell-2"
  defaultCode={`import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(out):
    return out * (1.0 - out)

z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
s = sigmoid(z)
ds = sigmoid_deriv(s)

print(" z       σ(z)     σ'(z)")
print("─" * 32)
for zi, si, dsi in zip(z, s, ds):
    print(f"{zi:+6.1f}   {si:.4f}   {dsi:.4f}")

print(f"\nMax gradient at z=0: {sigmoid_deriv(sigmoid(np.array([0.0])))[0]:.4f}")
print("Gradients vanish as |z| grows → vanishing gradient problem!")
`}
/>

> [!IMPORTANT] Why Sigmoid Fell Out of Favor
> The maximum gradient is only 0.25 (at z=0). For saturated neurons (z far from 0), gradients approach zero. In deep networks, these tiny gradients multiply layer after layer during backprop, effectively killing learning. This is the **vanishing gradient problem**.

## ReLU: The Modern Default

Rectified Linear Unit. Dead simple, brutally effective:

<BlockMath latex="\text{ReLU}(z) = \max(0, z)" />

```python
def relu(z):
    return np.maximum(0, z)
```

<PyRunner
  cellId="028-cell-3"
  defaultCode={`import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

z = np.linspace(-5, 5, 11)
r = relu(z)
dr = relu_deriv(z)

print(" z      ReLU(z)  dReLU/dz")
print("─" * 34)
for zi, ri, dri in zip(z, r, dr):
    print(f"{zi:+6.1f}   {ri:7.1f}    {dri:.0f}")

print("\n✅ Gradient is exactly 1 for all positive inputs")
print("❌ Gradient is exactly 0 for all negative inputs (dead neurons)")
`}
/>

### Why ReLU Works So Well

1. **No vanishing gradient** for positive inputs — gradient is always 1
2. **Computationally cheap** — just a comparison, no exponentials
3. **Sparse activation** — roughly half the neurons fire, which acts as implicit regularization
4. **Empirically superior** — trains faster and deeper than sigmoid/tanh in practice

<Callout type="info" title="Dead Neuron Problem">
If a ReLU neuron's weights push its input permanently negative, it outputs zero forever. The gradient is zero, so it never updates. Solutions: smaller learning rates, He initialization (covered in ch.005), or Leaky ReLU.
</Callout>

### Leaky ReLU

Fixes dead neurons by allowing a small gradient when negative:

<BlockMath latex="\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \ \alpha z & z \leq 0 \end{cases}" />

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
```

<PyRunner
  cellId="028-cell-4"
  defaultCode={`import numpy as np

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
lr = leaky_relu(z)

print(" z      LeakyReLU(z)")
print("─" * 26)
for zi, lri in zip(z, lr):
    print(f"{zi:+6.1f}   {lri:+8.4f}")

print("\nNegative inputs get a tiny slope (α=0.01)")
print("Neurons can recover even if pushed negative")
`}
/>

## Tanh

Zero-centered sigmoid. Maps to <InlineMath latex="(-1, 1)" />:

<BlockMath latex="\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1" />

```python
def tanh(z):
    return np.tanh(z)  # NumPy has a stable built-in

def tanh_derivative(tanh_output):
    """Like sigmoid, derivative expressed via output."""
    return 1.0 - tanh_output ** 2
```

<PyRunner
  cellId="028-cell-5"
  defaultCode={`import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.linspace(-4, 4, 9)
s = sigmoid(z)
t = np.tanh(z)

print(" z      σ(z)     tanh(z)")
print("─" * 34)
for zi, si, ti in zip(z, s, t):
    print(f"{zi:+6.1f}   {si:.4f}   {ti:+.4f}")

print("\ntanh is zero-centered → better gradient flow than sigmoid")
print("But still saturates at extremes → same vanishing gradient issue")
`}
/>

## Softmax: For Classification Output

Converts a vector of raw scores (logits) into a probability distribution:

<BlockMath latex="\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}" />

```python
def softmax(z):
    # Subtract max for numerical stability
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

<Callout type="warning" title="Always Subtract the Max">
Without subtracting `max(z)`, `np.exp(z)` overflows for large values. Since softmax is shift-invariant (<InlineMath latex="\text{softmax}(z + c) = \text{softmax}(z)" />), this doesn't change the result but prevents NaN.
</Callout>

<PyRunner
  cellId="028-cell-6"
  defaultCode={`import numpy as np

def softmax(z):
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# Raw logits from a network (10 classes, like MNIST)
np.random.seed(42)
logits = np.random.randn(10) * 2
probs = softmax(logits)

print("Class  Logit    Probability")
print("─" * 36)
for i, (l, p) in enumerate(zip(logits, probs)):
    bar = '█' * int(p * 40)
    print(f"  {i}    {l:+6.2f}    {p:.4f}  {bar}")

print(f"\nSum of probabilities: {probs.sum():.6f} ✅")
print(f"Predicted class: {np.argmax(probs)}")
`}
/>

### Batched Softmax

For a batch of samples, apply along the last axis:

```python
def softmax_batch(Z):
    """Z shape: (batch_size, num_classes)"""
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

<PyRunner
  cellId="028-cell-7"
  defaultCode={`import numpy as np

def softmax_batch(Z):
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

np.random.seed(0)
logits = np.random.randn(4, 5) * 2  # batch of 4, 5 classes
probs = softmax_batch(logits)

print("Batch softmax — each row sums to 1:")
for i in range(4):
    print(f"  Sample {i}: sum={probs[i].sum():.6f}, pred={np.argmax(probs[i])}")

print(f"\nAll rows sum to 1: {np.allclose(probs.sum(axis=1), 1.0)} ✅")
`}
/>

## Choosing Your Activation

| Activation | Range | Best For | Watch Out For |
|-----------|-------|----------|---------------|
| ReLU | <InlineMath latex="[0, ∞)" /> | Hidden layers (default) | Dead neurons |
| Leaky ReLU | <InlineMath latex="(-∞, ∞)" /> | Hidden layers (safer) | Slightly slower |
| Sigmoid | <InlineMath latex="(0, 1)" /> | Binary classification output | Vanishing gradients |
| Tanh | <InlineMath latex="(-1, 1)" /> | RNN hidden states | Vanishing gradients |
| Softmax | <InlineMath latex="(0, 1)^K" />, sums to 1 | Multi-class output | Always use numerically stable version |

> [!NOTE] Rule of Thumb
> **Hidden layers**: Start with ReLU. Switch to Leaky ReLU if you see dead neurons. **Output layer**: Softmax for multi-class, sigmoid for binary, nothing (linear) for regression.

<Quiz
  chapterSlug="028-activation-functions"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why can't a deep network with only linear activations learn complex functions?",
      options: [
        "It trains too slowly",
        "Multiple linear layers collapse into a single linear layer",
        "Linear activations cause overflow",
        "Linear activations only work for binary classification"
      ],
      correctIndex: 1,
      explanation: "Matrix multiplication is associative: W₂(W₁x + b₁) + b₂ = W'x + b'. No matter how many layers you stack, the entire network is equivalent to a single affine transformation.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the maximum value of the sigmoid derivative σ'(z)?",
      options: ["1.0", "0.5", "0.25", "0.1"],
      correctIndex: 2,
      explanation: "σ'(z) = σ(z)(1 - σ(z)). This is maximized when σ(z) = 0.5 (at z=0), giving 0.5 × 0.5 = 0.25. This small maximum is why sigmoid causes vanishing gradients in deep networks.",
      randomize: true,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What does this code print?",
      code: "import numpy as np\nz = np.array([1000.0, -1000.0, 0.0])\nshifted = z - np.max(z)\nexp_z = np.exp(shifted)\nprint(softmax := exp_z / exp_z.sum())",
      options: [
        "[1. 0. 0.]",
        "[nan nan nan]",
        "[0.5 0. 0.5]",
        "OverflowError"
      ],
      correctIndex: 0,
      explanation: "Subtracting max(z)=1000 gives [0, -2000, -1000]. exp([0, -2000, -1000]) ≈ [1, 0, 0]. Normalizing gives [1, 0, 0]. Without the max subtraction, exp(1000) would overflow.",
      randomize: false,
    }
  ]}
/>
