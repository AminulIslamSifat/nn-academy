---
title: "Loss Functions"
slug: "029-loss-functions"
description: "How neural networks measure their mistakes. Implement MSE, cross-entropy, and softmax cross-entropy from scratch with numerically stable gradients."
track: "nn-beginner"
order: 2
read_time: 22
code_time: 18
execution_timeout: 10
prerequisites: ["028-activation-functions"]
---

# Loss Functions

A loss function answers one question: ==how wrong is the network right now?== Without it, there's no signal for learning. The loss converts predictions and targets into a single scalar that backpropagation can differentiate.

Choosing the wrong loss is like navigating with a broken compass — your network will train, but it won't arrive anywhere useful.

## Mean Squared Error (Regression)

The default for continuous outputs:

<BlockMath latex="\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2" />

```python
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    """d(MSE)/d(y_pred) — what backprop needs."""
    return 2.0 * (y_pred - y_true) / y_true.size
```

<PyRunner
  cellId="029-cell-1"
  defaultCode={`import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return 2.0 * (y_pred - y_true) / y_true.size

np.random.seed(42)
y_true = np.array([3.0, -1.5, 2.0, 0.5])
y_pred = np.array([2.8, -1.0, 2.5, 0.0])

loss = mse(y_true, y_pred)
grad = mse_grad(y_true, y_pred)

print(f"MSE Loss: {loss:.4f}")
print(f"Gradient w.r.t. predictions: {grad}")
print(f"\nGradient points TOWARD the true values:")
for yt, yp, g in zip(y_true, y_pred, grad):
    direction = "↓ decrease" if g > 0 else "↑ increase"
    print(f"  pred={yp:+.1f} → true={yt:+.1f}, grad={g:+.4f} ({direction})")
`}
/>

<Callout type="info" title="Why Square the Error?">
Squaring makes the loss always positive and penalizes large errors more than small ones. It also makes the gradient smooth and proportional to the error — bigger mistakes produce bigger corrective signals.
</Callout>

## Binary Cross-Entropy

For binary classification (sigmoid output). Measures the divergence between predicted probability and true label:

<BlockMath latex="\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]" />

```python
def binary_cross_entropy(y_true, y_pred, eps=1e-7):
    """Numerically stable BCE."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

def bce_gradient(y_true, y_pred, eps=1e-7):
    """d(BCE)/d(y_pred) before sigmoid derivative."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
```

<PyRunner
  cellId="029-cell-2"
  defaultCode={`import numpy as np

def bce(y_true, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

# Perfect predictions vs terrible ones
y_true = np.array([1, 1, 0, 0])
good_pred = np.array([0.95, 0.90, 0.05, 0.10])
bad_pred = np.array([0.30, 0.40, 0.60, 0.70])

print(f"Good predictions BCE: {bce(y_true, good_pred):.4f}")
print(f"Bad predictions BCE:  {bce(y_true, bad_pred):.4f}")
print(f"\nRatio: {bce(y_true, bad_pred) / bce(y_true, good_pred):.1f}x worse")

# Edge case: what happens at exactly 0 or 1?
try:
    print(f"Without clipping: {bce(np.array([1]), np.array([0.0]))}")
except:
    print("Without clipping: log(0) = -inf → NaN loss! 💥")

print(f"With clipping:    {bce(np.array([1]), np.array([0.0])):.4f} ✅")
`}
/>

> [!IMPORTANT] Always Clip Predictions
> `log(0)` is negative infinity. Even a single prediction of exactly 0.0 or 1.0 produces NaN loss. Always clip to `[ε, 1-ε]` where ε ≈ 1e-7.

## Categorical Cross-Entropy (Multi-Class)

For multi-class classification with softmax output. This is ==the loss you'll use for MNIST==:

<BlockMath latex="\text{CCE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})" />

where <InlineMath latex="y" /> is one-hot encoded and <InlineMath latex="\hat{y}" /> is the softmax output.

```python
def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-7):
    """y_true_onehot: (N, C), y_pred: (N, C) from softmax."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))
```

<PyRunner
  cellId="029-cell-3"
  defaultCode={`import numpy as np

def softmax(z):
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cce(y_true_oh, y_pred, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_oh * np.log(y_pred), axis=1))

# Batch of 3 samples, 5 classes
np.random.seed(42)
logits = np.random.randn(3, 5)
probs = softmax(logits)

# True classes: 2, 0, 4
y_true_oh = np.zeros((3, 5))
y_true_oh[0, 2] = 1
y_true_oh[1, 0] = 1
y_true_oh[2, 4] = 1

loss = cce(y_true_oh, probs)
print(f"Predicted probabilities for true class:")
for i in range(3):
    true_class = np.argmax(y_true_oh[i])
    p = probs[i, true_class]
    print(f"  Sample {i}: class {true_class}, p={p:.4f}, -log(p)={-np.log(p):.4f}")

print(f"\nCCE Loss: {loss:.4f}")
print(f"Perfect prediction would give: {-np.log(1.0):.4f}")
print(f"Random guessing (5 classes):   {-np.log(0.2):.4f}")
`}
/>

### The Beautiful Shortcut: Softmax + CCE Gradient

When softmax and cross-entropy are paired, the gradient simplifies dramatically:

<BlockMath latex="\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i" />

That's it. ==Predicted probability minus true label.== No chain rule through softmax and log separately. This is why they're almost always used together.

```python
def softmax_cce_gradient(y_true_onehot, softmax_output):
    """Combined gradient: just subtraction!"""
    return (softmax_output - y_true_onehot) / y_true_onehot.shape[0]
```

<PyRunner
  cellId="029-cell-4"
  defaultCode={`import numpy as np

def softmax(z):
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

np.random.seed(42)
logits = np.array([[2.0, 1.0, 0.1],
                    [0.5, 3.0, 0.2]])
probs = softmax(logits)

y_true_oh = np.array([[1, 0, 0],
                       [0, 1, 0]])

grad = (probs - y_true_oh) / 2  # divide by batch size

print("Softmax + CCE gradient = probs - labels")
print("─" * 44)
for i in range(2):
    true_class = np.argmax(y_true_oh[i])
    print(f"Sample {i} (true class {true_class}):")
    for c in range(3):
        marker = " ← true class" if c == true_class else ""
        sign = "↓ reduce" if grad[i, c] > 0 else "↑ increase"
        print(f"  class {c}: prob={probs[i,c]:.4f}, grad={grad[i,c]:+.4f} ({sign}){marker}")

print("\nThe true class gets negative gradient (increase its prob)")
print("Wrong classes get positive gradient (decrease their prob)")
`}
/>

> [!NOTE] Why This Matters
> This combined gradient is numerically stable, computationally cheap, and avoids the vanishing gradient problem that plagues sigmoid + MSE combinations. ==Always pair softmax with cross-entropy, never with MSE.==

## Loss Cheat Sheet

| Task | Output Activation | Loss Function | Gradient |
|------|------------------|---------------|----------|
| Regression | Linear (none) | MSE | <InlineMath latex="2(\hat{y}-y)/N" /> |
| Binary Classification | Sigmoid | BCE | <InlineMath latex="(\hat{y}-y)/(\hat{y}(1-\hat{y})N)" /> |
| Multi-Class | Softmax | CCE | <InlineMath latex="(\hat{y}-y)/N" /> |
| Multi-Label | Sigmoid (per class) | BCE (per class) | Same as binary |

<Quiz
  chapterSlug="029-loss-functions"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why should you never pair softmax output with MSE loss?",
      options: [
        "MSE doesn't work with probabilities",
        "The gradient through softmax+MSE is complex and can vanish; softmax+CCE gives a clean (pred-true) gradient",
        "MSE requires regression targets",
        "Softmax outputs don't sum to 1 with MSE"
      ],
      correctIndex: 1,
      explanation: "Softmax + CCE has the elegant gradient (ŷ - y)/N. Softmax + MSE requires chaining through both softmax and squared error derivatives, producing messy gradients that can vanish when softmax saturates.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the CCE loss for a perfect prediction?",
      code: "import numpy as np\ny_true = np.array([[0, 1, 0]])\ny_pred = np.array([[0.0, 1.0, 0.0]])\ny_pred = np.clip(y_pred, 1e-7, 1-1e-7)\nloss = -np.sum(y_true * np.log(y_pred))\nprint(f'{loss:.6f}')",
      options: ["0.000000", "1.000000", "-0.000000", "NaN"],
      correctIndex: 0,
      explanation: "When the predicted probability for the true class is 1.0 (clipped to ~1.0), -log(1.0) = 0. Perfect predictions have zero cross-entropy loss.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In the softmax+CCE gradient (ŷ - y)/N, what does a POSITIVE gradient for class c mean?",
      options: [
        "Increase the logit for class c",
        "Decrease the logit for class c",
        "Class c is the correct class",
        "The model is confident about class c"
      ],
      correctIndex: 1,
      explanation: "Positive gradient means the loss increases when this logit increases. During gradient descent (which subtracts the gradient), a positive gradient causes the logit to decrease, reducing the predicted probability for this (incorrect) class.",
      randomize: true,
    }
  ]}
/>
