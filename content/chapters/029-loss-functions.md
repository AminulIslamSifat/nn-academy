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
      prompt: "What does a loss function do in a neural network?",
      options: ["Converts predictions and targets into a single scalar that measures how wrong the network is", "Initializes weights", "Normalizes inputs", "Applies activation functions"],
      correctIndex: 0,
      explanation: "The loss function quantifies error. Backpropagation differentiates this scalar to compute gradients for weight updates.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the formula for Mean Squared Error?",
      options: ["(1/N) × Σ(y_true − y_pred)²", "Σ|y_true − y_pred|", "−Σ y_true × log(y_pred)", "max(y_true − y_pred)"],
      correctIndex: 0,
      explanation: "MSE averages the squared differences between predictions and targets. Squaring penalizes large errors more heavily.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the gradient of MSE with respect to predictions?",
      options: ["2(y_pred − y_true) / N", "y_pred − y_true", "2(y_true − y_pred)", "(y_pred − y_true)²"],
      correctIndex: 0,
      explanation: "d(MSE)/d(y_pred) = 2(y_pred − y_true)/N. The gradient points in the direction that increases loss; gradient descent subtracts it.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why do we square the error in MSE?",
      options: ["Makes loss always positive, penalizes large errors more, and gives smooth proportional gradients", "To make it faster", "To prevent negative losses only", "Squaring is arbitrary — absolute value works equally well"],
      correctIndex: 0,
      explanation: "Squaring ensures positivity, amplifies large errors, and produces a gradient proportional to the error magnitude.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Which loss function is used for binary classification?",
      options: ["Binary Cross-Entropy (BCE)", "MSE", "Categorical Cross-Entropy", "Hinge loss"],
      correctIndex: 0,
      explanation: "BCE measures divergence between predicted probability and true binary label. Paired with sigmoid output.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Why must you clip predictions before computing log in cross-entropy?",
      options: ["Because log(0) = −inf which produces NaN loss", "To speed up computation", "To normalize predictions", "Clipping isn't necessary"],
      correctIndex: 0,
      explanation: "If any prediction is exactly 0 or 1, log produces −inf or 0 respectively, corrupting the loss. Clip to [ε, 1−ε].",
      randomize: true,
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "What is the CCE loss for a perfect prediction?",
      code: "import numpy as np\ny_true = np.array([[0, 1, 0]])\ny_pred = np.clip(np.array([[0.0, 1.0, 0.0]]), 1e-7, 1-1e-7)\nloss = -np.sum(y_true * np.log(y_pred))\nprint(f'{loss:.6f}')",
      options: ["0.000000", "1.000000", "-0.000000", "NaN"],
      correctIndex: 0,
      explanation: "Perfect prediction: p(true class) ≈ 1.0, so −log(1.0) = 0. Zero loss means perfect classification.",
      randomize: false,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What is the combined gradient of softmax + categorical cross-entropy?",
      options: ["(ŷ − y) / N — predicted probability minus true label", "ŷ × (1 − ŷ)", "2(ŷ − y) / N", "−y / ŷ"],
      correctIndex: 0,
      explanation: "The beautiful shortcut: softmax+CCE gradient simplifies to just subtraction. No chain rule through softmax and log separately.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "In the gradient (ŷ − y)/N, what does a positive gradient for class c mean?",
      options: ["The model assigned too much probability to class c — decrease its logit", "Class c is the correct class", "Increase the logit for class c", "The model is confident about class c"],
      correctIndex: 0,
      explanation: "Positive gradient → gradient descent subtracts it → logit decreases → probability for this (wrong) class goes down.",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Why should you never pair softmax with MSE loss?",
      options: ["Softmax+CCE gives a clean (pred−true) gradient; softmax+MSE produces complex vanishing gradients", "MSE doesn't work with probabilities", "MSE requires regression targets", "Softmax outputs don't sum to 1 with MSE"],
      correctIndex: 0,
      explanation: "Softmax+CCE gradient is simply (ŷ−y)/N. Softmax+MSE chains through both derivatives, producing messy gradients that vanish when softmax saturates.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "For multi-class classification with 10 classes, what shape should y_true_onehot have for a batch of 32 samples?",
      options: ["(32, 10)", "(32,)", "(10, 32)", "(32, 1)"],
      correctIndex: 0,
      explanation: "One-hot encoding: each sample has a vector of length C=10 with exactly one 1 and nine 0s. Batch shape is (N, C).",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What is the CCE loss for random guessing on a 5-class problem?",
      options: ["−log(0.2) ≈ 1.609", "0.0", "1.0", "−log(0.5) ≈ 0.693"],
      correctIndex: 0,
      explanation: "Random guessing assigns p=1/5=0.2 to each class. −log(0.2) ≈ 1.609. This is the baseline — your model must beat this.",
      randomize: true,
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\ndef mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)\nprint(mse(np.array([1,2,3]), np.array([1,2,3])))",
      options: ["0.0", "1.0", "3.0", "nan"],
      correctIndex: 0,
      explanation: "Perfect predictions: all errors are 0, so MSE = 0.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Which loss function should you use for regression?",
      options: ["MSE", "Binary Cross-Entropy", "Categorical Cross-Entropy", "Hinge loss"],
      correctIndex: 0,
      explanation: "MSE is the standard for continuous-valued regression. Cross-entropy losses are for classification.",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What happens to BCE loss as the predicted probability for the true class approaches 0?",
      options: ["Loss approaches +∞", "Loss approaches 0", "Loss stays constant", "Loss becomes negative"],
      correctIndex: 0,
      explanation: "−log(p) → ∞ as p → 0. The model is heavily penalized for being confidently wrong.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "In batched CCE, why do you sum over axis=1 then mean?",
      options: ["Sum across classes per sample (only true class contributes), then average across the batch", "Sum across samples, then average classes", "Both at once", "Mean first, then sum"],
      correctIndex: 0,
      explanation: "Each sample's loss is −Σ y_c × log(ŷ_c). Since y is one-hot, only the true class contributes. Then average over N samples.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "For multi-label classification (an image can be both 'cat' AND 'indoor'), which setup is correct?",
      options: ["Sigmoid per class + BCE per class", "Softmax + CCE", "Linear output + MSE", "Single sigmoid + BCE"],
      correctIndex: 0,
      explanation: "Multi-label means classes are independent. Sigmoid per class treats each as a separate binary decision. Softmax forces mutual exclusivity.",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "What typical epsilon value is used for clipping in cross-entropy?",
      options: ["1e-7", "0.1", "1e-1", "0.5"],
      correctIndex: 0,
      explanation: "ε ≈ 1e-7 is small enough to not affect normal predictions but prevents log(0). Too large distorts the loss.",
      randomize: true,
    },
    {
      id: "q19",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprobs = np.array([0.1, 0.7, 0.2])\ny_true_oh = np.array([0, 1, 0])\ngrad = probs - y_true_oh\nprint(grad)",
      options: ["[ 0.1 -0.3  0.2]", "[-0.1  0.3 -0.2]", "[0.1 0.7 0.2]", "[0 1 0]"],
      correctIndex: 0,
      explanation: "softmax+CCE gradient = ŷ − y. True class gets negative gradient (increase prob), wrong classes get positive (decrease prob).",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Why is the softmax+CCE gradient numerically better than sigmoid+BCE for deep networks?",
      options: ["It avoids vanishing gradients because the gradient is directly (ŷ−y), not multiplied by σ'(z)", "It's computationally cheaper", "It works without clipping", "It produces larger gradients always"],
      correctIndex: 0,
      explanation: "Sigmoid+BCE gradient includes σ'(z) = σ(z)(1−σ(z)) which vanishes at extremes. Softmax+CCE gradient (ŷ−y) doesn't have this problem.",
      randomize: true,
    }
  ]}
/>
