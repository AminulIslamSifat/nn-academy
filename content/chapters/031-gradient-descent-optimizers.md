---
title: "Gradient Descent & Optimizers"
slug: "031-gradient-descent-optimizers"
description: "From vanilla SGD to Momentum, AdaGrad, RMSProp, and Adam. Understand why each optimizer exists and implement them all in NumPy."
track: "nn-beginner"
order: 4
read_time: 25
code_time: 20
execution_timeout: 15
prerequisites: ["030-backpropagation"]
---

# Gradient Descent & Optimizers

Backprop tells you ==which direction== to move. The optimizer decides ==how far and how smartly== to move. Vanilla gradient descent works, but it's slow, gets stuck in ravines, and treats every parameter identically. Modern optimizers fix these problems.

## Vanilla SGD

The simplest update rule:

<BlockMath latex="w \leftarrow w - \eta \cdot \nabla_w L" />

```python
import numpy as np

def sgd_update(params, grads, lr):
    for key in params:
        params[key] -= lr * grads[key]
```

Problems:
- Same learning rate for all parameters
- Oscillates in narrow valleys
- No memory of past gradients
- Sensitive to feature scaling

<PyRunner
  cellId="031-cell-1"
  defaultCode={`import numpy as np

# Visualize SGD on a loss landscape
def loss_fn(x, y):
    return x**2 + 10*y**2  # elongated valley

def grad_fn(x, y):
    return np.array([2*x, 20*y])

lr = 0.05
pos = np.array([3.0, 3.0])
trajectory = [pos.copy()]

for _ in range(30):
    g = grad_fn(*pos)
    pos = pos - lr * g
    trajectory.append(pos.copy())

print("Vanilla SGD on elongated valley (x² + 10y²):")
print(f"{'Step':>4} | {'x':>7} | {'y':>7} | {'Loss':>9}")
print("─" * 38)
for i, p in enumerate(trajectory[::5]):
    print(f"{i*5:4d} | {p[0]:+7.4f} | {p[1]:+7.4f} | {loss_fn(*p):9.4f}")

print("\n⚠️ y converges fast (steep), x crawls (shallow)")
print("   This is the conditioning problem of vanilla SGD")
`}
/>

## Momentum

Idea: accumulate velocity. If gradients keep pointing the same way, go faster. If they oscillate, dampen.

<BlockMath latex="v \leftarrow \beta v + (1-\beta)\nabla_w L, \quad w \leftarrow w - \eta \cdot v" />

```python
class MomentumSGD:
    def __init__(self, params, lr=0.01, beta=0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocity = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, grads):
        for key in self.params:
            self.velocity[key] = (
                self.beta * self.velocity[key] + 
                (1 - self.beta) * grads[key]
            )
            self.params[key] -= self.lr * self.velocity[key]
```

<PyRunner
  cellId="031-cell-2"
  defaultCode={`import numpy as np

def loss_fn(x, y):
    return x**2 + 10*y**2

def grad_fn(x, y):
    return np.array([2*x, 20*y])

# Compare SGD vs Momentum
results = {}
for name, use_momentum in [("SGD", False), ("Momentum", True)]:
    pos = np.array([3.0, 3.0])
    vel = np.array([0.0, 0.0])
    lr, beta = 0.05, 0.9
    traj = [pos.copy()]
    
    for _ in range(30):
        g = grad_fn(*pos)
        if use_momentum:
            vel = beta * vel + (1 - beta) * g
            pos = pos - lr * vel
        else:
            pos = pos - lr * g
        traj.append(pos.copy())
    results[name] = traj

print(f"{'Step':>4} | {'SGD Loss':>10} | {'Momentum Loss':>14}")
print("─" * 36)
for i in range(0, 31, 5):
    sg = results["SGD"][i]
    mo = results["Momentum"][i]
    print(f"{i:4d} | {loss_fn(*sg):10.4f} | {loss_fn(*mo):14.4f}")

print("\n✅ Momentum accelerates through consistent gradients")
print("   and dampens oscillation in noisy directions")
`}
/>

> [!NOTE] Why β = 0.9?
> With β=0.9, the effective window is roughly 1/(1-β) = 10 steps. The velocity averages the last ~10 gradients. Higher β = more smoothing but slower adaptation.

## AdaGrad: Adaptive Learning Rates

Problem: some parameters need big updates, others need tiny ones. AdaGrad scales the learning rate per-parameter based on historical gradient magnitude:

<BlockMath latex="s \leftarrow s + (\nabla_w L)^2, \quad w \leftarrow w - \frac{\eta}{\sqrt{s} + \epsilon}\nabla_w L" />

Parameters with consistently large gradients get smaller effective learning rates. Parameters with small gradients get boosted.

```python
class AdaGrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        self.params = params
        self.lr = lr
        self.eps = eps
        self.cache = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, grads):
        for key in self.params:
            self.cache[key] += grads[key] ** 2
            self.params[key] -= self.lr * grads[key] / (
                np.sqrt(self.cache[key]) + self.eps
            )
```

<Callout type="warning" title="AdaGrad's Fatal Flaw">
The cache only grows, never shrinks. Over time, the denominator keeps increasing and the effective learning rate → 0. Training stalls. This is why AdaGrad was superseded by RMSProp.
</Callout>

## RMSProp: Fixing AdaGrad

Replace the cumulative sum with an exponential moving average:

<BlockMath latex="s \leftarrow \beta s + (1-\beta)(\nabla_w L)^2, \quad w \leftarrow w - \frac{\eta}{\sqrt{s} + \epsilon}\nabla_w L" />

Now old squared gradients decay. The effective learning rate stays alive throughout training.

```python
class RMSProp:
    def __init__(self, params, lr=0.001, beta=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, grads):
        for key in self.params:
            self.cache[key] = (
                self.beta * self.cache[key] + 
                (1 - self.beta) * grads[key] ** 2
            )
            self.params[key] -= self.lr * grads[key] / (
                np.sqrt(self.cache[key]) + self.eps
            )
```

## Adam: Best of Both Worlds

==The default optimizer for most deep learning.== Combines momentum (first moment) with RMSProp (second moment), plus bias correction:

<BlockMath latex="m \leftarrow \beta_1 m + (1-\beta_1)\nabla_w L" />
<BlockMath latex="v \leftarrow \beta_2 v + (1-\beta_2)(\nabla_w L)^2" />
<BlockMath latex="\hat{m} = \frac{m}{1-\beta_1^t}, \quad \hat{v} = \frac{v}{1-\beta_2^t}" />
<BlockMath latex="w \leftarrow w - \frac{\eta}{\sqrt{\hat{v}} + \epsilon}\hat{m}" />

<PyRunner
  cellId="031-cell-3"
  defaultCode={`import numpy as np

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, grads):
        self.t += 1
        for key in self.params:
            # Update biased moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            
            # Bias correction (critical for early training!)
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            # Update
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Demo: train a simple model with Adam
np.random.seed(42)
params = {"W": np.random.randn(10, 5) * 0.5, "b": np.zeros(5)}
optimizer = Adam(params, lr=0.01)

X = np.random.randn(32, 10)
y = np.random.randint(0, 5, 32)
y_oh = np.zeros((32, 5)); y_oh[np.arange(32), y] = 1

print("Training with Adam optimizer:")
print(f"{'Step':>4} | {'Loss':>8} | {'|grad_W|':>10}")
print("─" * 30)

for step in range(50):
    # Forward
    logits = X @ params["W"] + params["b"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-7, 1)), axis=1))
    
    # Backward
    dZ = (probs - y_oh) / 32
    grads = {"W": X.T @ dZ, "b": dZ.sum(axis=0)}
    
    if step % 10 == 0:
        gnorm = np.linalg.norm(grads["W"])
        print(f"{step:4d} | {loss:8.4f} | {gnorm:10.6f}")
    
    optimizer.step(grads)

print(f"\n✅ Adam adapts learning rates per-parameter")
print(f"   Default hyperparams (lr=0.001, β₁=0.9, β₂=0.999) work well out of the box")
`}
/>

### Why Bias Correction?

At step 1 with β₁=0.9: <InlineMath latex="m = 0.1 \cdot g" />. The momentum estimate is biased toward zero. Dividing by <InlineMath latex="(1 - 0.9^1) = 0.1" /> corrects this. Without bias correction, Adam takes tiny steps at the start of training.

## Optimizer Comparison

| Optimizer | Adaptive LR | Momentum | Memory/Param | Best For |
|-----------|------------|----------|-------------|----------|
| SGD | ❌ | ❌ | 0× | Simple problems, fine-tuning |
| Momentum | ❌ | ✅ | 1× | Consistent gradient directions |
| AdaGrad | ✅ | ❌ | 1× | Sparse gradients (NLP) |
| RMSProp | ✅ | ❌ | 1× | Non-stationary problems |
| Adam | ✅ | ✅ | 2× | ==Default choice for most tasks== |

> [!IMPORTANT] Practical Advice
> Start with Adam (lr=0.001). If your model isn't training, check your loss and gradients before switching optimizers. Only switch to SGD+momentum if you need better generalization and have time to tune the learning rate schedule.

<Quiz
  chapterSlug="031-gradient-descent-optimizers"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does Adam need bias correction?",
      options: [
        "To prevent division by zero",
        "Because m and v are initialized to zero, their early estimates are biased toward zero; correction rescales them",
        "To normalize across different batch sizes",
        "To prevent gradient explosion"
      ],
      correctIndex: 1,
      explanation: "At t=1 with β₁=0.9, m = 0.1·g which severely underestimates the true first moment. Dividing by (1-β₁ᵗ) corrects this bias. Without it, Adam's initial steps are too small.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What problem does RMSProp solve that AdaGrad cannot?",
      options: [
        "RMSProp uses momentum",
        "AdaGrad's cumulative sum of squared gradients only grows, causing the learning rate to shrink to zero over time",
        "RMSProp handles sparse gradients better",
        "AdaGrad doesn't support batches"
      ],
      correctIndex: 1,
      explanation: "AdaGrad accumulates ALL past squared gradients. In long training runs, the denominator grows without bound and the effective learning rate → 0. RMSProp uses exponential moving average instead, allowing old gradients to decay.",
      randomize: false,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "After 1 step of Adam with β₁=0.9, what is the bias-corrected first moment if the gradient is [2.0]?",
      code: "import numpy as np\nbeta1 = 0.9\ng = np.array([2.0])\nm = beta1 * 0 + (1 - beta1) * g  # m after step 1\nt = 1\nm_hat = m / (1 - beta1**t)\nprint(f'{m_hat[0]:.1f}')",
      options: ["0.2", "2.0", "1.8", "0.0"],
      correctIndex: 1,
      explanation: "m = 0.1 × 2.0 = 0.2. Bias correction: m_hat = 0.2 / (1 - 0.9¹) = 0.2 / 0.1 = 2.0. The correction perfectly recovers the true gradient value at step 1.",
      randomize: false,
    }
  ]}
/>
