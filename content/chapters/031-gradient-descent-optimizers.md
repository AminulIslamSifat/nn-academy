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
      prompt: "What is the vanilla SGD update rule?",
      options: ["w ← w − η · ∇L", "w ← w + η · ∇L", "w ← w − ∇L / η", "w ← η · w − ∇L"],
      correctIndex: 0,
      explanation: "Subtract learning rate times gradient. Move in the direction that decreases loss.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is a key problem with vanilla SGD?",
      options: ["Same learning rate for all parameters, oscillates in narrow valleys, no memory of past gradients", "It's too fast", "It can't handle batches", "It requires GPU"],
      correctIndex: 0,
      explanation: "Vanilla SGD treats every parameter identically and has no mechanism to accelerate through consistent gradients or dampen oscillation.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does Momentum SGD add to vanilla SGD?",
      options: ["A velocity term that accumulates past gradients, accelerating in consistent directions and dampening oscillation", "Adaptive per-parameter learning rates", "Second-order derivative information", "Random noise"],
      correctIndex: 0,
      explanation: "v ← βv + (1−β)∇L, then w ← w − ηv. Velocity builds up when gradients point the same way, smoothing out noise.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "With momentum β=0.9, roughly how many past gradients does the velocity average?",
      options: ["~10 steps (1/(1−β))", "~100 steps", "~2 steps", "All past gradients equally"],
      correctIndex: 0,
      explanation: "Effective window ≈ 1/(1−β) = 1/0.1 = 10. Higher β means more smoothing but slower adaptation.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does AdaGrad do?",
      options: ["Scales the learning rate per-parameter based on cumulative squared gradients", "Adds momentum to SGD", "Uses second derivatives", "Randomly drops gradients"],
      correctIndex: 0,
      explanation: "s ← s + (∇L)², then w ← w − η·∇L/(√s + ε). Parameters with large historical gradients get smaller effective LRs.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What is AdaGrad's fatal flaw?",
      options: ["The cache only grows, never shrinks — effective learning rate → 0 over time", "It doesn't support batches", "It's slower than SGD", "It causes gradient explosion"],
      correctIndex: 0,
      explanation: "Cumulative sum of squared gradients monotonically increases. Denominator grows forever, eventually stopping learning. Fixed by RMSProp.",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "How does RMSProp fix AdaGrad's problem?",
      options: ["Replaces cumulative sum with exponential moving average so old gradients decay", "Adds momentum", "Removes the square root", "Uses a fixed learning rate"],
      correctIndex: 0,
      explanation: "s ← βs + (1−β)(∇L)². Old squared gradients decay exponentially, keeping the effective learning rate alive throughout training.",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does Adam combine?",
      options: ["Momentum (first moment) + RMSProp (second moment) + bias correction", "SGD + AdaGrad", "Only momentum", "Only adaptive learning rates"],
      correctIndex: 0,
      explanation: "Adam tracks both the mean (m) and variance (v) of gradients, applies bias correction, and combines the best of momentum and adaptive LR.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why does Adam need bias correction?",
      options: ["m and v start at zero, so early estimates are biased toward zero; correction rescales them", "To prevent division by zero", "To normalize across batch sizes", "To prevent gradient explosion"],
      correctIndex: 0,
      explanation: "At t=1 with β₁=0.9, m = 0.1·g — severely underestimated. Dividing by (1−β₁ᵗ) corrects this. Without it, initial steps are too tiny.",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nbeta1 = 0.9\ng = np.array([2.0])\nm = (1 - beta1) * g\nt = 1\nm_hat = m / (1 - beta1**t)\nprint(f'{m_hat[0]:.1f}')",
      options: ["2.0", "0.2", "1.8", "0.0"],
      correctIndex: 0,
      explanation: "m = 0.1 × 2.0 = 0.2. Bias correction: 0.2 / (1 − 0.9) = 0.2 / 0.1 = 2.0. Perfectly recovers the true gradient.",
      randomize: false,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What are Adam's default hyperparameters?",
      options: ["lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8", "lr=0.01, β₁=0.99, β₂=0.9999", "lr=0.1, β₁=0.5, β₂=0.5", "lr=0.001, β₁=0.99, β₂=0.9"],
      correctIndex: 0,
      explanation: "These defaults work well out of the box for most problems. Only tune if you have a specific reason.",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What problem does RMSProp solve that AdaGrad cannot?",
      options: ["AdaGrad's cumulative cache grows forever, killing the learning rate; RMSProp uses EMA so old gradients decay", "RMSProp uses momentum", "RMSProp handles sparse gradients better", "AdaGrad doesn't support batches"],
      correctIndex: 0,
      explanation: "AdaGrad accumulates ALL past squared gradients → denominator → ∞ → LR → 0. RMSProp's EMA lets old values decay.",
      randomize: false,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "How much extra memory does Adam require per parameter compared to vanilla SGD?",
      options: ["2× (stores m and v for each parameter)", "1×", "0×", "3×"],
      correctIndex: 0,
      explanation: "Adam stores first moment (m) and second moment (v) per parameter. For large models, this doubles memory overhead.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Which optimizer should you start with for most deep learning tasks?",
      options: ["Adam", "Vanilla SGD", "AdaGrad", "RMSProp"],
      correctIndex: 0,
      explanation: "Adam is the default choice: adaptive LR + momentum + works well with default hyperparams. Only switch if you have a specific reason.",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "When might SGD+momentum generalize better than Adam?",
      options: ["When you have time to tune the learning rate schedule and want sharper minima", "Always — SGD is superior", "Never — Adam always generalizes better", "Only for regression"],
      correctIndex: 0,
      explanation: "SGD tends to find sharper minima which can generalize better. But it requires careful LR scheduling. Adam is easier but may find flatter minima.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What does the epsilon (ε) term prevent in adaptive optimizers?",
      options: ["Division by zero when the accumulated squared gradient is zero", "Gradient explosion", "Negative learning rates", "NaN in the loss"],
      correctIndex: 0,
      explanation: "If a parameter has never received a gradient, s=0 and √s=0. Adding ε (≈1e-8) prevents division by zero.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "In an elongated valley loss landscape (x² + 10y²), why does vanilla SGD struggle?",
      options: ["y converges fast (steep gradient) while x crawls (shallow gradient) — the conditioning problem", "Both dimensions converge equally", "x converges faster than y", "SGD can't handle 2D problems"],
      correctIndex: 0,
      explanation: "Different curvatures cause different convergence speeds. Adaptive optimizers (Adam, RMSProp) equalize this automatically.",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "What does the 'second moment' in Adam track?",
      options: ["Exponential moving average of squared gradients (variance estimate)", "The second derivative of the loss", "The gradient from two steps ago", "Squared weights"],
      correctIndex: 0,
      explanation: "v ← β₂v + (1−β₂)(∇L)². This estimates the variance of gradients, used to scale the learning rate per-parameter.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "If your model isn't training with Adam, what should you check FIRST before switching optimizers?",
      options: ["Loss values and gradients for bugs (NaN, inf, zero gradients)", "Switch to SGD immediately", "Increase the learning rate 10×", "Add more layers"],
      correctIndex: 0,
      explanation: "Optimizer issues are rarely the root cause. Check for NaN/inf loss, zero gradients, wrong shapes, or data problems first.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which optimizer is best suited for sparse gradients (e.g., NLP with rare words)?",
      options: ["AdaGrad — boosts learning rate for infrequently updated parameters", "Vanilla SGD", "Momentum SGD", "Adam with default settings"],
      correctIndex: 0,
      explanation: "AdaGrad gives large effective LR to parameters with few updates (small cumulative gradient). Ideal for sparse features. Though RMSProp/Adam also handle this reasonably.",
      randomize: true,
    }
  ]}
/>
