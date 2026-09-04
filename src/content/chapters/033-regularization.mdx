---
title: "Regularization"
slug: "033-regularization"
description: "Stop your network from memorizing training data. Implement L2 weight decay, dropout, and early stopping from scratch in NumPy."
track: "nn-beginner"
order: 6
read_time: 22
code_time: 18
execution_timeout: 15
prerequisites: ["032-mnist-from-scratch"]
---

# Regularization

Your MNIST network hits 97% on training but maybe 95% on test. That gap? ==Overfitting.== The network memorized noise instead of learning patterns. Regularization fights this by constraining what the network can learn.

Three weapons: weight decay, dropout, and early stopping.

## L2 Weight Decay

Add a penalty proportional to the squared magnitude of weights:

<BlockMath latex="L_{total} = L_{original} + \frac{\lambda}{2}\sum w^2" />

The gradient gets an extra term:

<BlockMath latex="\frac{\partial L_{total}}{\partial w} = \frac{\partial L_{original}}{\partial w} + \lambda w" />

This pushes weights toward zero. Smaller weights → smoother decision boundaries → better generalization.

```python
def add_weight_decay(grads, params, lam):
    """Add L2 regularization gradient to existing grads."""
    for key in grads:
        grads[key] = grads[key] + lam * params[key]
    return grads
```

<PyRunner
  cellId="033-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)
W = np.random.randn(5, 3) * 2.0  # large weights
lam = 0.01

# Original gradient (fake for demo)
orig_grad = np.random.randn(5, 3) * 0.1

# With weight decay
wd_grad = orig_grad + lam * W

print("Effect of L2 weight decay (λ=0.01):")
print(f"{'':>12} | {'Original':>10} | {'With WD':>10} | {'Shift':>10}")
print("─" * 52)
for i in range(3):
    og = orig_grad[i, 0]
    wg = wd_grad[i, 0]
    print(f"  W[{i},0]={W[i,0]:+.3f} | {og:+10.6f} | {wg:+10.6f} | {wg-og:+10.6f}")

print(f"\n💡 Large weights get pushed harder toward zero")
print(f"   Weight norm before: {np.linalg.norm(W):.4f}")
`}
/>

<Callout type="info" title="Weight Decay ≠ Learning Rate">
λ controls how aggressively weights shrink. Too high → underfitting (weights collapse to zero). Too low → no effect. Start with λ=0.0001 and adjust.
</Callout>

## Dropout

During training, randomly zero out neurons with probability <InlineMath latex="p" />. This forces the network to not rely on any single neuron — it must learn redundant representations.

<BlockMath latex="h_{train} = \frac{mask \odot h}{1 - p}" />

The division by <InlineMath latex="(1-p)" /> is ==inverted dropout== — it scales activations UP during training so you don't need to scale DOWN during inference.

```python
def dropout(H, p_drop, training=True):
    if not training or p_drop == 0:
        return H, np.ones_like(H)
    mask = (np.random.rand(*H.shape) > p_drop).astype(float)
    return H * mask / (1.0 - p_drop), mask
```

<PyRunner
  cellId="033-cell-2"
  defaultCode={`import numpy as np

np.random.seed(42)
H = np.ones((4, 8)) * 2.0  # all activations = 2.0
p_drop = 0.5

# Training mode
mask = (np.random.rand(*H.shape) > p_drop).astype(float)
H_dropped = H * mask / (1.0 - p_drop)

print("Dropout (p=0.5) on activations all equal to 2.0:")
print(f"Original:  {H[0]}")
print(f"Mask:      {mask[0].astype(int)}")
print(f"Dropped:   {H_dropped[0]}")
print(f"\nMean before dropout: {H.mean():.2f}")
print(f"Mean after dropout:  {H_dropped.mean():.2f} ← same! (inverted dropout)")
print(f"Active neurons: {mask.sum()}/{mask.size} ({mask.mean()*100:.0f}%)")

# Inference mode: no dropout, no scaling
print(f"\nInference: use full activations, no mask needed ✅")
`}
/>

### Dropout in Backprop

During backward pass, apply the SAME mask used in forward:

```python
# Forward
H_drop, mask = dropout(H, p_drop, training=True)

# Backward
dH = dH_drop * mask / (1.0 - p_drop)  # same mask!
```

<Callout type="warning" title="Save the Mask">
You MUST store the dropout mask from the forward pass and reuse it in backward. Generating a new random mask breaks the gradient computation entirely.
</Callout>

## Early Stopping

The simplest regularizer: stop training when validation loss stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_params = None
    
    def step(self, val_loss, params):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_params = {k: v.copy() for k, v in params.items()}
        else:
            self.counter += 1
        return self.counter >= self.patience
```

<PyRunner
  cellId="033-cell-3"
  defaultCode={`import numpy as np

# Simulate training curves
np.random.seed(42)
epochs = 30
train_loss = 2.0 * np.exp(-np.arange(epochs) * 0.15) + 0.1
val_loss = 2.0 * np.exp(-np.arange(epochs) * 0.10) + 0.3 + np.random.randn(epochs) * 0.05

# After epoch ~15, val_loss starts increasing while train_loss keeps dropping
val_loss[15:] += np.arange(15) * 0.03

patience = 5
best_loss = float('inf')
counter = 0
stop_epoch = epochs

print(f"{'Epoch':>5} | {'Train':>7} | {'Val':>7} | {'Status':>12}")
print("─" * 40)
for e in range(epochs):
    status = ""
    if val_loss[e] < best_loss:
        best_loss = val_loss[e]
        counter = 0
        status = "✓ improved"
    else:
        counter += 1
        status = f"no improve ({counter}/{patience})"
    
    if counter >= patience:
        stop_epoch = e
        status = "🛑 STOPPED"
        print(f"{e+1:5d} | {train_loss[e]:7.4f} | {val_loss[e]:7.4f} | {status}")
        break
    
    if e % 5 == 0 or counter > 0:
        print(f"{e+1:5d} | {train_loss[e]:7.4f} | {val_loss[e]:7.4f} | {status}")

print(f"\nBest validation loss: {best_loss:.4f} at epoch {stop_epoch - patience + 1}")
print(f"Training stopped at epoch {stop_epoch} (patience={patience})")
`}
/>

## Putting It All Together

Here's how regularization integrates into the training loop from chapter 032:

```python
# In forward pass:
Z1 = X @ W1 + b1
H = np.maximum(0, Z1)
H_drop, mask = dropout(H, p_drop=0.2, training=True)  # ← dropout
Z2 = H_drop @ W2 + b2

# In backward pass:
dZ2 = (probs - y_oh) / N
dW2 = H_drop.T @ dZ2  # ← use dropped-out activations
db2 = np.sum(dZ2, axis=0)
dH_drop = dZ2 @ W2.T
dH = dH_drop * mask / (1.0 - 0.2)  # ← same mask
dZ1 = dH * (Z1 > 0).astype(float)
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0)

# Add weight decay to gradients:
dW1 += lam * W1
dW2 += lam * W2

# After each epoch, check early stopping:
if early_stopper.step(val_loss, {"W1": W1, "b1": b1, "W2": W2, "b2": b2}):
    break
```

> [!IMPORTANT] Regularization Priority
> 1. **Start without regularization** — establish baseline
> 2. **Add weight decay** (λ=0.0001) if overfitting
> 3. **Add dropout** (p=0.2–0.5) if still overfitting
> 4. **Use early stopping** always — it's free insurance
> Don't stack everything at once. Diagnose first.

## Quick Reference

| Technique | When to Use | Typical Value | Cost |
|-----------|------------|---------------|------|
| L2 Weight Decay | Default regularizer | λ = 0.0001 | Negligible |
| Dropout | Wide networks, overfitting | p = 0.2–0.5 | Store mask per layer |
| Early Stopping | Always | patience = 5–10 | Save best params |
| Data Augmentation | Image tasks | Varies | Preprocessing time |

<Quiz
  chapterSlug="033-regularization"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does inverted dropout divide by (1-p) during training?",
      options: [
        "To increase the gradient signal",
        "So that expected activation values stay the same, meaning no scaling is needed at inference time",
        "To compensate for the reduced number of active neurons",
        "It doesn't matter — it's just a convention"
      ],
      correctIndex: 1,
      explanation: "Without scaling, the expected value of dropped activations is (1-p)·h. Dividing by (1-p) restores the expected value to h. At inference, all neurons are active and no scaling is needed.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What happens if you use a different random mask in backward than in forward for dropout?",
      options: [
        "Nothing — dropout is stochastic anyway",
        "The gradient becomes noisy but still correct on average",
        "The gradient computation is completely wrong — you're differentiating through a different function than the forward pass",
        "It acts as additional regularization"
      ],
      correctIndex: 2,
      explanation: "Backprop computes the gradient of the EXACT function computed in forward. Using a different mask means you're computing the gradient of a different function. The saved mask ensures consistency.",
      randomize: false,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "What is the L2 regularization gradient for w = [3.0, -2.0] with λ = 0.1?",
      code: "import numpy as np\nw = np.array([3.0, -2.0])\nlam = 0.1\nreg_grad = lam * w\nprint(reg_grad)",
      options: ["[0.3, -0.2]", "[0.09, 0.04]", "[3.0, -2.0]", "[0.03, -0.02]"],
      correctIndex: 0,
      explanation: "L2 gradient = λ·w = 0.1 × [3.0, -2.0] = [0.3, -0.2]. This pushes both weights toward zero proportionally to their magnitude.",
      randomize: false,
    }
  ]}
/>
