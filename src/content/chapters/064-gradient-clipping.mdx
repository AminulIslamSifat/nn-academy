---
title: "Gradient Clipping"
slug: "064-gradient-clipping"
description: "Prevent exploding gradients by capping gradient magnitudes. Implement norm-based and value-based clipping in NumPy."
track: "training"
order: 2
read_time: 15
code_time: 10
execution_timeout: 10
prerequisites: ["063-mini-batch-training"]
---

# Gradient Clipping

Sometimes gradients explode — especially in RNNs or with high learning rates. ==Gradient clipping caps the gradient magnitude== without changing its direction, preventing catastrophic parameter updates.

## Norm-Based Clipping

Scale all gradients down if their total norm exceeds a threshold:

```python
def clip_grad_norm(grads, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        grads = {k: v * scale for k, v in grads.items()}
    return grads, total_norm
```

<PyRunner
  cellId="064-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)

# Simulate exploding gradients
grads = {
    "W1": np.random.randn(100, 50) * 10,  # large!
    "b1": np.random.randn(50) * 10,
    "W2": np.random.randn(50, 10) * 10,
}

total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
print(f"Before clipping: total gradient norm = {total_norm:.2f}")

# Clip to max_norm=1.0
max_norm = 1.0
if total_norm > max_norm:
    scale = max_norm / total_norm
    grads = {k: v * scale for k, v in grads.items()}

clipped_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
print(f"After clipping:  total gradient norm = {clipped_norm:.2f}")
print(f"Scale factor: {scale:.4f}")
print(f"\n✅ Direction preserved, magnitude capped")
print(f"   This prevents a single bad batch from destroying trained weights")
`}
/>

## Value-Based Clipping

Simpler: clamp each gradient element to [-threshold, threshold]:

```python
def clip_grad_value(grads, threshold=1.0):
    return {k: np.clip(v, -threshold, threshold) for k, v in grads.items()}
```

> [!IMPORTANT] When to Use Which
> - **Norm clipping**: Preferred for most cases. Preserves gradient direction.
> - **Value clipping**: Simpler, used in original RNN papers. Can distort direction.
> - **Typical threshold**: 1.0 for norm, 5.0 for value. Start here and adjust.

<Quiz
  chapterSlug="064-gradient-clipping"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does norm-based clipping preserve gradient direction?",
      options: ["It doesn't", "It multiplies ALL gradients by the same scalar (max_norm/total_norm), which scales magnitude uniformly without changing the relative proportions between parameters", "It only clips the largest gradient", "Direction doesn't matter"],
      correctIndex: 1,
      explanation: "Multiplying a vector by a positive scalar changes its length but not its direction. Since all gradients are scaled by the same factor, the update direction is preserved.",
      randomize: true,
    }
  ]}
/>
