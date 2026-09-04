---
title: "Mini-Batch Training"
slug: "063-mini-batch-training"
description: "Why not use the full dataset or single samples? Understand the bias-variance tradeoff of batch sizes and implement efficient mini-batch loops."
track: "training"
order: 1
read_time: 18
code_time: 12
execution_timeout: 10
prerequisites: ["031-gradient-descent-optimizers", "032-mnist-from-scratch"]
---

# Mini-Batch Training

Full-batch gradient descent uses ALL data for one update — accurate but slow. Stochastic (single sample) is fast but noisy. ==Mini-batches hit the sweet spot.==

## The Three Regimes

| Mode | Batch Size | Gradient Quality | Speed per Step | Steps per Epoch |
|------|-----------|-----------------|---------------|----------------|
| Full-batch | N | Exact | Slow | 1 |
| Mini-batch | 32-512 | Noisy but good | Fast | N/B |
| Stochastic | 1 | Very noisy | Fastest | N |

<PyRunner
  cellId="063-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)
N = 1000
X = np.random.randn(N, 10)
w_true = np.random.randn(10)
y = X @ w_true + np.random.randn(N) * 0.5

# Compare gradient noise at different batch sizes
w = np.zeros(10)
true_grad = 2 * X.T @ (X @ w - y) / N

print(f"True gradient norm: {np.linalg.norm(true_grad):.4f}")
print(f"\nGradient noise by batch size:")
print(f"{'Batch':>6} | {'Grad norm':>10} | {'Cosine sim':>10} | {'Rel error':>10}")
print("─" * 44)

for bs in [1, 8, 32, 128, 512, 1000]:
    idx = np.random.choice(N, min(bs, N), replace=False)
    X_b, y_b = X[idx], y[idx]
    grad = 2 * X_b.T @ (X_b @ w - y_b) / len(idx)
    cos_sim = np.dot(grad, true_grad) / (np.linalg.norm(grad) * np.linalg.norm(true_grad) + 1e-8)
    rel_err = np.linalg.norm(grad - true_grad) / (np.linalg.norm(true_grad) + 1e-8)
    print(f"{bs:6d} | {np.linalg.norm(grad):10.4f} | {cos_sim:10.4f} | {rel_err:10.4f}")

print(f"\n💡 Larger batches → gradients closer to true gradient")
print(f"   But diminishing returns past ~256")
`}
/>

## Efficient Mini-Batch Loop

```python
def create_batches(X, y, batch_size, rng):
    indices = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
```

<PyRunner
  cellId="063-cell-2"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)
N, D, C = 1000, 784, 10
X = np.random.randn(N, D)
y = np.random.randint(0, C, N)

def create_batches(X, y, bs, rng):
    idx = rng.permutation(len(X))
    for s in range(0, len(X), bs):
        e = min(s + bs, len(X))
        bi = idx[s:e]
        yield X[bi], y[bi]

batch_size = 128
batches = list(create_batches(X, y, batch_size, rng))

print(f"Dataset: {N} samples")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(batches)}")
print(f"Samples covered: {sum(len(b[1]) for b in batches)}")
print(f"Last batch size: {len(batches[-1][1])}")

# Verify no duplicates within epoch
all_idx = []
idx = rng.permutation(N)
for s in range(0, N, batch_size):
    all_idx.extend(idx[s:min(s+batch_size, N)].tolist())
print(f"All samples seen exactly once: {len(set(all_idx)) == N}")
`}
/>

## Choosing Batch Size

| Factor | Small Batch (16-64) | Large Batch (256-1024) |
|--------|--------------------|-----------------------|
| Generalization | Often better (noise = regularization) | Can generalize worse |
| GPU utilization | Lower | Higher |
| BN statistics | Noisy | Stable |
| Learning rate | Lower | Higher (linear scaling rule) |
| Convergence steps | More | Fewer |

> [!IMPORTANT] Linear Scaling Rule
> When you double the batch size, double the learning rate. This keeps the effective update magnitude similar. Works well up to ~8K batch size.

<Quiz
  chapterSlug="063-mini-batch-training"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why do small batches often generalize better than large batches?",
      options: ["They're faster", "The gradient noise from small batches acts as implicit regularization, preventing the model from settling into sharp minima that don't generalize", "Large batches overfit", "Small batches use more data"],
      correctIndex: 1,
      explanation: "Noisy gradients from small batches prevent convergence to sharp minima (which are sensitive to parameter perturbation). The model settles into flatter minima that are more robust to unseen data.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "According to the linear scaling rule, if you increase batch size from 64 to 256, what should you do to the learning rate?",
      options: ["Keep it the same", "Multiply by 4 (same ratio as batch size increase)", "Divide by 4", "Square it"],
      correctIndex: 1,
      explanation: "4× larger batch means 4× more accurate gradient estimate. To maintain similar update dynamics, scale LR proportionally: 256/64 = 4×.",
      randomize: false,
    }
  ]}
/>
