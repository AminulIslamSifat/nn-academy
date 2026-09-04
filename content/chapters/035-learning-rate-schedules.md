---
title: "Learning Rate Schedules"
slug: "035-learning-rate-schedules"
description: "One learning rate doesn't fit all epochs. Implement step decay, cosine annealing, warmup, and cyclical schedules in NumPy."
track: "nn-beginner"
order: 8
read_time: 18
code_time: 12
execution_timeout: 10
prerequisites: ["031-gradient-descent-optimizers", "033-regularization"]
---

# Learning Rate Schedules

Early training needs large steps to escape bad initializations. Late training needs small steps to settle into a good minimum. ==A fixed learning rate can't do both.== Schedules adapt the learning rate during training.

## Step Decay

The simplest: multiply the learning rate by a factor every N epochs.

<BlockMath latex="\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}" />

```python
def step_decay(lr_init, epoch, step_size=10, gamma=0.1):
    return lr_init * (gamma ** (epoch // step_size))
```

<PyRunner
  cellId="035-cell-1"
  defaultCode={`import numpy as np

lr_init = 0.01
epochs = 40

print("Step Decay (γ=0.1, step=10):")
print(f"{'Epoch':>5} | {'LR':>10} | {'Visual'}")
print("─" * 40)
for e in range(epochs):
    lr = lr_init * (0.1 ** (e // 10))
    bar = "█" * int(lr / lr_init * 30)
    if e % 10 == 0 or e == epochs - 1:
        print(f"{e+1:5d} | {lr:10.6f} | {bar}")
`}
/>

## Cosine Annealing

Smoothly decay from max to min following a cosine curve. No abrupt drops:

<BlockMath latex="\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t \cdot \pi}{T}\right)\right)" />

```python
def cosine_annealing(lr_max, lr_min, epoch, total_epochs):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + np.cos(epoch * np.pi / total_epochs)
    )
```

<PyRunner
  cellId="035-cell-2"
  defaultCode={`import numpy as np

lr_max, lr_min = 0.01, 0.0001
T = 50

print("Cosine Annealing:")
print(f"{'Epoch':>5} | {'LR':>10} | {'Visual'}")
print("─" * 42)
for e in range(0, T+1, 5):
    lr = lr_min + 0.5*(lr_max-lr_min)*(1 + np.cos(e*np.pi/T))
    bar = "█" * int((lr - lr_min)/(lr_max - lr_min) * 30)
    print(f"{e:5d} | {lr:10.6f} | {bar}")

print(f"\n✅ Smooth decay — no sudden jumps")
print(f"   Starts at {lr_max}, ends at {lr_min}")
`}
/>

> [!NOTE] Why Cosine Works Well
> The gradual decay lets the optimizer explore early and converge smoothly late. Empirically outperforms step decay on most tasks. Used by default in many modern training recipes.

## Warmup

Start with a tiny learning rate and linearly increase to the target over the first few epochs. Prevents early divergence when gradients are noisy:

```python
def warmup_schedule(lr_max, epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    # Then cosine anneal
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return lr_max * 0.5 * (1 + np.cos(progress * np.pi))
```

<PyRunner
  cellId="035-cell-3"
  defaultCode={`import numpy as np

lr_max = 0.01
warmup = 5
total = 40

print("Warmup + Cosine Annealing:")
print(f"{'Epoch':>5} | {'LR':>10} | {'Phase':>10} | Visual")
print("─" * 52)
for e in range(total):
    if e < warmup:
        lr = lr_max * (e + 1) / warmup
        phase = "warmup"
    else:
        progress = (e - warmup) / (total - warmup)
        lr = lr_max * 0.5 * (1 + np.cos(progress * np.pi))
        phase = "cosine"
    
    bar = "█" * int(lr / lr_max * 25)
    if e < warmup or e % 5 == 0:
        print(f"{e+1:5d} | {lr:10.6f} | {phase:>10} | {bar}")

print(f"\n💡 Warmup prevents early divergence with large batch sizes")
print(f"   Standard recipe: 5-10 epoch warmup + cosine decay")
`}
/>

## Cyclical Learning Rates

Instead of monotonically decreasing, oscillate between bounds. Helps escape saddle points:

<BlockMath latex="\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \max\left(0, 1 - \left|\frac{t \bmod (2s)}{s} - 1\right|\right)" />

<PyRunner
  cellId="035-cell-4"
  defaultCode={`import numpy as np

lr_min, lr_max = 0.001, 0.01
step_size = 10
epochs = 50

print("Cyclical Learning Rate (triangular):")
print(f"{'Epoch':>5} | {'LR':>10} | Visual")
print("─" * 42)
for e in range(epochs):
    cycle_pos = abs((e % (2*step_size)) / step_size - 1)
    lr = lr_min + (lr_max - lr_min) * max(0, 1 - cycle_pos)
    bar = "█" * int((lr - lr_min)/(lr_max - lr_min) * 30)
    if e % 5 == 0:
        print(f"{e+1:5d} | {lr:10.6f} | {bar}")

print(f"\n🔄 LR bounces between {lr_min} and {lr_max}")
print(f"   High LR escapes saddle points, low LR refines")
`}
/>

## Choosing a Schedule

| Schedule | Best For | Complexity |
|----------|---------|------------|
| Fixed LR | Quick prototyping | None |
| Step Decay | Known convergence pattern | Low |
| Cosine Annealing | ==Default for most tasks== | Low |
| Warmup + Cosine | Large batches, transformers | Medium |
| Cyclical | Escaping saddle points | Medium |

> [!IMPORTANT] Practical Rule
> Start with **warmup (5 epochs) + cosine annealing**. Only try other schedules if you have a specific reason. The schedule matters less than getting the base learning rate right.

<Quiz
  chapterSlug="035-learning-rate-schedules"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does warmup help with large batch sizes?",
      options: [
        "Large batches produce smaller gradient variance, so warmup isn't needed",
        "Early in training, weights are random and gradients are noisy; large batches amplify this noise, causing divergence. Warmup starts conservatively.",
        "Warmup increases the effective batch size",
        "It's only useful for SGD, not Adam"
      ],
      correctIndex: 1,
      explanation: "Random initialization + large batch = high-variance gradient estimates. A full learning rate step early on can move weights catastrophically. Warmup gradually increases confidence as the model stabilizes.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What learning rate does cosine annealing give at exactly halfway through training?",
      code: "import numpy as np\nlr_max, lr_min = 0.01, 0.001\nt, T = 25, 50\nlr = lr_min + 0.5*(lr_max-lr_min)*(1 + np.cos(t*np.pi/T))\nprint(f'{lr:.4f}')",
      options: ["0.0055", "0.0050", "0.0010", "0.0100"],
      correctIndex: 0,
      explanation: "At t=T/2, cos(π/2)=0. So lr = 0.001 + 0.5×(0.01-0.001)×(1+0) = 0.001 + 0.0045 = 0.0055. Exactly the midpoint between lr_min and lr_max.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What advantage does cosine annealing have over step decay?",
      options: [
        "It's computationally cheaper",
        "No abrupt learning rate drops — smooth continuous decay avoids sudden changes in optimization dynamics",
        "It always converges faster",
        "It doesn't require tuning any hyperparameters"
      ],
      correctIndex: 1,
      explanation: "Step decay causes sudden LR drops that can destabilize training momentarily. Cosine provides smooth, continuous decay that empirically leads to better final performance on most benchmarks.",
      randomize: true,
    }
  ]}
/>
