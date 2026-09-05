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
      prompt: "Why use a learning rate schedule instead of a fixed learning rate?",
      options: ["Early training needs large steps to escape bad init; late training needs small steps to settle into minima", "Fixed LR never works", "Schedules make training faster computationally", "Schedules prevent NaN"],
      correctIndex: 0,
      explanation: "A single LR can't be both aggressive (exploration) and precise (convergence). Schedules adapt over time.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How does step decay work?",
      options: ["Multiply LR by γ every N epochs: η = η₀ × γ^(epoch//step)", "Linearly decrease LR each epoch", "Randomly adjust LR", "Double LR every N epochs"],
      correctIndex: 0,
      explanation: "Step decay drops LR by a factor (e.g., 0.1) at fixed intervals. Simple but causes abrupt changes.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is cosine annealing?",
      options: ["Smoothly decays LR from max to min following a cosine curve", "Steps LR down at fixed intervals", "Oscillates LR between bounds", "Keeps LR constant"],
      correctIndex: 0,
      explanation: "η = η_min + ½(η_max − η_min)(1 + cos(t·π/T)). Smooth, no abrupt drops. Empirically outperforms step decay.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nlr_max, lr_min = 0.01, 0.001\nt, T = 25, 50\nlr = lr_min + 0.5*(lr_max-lr_min)*(1 + np.cos(t*np.pi/T))\nprint(f'{lr:.4f}')",
      options: ["0.0055", "0.0050", "0.0010", "0.0100"],
      correctIndex: 0,
      explanation: "At t=T/2, cos(π/2)=0. lr = 0.001 + 0.5×0.009×1 = 0.0055. Midpoint between min and max.",
      randomize: false,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does warmup do?",
      options: ["Starts with tiny LR and linearly increases to target over first few epochs", "Starts with large LR and decreases", "Keeps LR constant for initial epochs", "Randomly initializes LR"],
      correctIndex: 0,
      explanation: "Warmup prevents early divergence when weights are random and gradients are noisy. Gradually builds confidence.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Why does warmup help with large batch sizes?",
      options: ["Random init + large batch = noisy gradients; full LR early causes catastrophic updates", "Large batches don't need warmup", "Warmup increases effective batch size", "Only useful for SGD"],
      correctIndex: 0,
      explanation: "Large batches amplify gradient noise from random initialization. Warmup starts conservatively, letting the model stabilize before full LR.",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is the standard modern training recipe for learning rate?",
      options: ["Warmup (5-10 epochs) + cosine annealing", "Fixed LR throughout", "Step decay only", "Cyclical LR only"],
      correctIndex: 0,
      explanation: "Warmup + cosine is the default for transformers and most modern architectures. Simple and effective.",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What advantage does cosine annealing have over step decay?",
      options: ["Smooth continuous decay — no abrupt LR drops that destabilize optimization", "Computationally cheaper", "Always converges faster", "No hyperparameters needed"],
      correctIndex: 0,
      explanation: "Step decay causes sudden jumps. Cosine provides smooth decay that empirically leads to better final performance.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "What do cyclical learning rates do?",
      options: ["Oscillate LR between min and max bounds, helping escape saddle points", "Monotonically decrease LR", "Keep LR constant", "Randomly change LR"],
      correctIndex: 0,
      explanation: "High LR phases escape saddle points/local minima. Low LR phases refine the solution. Triangular or cosine cycles.",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "At what point in cosine annealing is the LR at its minimum?",
      options: ["At the end of training (t = T)", "At the start", "At the midpoint", "Never reaches minimum"],
      correctIndex: 0,
      explanation: "cos(T·π/T) = cos(π) = −1. So lr = lr_min + ½(lr_max−lr_min)(1−1) = lr_min.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "At what point in cosine annealing is the LR at its maximum?",
      options: ["At the start of training (t = 0)", "At the end", "At the midpoint", "After warmup"],
      correctIndex: 0,
      explanation: "cos(0) = 1. So lr = lr_min + ½(lr_max−lr_min)(1+1) = lr_max.",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "With step decay (γ=0.1, step=10), what is the LR at epoch 25 if lr_init=0.01?",
      options: ["0.0001", "0.001", "0.01", "0.00001"],
      correctIndex: 0,
      explanation: "epoch//step = 25//10 = 2. lr = 0.01 × 0.1² = 0.01 × 0.01 = 0.0001.",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Why might you use cyclical LR instead of monotonic decay?",
      options: ["To periodically escape saddle points and local minima with high-LR phases", "Cyclical is always better", "To reduce memory usage", "To avoid warmup"],
      correctIndex: 0,
      explanation: "Periodic high-LR phases can push the optimizer out of sharp local minima, potentially finding flatter (better generalizing) minima.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "How many warmup epochs are typical?",
      options: ["5-10 epochs", "50-100 epochs", "1 epoch", "0 — warmup isn't used"],
      correctIndex: 0,
      explanation: "5-10 warmup epochs is standard. Enough to stabilize without wasting training time.",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Does the learning rate schedule matter more than the base learning rate?",
      options: ["No — getting the base LR right matters more than the schedule choice", "Yes — schedule is everything", "They're equally important always", "Schedule matters only for SGD"],
      correctIndex: 0,
      explanation: "A good base LR with no schedule beats a bad LR with a fancy schedule. Fix the LR first, then tune the schedule.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What happens at the end of cosine annealing (t=T)?",
      options: ["LR reaches lr_min", "LR reaches lr_max", "LR is at midpoint", "LR becomes zero"],
      correctIndex: 0,
      explanation: "cos(π) = −1, so lr = lr_min + ½(lr_max−lr_min)(0) = lr_min. The model fine-tunes with minimal updates.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Which schedule is recommended as default for most tasks?",
      options: ["Cosine annealing (with optional warmup)", "Step decay", "Cyclical", "Fixed LR"],
      correctIndex: 0,
      explanation: "Cosine annealing is the modern default. Smooth, simple, and empirically strong across architectures.",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "In warmup + cosine, what happens during the warmup phase?",
      options: ["LR linearly increases from near-zero to lr_max", "LR stays constant", "LR decreases", "LR oscillates"],
      correctIndex: 0,
      explanation: "Linear ramp-up: lr = lr_max × (epoch+1)/warmup_epochs. Then transitions to cosine decay.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What is the cosine annealing LR at exactly the midpoint of training?",
      options: ["The average of lr_max and lr_min", "lr_max", "lr_min", "Zero"],
      correctIndex: 0,
      explanation: "At t=T/2, cos(π/2)=0. lr = lr_min + ½(lr_max−lr_min) = (lr_max+lr_min)/2.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "When should you try a different schedule than cosine annealing?",
      options: ["When you have a specific reason (e.g., cyclical for saddle points, step decay for known convergence pattern)", "Always experiment with all schedules", "Never — cosine is always best", "Only for regression tasks"],
      correctIndex: 0,
      explanation: "Start with cosine. Only switch if diagnostics show a problem (stuck in saddle → cyclical, known milestones → step decay).",
      randomize: true,
    }
  ]}
/>
