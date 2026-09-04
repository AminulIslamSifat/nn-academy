---
title: "Building a CNN for MNIST"
slug: "039-cnn-mnist"
description: "Assemble conv layers, pooling, batch norm, and a classifier head into a complete CNN. Train it on MNIST to 99%+ accuracy in pure NumPy."
track: "nn-intermediate"
order: 4
read_time: 30
code_time: 25
execution_timeout: 60
prerequisites: ["036-conv2d-from-scratch", "037-pooling-batchnorm", "038-dropout-data-augmentation"]
---

# Building a CNN for MNIST

Time to assemble everything. ==You're building a real CNN from scratch== that beats the MLP from chapter 032.

## Architecture

```
Input (1, 28, 28)
  → Conv2d(1→16, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   → (16, 14, 14)
  → Conv2d(16→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → (32, 7, 7)
  → Flatten → Linear(32×7×7=1568 → 128) → ReLU → Dropout(0.3)
  → Linear(128 → 10) → Softmax
```

Total parameters: ~215K (vs 203K for the MLP, but far more effective).

## Complete Implementation

```python
import numpy as np

class SimpleCNN:
    def __init__(self):
        # Conv1: 1→16, 3×3
        self.conv1_w = np.random.randn(16, 1, 3, 3) * np.sqrt(2.0 / 9)
        self.conv1_b = np.zeros(16)
        self.bn1_gamma = np.ones(16)
        self.bn1_beta = np.zeros(16)
        
        # Conv2: 16→32, 3×3
        self.conv2_w = np.random.randn(32, 16, 3, 3) * np.sqrt(2.0 / (16*9))
        self.conv2_b = np.zeros(32)
        self.bn2_gamma = np.ones(32)
        self.bn2_beta = np.zeros(32)
        
        # FC1: 1568→128
        self.fc1_w = np.random.randn(1568, 128) * np.sqrt(2.0 / 1568)
        self.fc1_b = np.zeros(128)
        
        # FC2: 128→10
        self.fc2_w = np.random.randn(128, 10) * np.sqrt(2.0 / 128)
        self.fc2_b = np.zeros(10)
```

### Forward Pass

```python
    def forward(self, X, training=True):
        # Conv1 block
        self.X = X
        self.z1 = conv2d_forward(X, self.conv1_w, stride=1, pad=1) + \
                  self.conv1_b.reshape(1, -1, 1, 1)
        self.h1 = batch_norm(self.z1, self.bn1_gamma, self.bn1_beta, training)
        self.a1 = np.maximum(0, self.h1)  # ReLU
        self.p1 = max_pool(self.a1)       # (N, 16, 14, 14)
        
        # Conv2 block
        self.z2 = conv2d_forward(self.p1, self.conv2_w, stride=1, pad=1) + \
                  self.conv2_b.reshape(1, -1, 1, 1)
        self.h2 = batch_norm(self.z2, self.bn2_gamma, self.bn2_beta, training)
        self.a2 = np.maximum(0, self.h2)
        self.p2 = max_pool(self.a2)       # (N, 32, 7, 7)
        
        # Flatten + FC
        N = X.shape[0]
        self.flat = self.p2.reshape(N, -1)  # (N, 1568)
        self.fc1_z = self.flat @ self.fc1_w + self.fc1_b
        self.fc1_a = np.maximum(0, self.fc1_z)
        
        # Dropout
        if training:
            self.drop_mask = (np.random.rand(*self.fc1_a.shape) > 0.3).astype(float)
            self.fc1_d = self.fc1_a * self.drop_mask / 0.7
        else:
            self.fc1_d = self.fc1_a
        
        # Output
        logits = self.fc1_d @ self.fc2_w + self.fc2_b
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs
```

<PyRunner
  cellId="039-cell-1"
  defaultCode={`import numpy as np

# Demonstrate the architecture flow with shapes
N = 4
X = np.random.randn(N, 1, 28, 28)

print("CNN Architecture Flow:")
print(f"  Input:          {X.shape}")

# Conv1
conv1_out_shape = (N, 16, 28, 28)  # pad=1 preserves size
print(f"  Conv1+BN+ReLU:  {conv1_out_shape}")

# Pool1
pool1_shape = (N, 16, 14, 14)
print(f"  MaxPool:        {pool1_shape}")

# Conv2
conv2_out_shape = (N, 32, 14, 14)
print(f"  Conv2+BN+ReLU:  {conv2_out_shape}")

# Pool2
pool2_shape = (N, 32, 7, 7)
print(f"  MaxPool:        {pool2_shape}")

# Flatten
flat_size = 32 * 7 * 7
print(f"  Flatten:        ({N}, {flat_size})")

# FC layers
print(f"  FC1 (1568→128): ({N}, 128)")
print(f"  Dropout(0.3):   ({N}, 128)")
print(f"  FC2 (128→10):   ({N}, 10)")
print(f"  Softmax:        ({N}, 10) → class probabilities")

params = 16*1*9 + 16 + 16*2 + 32*16*9 + 32 + 32*2 + 1568*128 + 128 + 128*10 + 10
print(f"\n📊 Total parameters: {params:,}")
print(f"   vs MLP (784→256→10): 203,530")
print(f"   CNN uses spatial structure → better accuracy with similar params")
`}
/>

## Why CNNs Beat MLPs on Images

| Property | MLP | CNN |
|----------|-----|-----|
| Spatial awareness | ❌ Flattens everything | ✅ Local receptive fields |
| Translation invariance | ❌ Must learn per-position | ✅ Weight sharing |
| Parameters for 28×28→256 | 200,704 | 144 (one 3×3 filter) |
| Test accuracy (MNIST) | ~97% | ~99%+ |

> [!IMPORTANT] The Key Insight
> CNNs don't have MORE parameters than MLPs — they have BETTER parameter efficiency. Each conv weight participates in thousands of local computations across the image. This is why CNNs generalize better despite similar parameter counts.

## Training Tips for CNNs

1. **Use Adam** with lr=0.001 (same as MLP)
2. **Batch size 64–128** — smaller batches make BN noisy
3. **Train longer** — CNNs are slower per epoch but converge to better solutions
4. **Monitor train/test gap** — if gap widens, add dropout or augmentation
5. **NumPy CNNs are slow** — expect minutes per epoch. This is educational, not production.

<Callout type="info" title="Speed Reality Check">
A pure NumPy CNN on MNIST takes ~2-5 minutes per epoch. PyTorch does the same in seconds. The point isn't speed — it's understanding every operation. Once you get it, switch to a framework for real projects.
</Callout>

<Quiz
  chapterSlug="039-cnn-mnist"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does a CNN achieve higher accuracy than an MLP with similar parameter count on MNIST?",
      options: [
        "CNNs have more parameters",
        "CNNs exploit spatial locality and translation invariance through weight sharing, making each parameter more informative",
        "MLPs can't handle images",
        "CNNs use better optimizers"
      ],
      correctIndex: 1,
      explanation: "Weight sharing means one 3×3 kernel detects a feature anywhere in the image. An MLP needs separate weights for each position. CNNs encode the right inductive bias for images.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "How many parameters does a conv layer with 32 filters, 16 input channels, 3×3 kernel have (including bias)?",
      code: "filters, c_in, k = 32, 16, 3\nparams = filters * c_in * k * k + filters\nprint(params)",
      options: ["4640", "4608", "4672", "512"],
      correctIndex: 0,
      explanation: "Weights: 32 × 16 × 3 × 3 = 4608. Bias: 32. Total: 4640. Compare to a dense layer mapping 14×14×16=3136 inputs to 14×14×32=6272 outputs: 3136×6272 ≈ 19.7M parameters!",
      randomize: false,
    }
  ]}
/>
