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
        "CNNs have significantly more parameters",
        "CNNs exploit spatial locality and translation invariance through weight sharing, making each parameter participate in thousands of local computations",
        "MLPs cannot process image data at all",
        "CNNs automatically use better optimizers"
      ],
      correctIndex: 1,
      explanation: "Weight sharing means one 3×3 kernel detects a feature anywhere in the image. An MLP needs separate weights for each pixel position. CNNs encode the right inductive bias for images — spatial structure matters.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "How many parameters does Conv2d(16→32, 3×3) have including bias?",
      code: "filters, c_in, k = 32, 16, 3\nparams = filters * c_in * k * k + filters\nprint(params)",
      options: ["4640", "4608", "4672", "512"],
      correctIndex: 0,
      explanation: "Weights: 32 × 16 × 3 × 3 = 4608. Bias: 32 (one per filter). Total: 4640. Compare to a dense layer from 14×14×16=3136 to 14×14×32=6272: that would need ~19.7M parameters.",
      randomize: false
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "Input (N, 1, 28, 28) → Conv2d(1→16, 3×3, pad=1) → MaxPool(2×2). What is the output shape?",
      options: ["(N, 16, 14, 14)", "(N, 16, 28, 28)", "(N, 16, 13, 13)", "(N, 1, 14, 14)"],
      correctIndex: 0,
      explanation: "Conv with pad=1 preserves spatial dims: (N, 16, 28, 28). MaxPool(2×2) halves each spatial dim: (N, 16, 14, 14). Channel count changes only via conv filters.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "In the CNN forward pass, why must intermediate values (z1, a1, p1, etc.) be saved as self.xxx?",
      options: [
        "For logging and debugging only",
        "Backpropagation requires these cached values to compute gradients via the chain rule",
        "To enable dropout during inference",
        "NumPy requires explicit variable storage"
      ],
      correctIndex: 1,
      explanation: "Every backward pass needs the forward activations: ReLU gradient needs pre-activation sign, BN needs batch stats, conv needs input for dW and dX. Without caching, you'd have to recompute the entire forward pass.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What is the flattened size after two Conv→Pool blocks starting from (1, 28, 28)?",
      options: ["1568 (32×7×7)", "784 (16×7×7)", "3136 (32×14×14)", "256"],
      correctIndex: 0,
      explanation: "After Conv1+Pool: (16, 14, 14). After Conv2+Pool: (32, 7, 7). Flatten: 32 × 7 × 7 = 1568. This becomes the input dimension for the first fully-connected layer.",
      randomize: true
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What does this He initialization compute for conv1 weights?",
      code: "import numpy as np\nfan_in = 1 * 3 * 3  # c_in * k * k\nstd = np.sqrt(2.0 / fan_in)\nprint(f'{std:.4f}')",
      options: ["0.4714", "0.6667", "0.3333", "1.0000"],
      correctIndex: 0,
      explanation: "He init: std = sqrt(2/fan_in). For conv layers, fan_in = c_in × k × k = 1×9 = 9. sqrt(2/9) ≈ 0.4714. This maintains activation variance across layers when using ReLU.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "Why is batch normalization placed BEFORE ReLU in this CNN architecture?",
      options: [
        "It's arbitrary — either order works equally well",
        "BN normalizes the linear convolution output to zero-mean/unit-variance before the nonlinearity, ensuring ReLU receives well-distributed inputs and reducing dead neurons",
        "ReLU must always come last in a block",
        "BN after ReLU causes gradient explosion"
      ],
      correctIndex: 1,
      explanation: "Pre-ReLU BN ensures the normalized signal enters ReLU centered around zero. This prevents systematic bias toward positive or negative values, keeping roughly half of ReLU units active and improving gradient flow.",
      randomize: true
    },
    {
      id: "q8",
      type: "fill-blank",
      prompt: "The softmax stabilization trick subtracts ___ from logits before exponentiation to prevent overflow.",
      options: ["np.mean(logits)", "np.max(logits, axis=1, keepdims=True)", "np.min(logits)", "0.5"],
      correctIndex: 1,
      explanation: "Subtracting the row-wise max makes the largest logit zero, so exp(max) = 1. All other exp values are ≤ 1, preventing overflow. The math is equivalent since softmax is shift-invariant.",
      randomize: false
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "During inference, how does the dropout layer behave differently from training?",
      options: [
        "It drops different neurons randomly",
        "It passes activations through unchanged — no masking, no scaling",
        "It doubles the dropout rate",
        "It applies the inverse mask from training"
      ],
      correctIndex: 1,
      explanation: "With inverted dropout (divide by 1-p during training), inference simply uses the raw activations. The expected value was already corrected during training, so no adjustment is needed at test time.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Why does the CNN use padding=1 with 3×3 kernels?",
      options: [
        "To increase the output size",
        "To preserve spatial dimensions: output H = (H + 2×1 - 3)/1 + 1 = H. Without padding, each conv shrinks the feature map by 2 pixels.",
        "Padding is required for batch normalization",
        "To add regularization"
      ],
      correctIndex: 1,
      explanation: "With pad=1 and kernel=3: output_size = (input + 2 - 3)/1 + 1 = input. This 'same' padding preserves spatial resolution through conv layers, letting pooling be the sole downsampling mechanism.",
      randomize: true
    },
    {
      id: "q11",
      type: "ordering",
      prompt: "Order the operations in one conv block of this CNN:",
      items: ["Conv2d forward", "Add bias", "Batch normalization", "ReLU activation", "Max pooling"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Standard conv block: Conv → Bias → BN → ReLU → Pool. BN normalizes the linear output before nonlinearity. Pooling comes last to downsample the activated feature map.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "If you remove batch normalization from this CNN, what problem are you most likely to encounter?",
      options: [
        "Overfitting increases dramatically",
        "Training becomes unstable — activations may explode or vanish, requiring careful learning rate tuning and possibly failing to converge",
        "The model can't learn spatial features",
        "Dropout stops working"
      ],
      correctIndex: 1,
      explanation: "BN stabilizes layer inputs by normalizing to zero-mean/unit-variance. Without it, deep networks suffer from internal covariate shift — each layer's input distribution changes as earlier layers update, making optimization fragile.",
      randomize: true
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What is the total parameter count for FC1 (1568→128) including bias?",
      code: "w_params = 1568 * 128\nb_params = 128\nprint(w_params + b_params)",
      options: ["200832", "200704", "1568", "128"],
      correctIndex: 0,
      explanation: "Weights: 1568 × 128 = 200,704. Bias: 128. Total: 200,832. This single FC layer contains ~93% of all CNN parameters — the conv layers are extremely parameter-efficient by comparison.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why are pure NumPy CNNs slow compared to PyTorch/TensorFlow?",
      options: [
        "NumPy doesn't support matrix multiplication",
        "NumPy runs on CPU without GPU acceleration, lacks optimized cuDNN kernels, and has Python overhead for loop-based operations like im2col",
        "NumPy uses single precision only",
        "The algorithm is fundamentally different"
      ],
      correctIndex: 1,
      explanation: "Frameworks use GPU-optimized libraries (cuDNN, MKL) with fused operations. NumPy executes on CPU with generic BLAS. The algorithms are identical — it's purely an implementation efficiency gap.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What happens if you accidentally apply dropout during inference (training=False not set)?",
      options: [
        "Nothing — dropout is always beneficial",
        "Predictions become stochastic and unreliable — different forward passes give different results, and activations are incorrectly scaled",
        "The model runs faster",
        "Accuracy improves due to ensemble effect"
      ],
      correctIndex: 1,
      explanation: "Dropout at test time randomly zeros activations and scales survivors, making outputs noisy and biased. Inference must use deterministic, unscaled activations for consistent predictions.",
      randomize: true
    },
    {
      id: "q16",
      type: "shape-prediction",
      prompt: "After flattening (N, 32, 7, 7) and passing through FC1 (1568→128) + ReLU + Dropout, what shape enters FC2?",
      options: ["(N, 128)", "(N, 1568)", "(N, 10)", "(N, 32, 7, 7)"],
      correctIndex: 0,
      explanation: "Flatten: (N, 1568). FC1: (N, 128). ReLU and dropout preserve shape. FC2 input: (N, 128). Output of FC2: (N, 10) for 10 digit classes.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "The CNN has ~215K parameters and the MLP has ~203K. Why does the CNN perform better despite similar counts?",
      options: [
        "More parameters always means better performance",
        "CNN parameters are shared across spatial positions — each weight participates in thousands of computations, providing massive effective capacity with built-in translation equivariance",
        "The MLP is poorly implemented",
        "Parameter count is the only metric that matters"
      ],
      correctIndex: 1,
      explanation: "A single 3×3 conv filter slides across 26×26=676 positions, reusing its 9 weights each time. That's 676× more computation per parameter than a dense layer. Weight sharing provides enormous computational leverage.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Why use smaller batch sizes (64-128) carefully with batch normalization?",
      options: [
        "Smaller batches train faster",
        "BN computes statistics per-batch. Very small batches give noisy mean/variance estimates, destabilizing normalization. Very large batches reduce the regularizing effect of BN noise.",
        "Batch size doesn't affect BN",
        "Smaller batches prevent overfitting"
      ],
      correctIndex: 1,
      explanation: "BN's running statistics approximate the true distribution. Small batches → high variance estimates → noisy normalization → training instability. The recommended 64-128 range balances statistical quality with regularization benefit.",
      randomize: true
    },
    {
      id: "q19",
      type: "fill-blank",
      prompt: "In the softmax computation, `shifted = logits - np.max(logits, axis=1, keepdims=True)` uses keepdims=True so that the subtraction ___ correctly across columns.",
      options: ["broadcasts", "concatenates", "flattens", "transposes"],
      correctIndex: 0,
      explanation: "Without keepdims, max returns shape (N,) which can't broadcast against (N, 10) logits along axis=1. keepdims=True gives shape (N, 1), enabling proper row-wise broadcasting.",
      randomize: false
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You're debugging this CNN and notice training loss decreases but test accuracy plateaus at 95%. What should you try FIRST?",
      options: [
        "Add more conv layers",
        "Increase learning rate",
        "Add data augmentation and/or spatial dropout to reduce overfitting",
        "Switch to sigmoid activation"
      ],
      correctIndex: 2,
      explanation: "Decreasing train loss + stagnant test accuracy = overfitting. The first response is stronger regularization: augmentation expands training diversity, spatial dropout prevents feature co-adaptation. Architecture changes come later.",
      randomize: true
    }
  ]}
/>
