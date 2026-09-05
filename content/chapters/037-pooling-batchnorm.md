---
title: "Pooling & Batch Normalization"
slug: "037-pooling-batchnorm"
description: "Downsample spatial dimensions and stabilize training. Implement max pooling, average pooling, and batch normalization with forward and backward passes in NumPy."
track: "nn-intermediate"
order: 2
read_time: 25
code_time: 20
execution_timeout: 15
prerequisites: ["036-conv2d-from-scratch"]
---

# Pooling & Batch Normalization

Two essential CNN building blocks. ==Pooling reduces spatial size== (less computation, more invariance). ==Batch norm stabilizes training== (faster convergence, less sensitivity to initialization).

## Max Pooling

Take the maximum value in each window. Captures the strongest activation:

```python
import numpy as np

def max_pool_forward(X, pool_size=2, stride=2):
    """X: (N, C, H, W) → output: (N, C, H//stride, W//stride)"""
    N, C, H, W = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    output = np.zeros((N, C, out_H, out_W))
    mask = np.zeros_like(X)  # remember which positions were max
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            window = X[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[:, :, i, j] = np.max(window, axis=(2, 3))
            
            # Store mask for backward pass
            max_vals = output[:, :, i:i+1, j:j+1]
            mask[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] += (
                window == max_vals
            ).astype(float)
    
    return output, mask
```

<PyRunner
  cellId="037-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)
X = np.array([[[[1, 3, 2, 4],
                 [5, 1, 6, 2],
                 [3, 7, 1, 8],
                 [2, 4, 9, 1]]]])  # (1, 1, 4, 4)

def max_pool(X, ps=2, s=2):
    N, C, H, W = X.shape
    oH = (H-ps)//s+1; oW = (W-ps)//s+1
    out = np.zeros((N,C,oH,oW))
    for i in range(oH):
        for j in range(oW):
            hs, ws = i*s, j*s
            out[:,:,i,j] = np.max(X[:,:,hs:hs+ps,ws:ws+ps], axis=(2,3))
    return out

result = max_pool(X)
print("Input (4×4):")
print(X[0,0])
print(f"\nMax Pool 2×2, stride 2 → Output (2×2):")
print(result[0,0])
print(f"\n💡 Each 2×2 region → its maximum value")
print(f"   Spatial size halved, channels unchanged")
`}
/>

### Max Pool Backward

Gradient flows ONLY through the positions that were selected as max:

```python
def max_pool_backward(dout, mask, input_shape, pool_size=2, stride=2):
    """Route gradients back through the max positions."""
    N, C, H, W = input_shape
    dX = np.zeros(input_shape)
    _, _, out_H, out_W = dout.shape
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            # Distribute gradient only to max positions
            dX[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size] += (
                dout[:, :, i:i+1, j:j+1] * 
                mask[:, :, h_start:h_start+pool_size, w_start:w_start+pool_size]
            )
    return dX
```

> [!NOTE] Average Pooling
> Simpler than max pool: every position in the window gets equal gradient <InlineMath latex="\frac{1}{k^2}" />. Less commonly used in modern architectures except for global average pooling before the classifier.

## Batch Normalization

The problem: as earlier layers update, the distribution of their outputs shifts. Later layers constantly chase a moving target. ==Batch norm fixes this by normalizing each layer's inputs to zero mean and unit variance==, then applying learnable scale and shift:

<BlockMath latex="\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta" />

where <InlineMath latex="\mu_B" /> and <InlineMath latex="\sigma_B^2" /> are computed per-batch, and <InlineMath latex="\gamma, \beta" /> are learned parameters.

```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        
        # Running stats for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward
        self.cache = None
    
    def forward(self, X, training=True):
        if training:
            # For dense layers: X is (N, D), normalize over axis=0
            # For conv layers: X is (N, C, H, W), normalize over axis=(0,2,3)
            if X.ndim == 2:
                mu = np.mean(X, axis=0)
                var = np.var(X, axis=0)
                self.cache = (X, mu, var)
            else:  # conv
                mu = np.mean(X, axis=(0, 2, 3), keepdims=True)
                var = np.var(X, axis=(0, 2, 3), keepdims=True)
                self.cache = (X, mu, var)
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.ravel()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.ravel()
        else:
            mu = self.running_mean
            var = self.running_var
            if X.ndim == 4:
                mu = mu.reshape(1, -1, 1, 1)
                var = var.reshape(1, -1, 1, 1)
        
        x_norm = (X - mu) / np.sqrt(var + self.eps)
        
        if X.ndim == 4:
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)
        else:
            gamma, beta = self.gamma, self.beta
        
        return gamma * x_norm + beta
```

<PyRunner
  cellId="037-cell-2"
  defaultCode={`import numpy as np

np.random.seed(42)

# Simulate activations with weird distribution
X = np.random.randn(64, 128) * 5 + 10  # mean=10, std=5

print("Before Batch Norm:")
print(f"  Mean: {X.mean(axis=0).mean():.2f}")
print(f"  Std:  {X.std(axis=0).mean():.2f}")

# Apply batch norm
mu = X.mean(axis=0)
var = X.var(axis=0)
x_norm = (X - mu) / np.sqrt(var + 1e-5)
gamma = np.ones(128)
beta = np.zeros(128)
out = gamma * x_norm + beta

print(f"\nAfter Batch Norm:")
print(f"  Mean: {out.mean(axis=0).mean():.6f} ≈ 0")
print(f"  Std:  {out.std(axis=0).mean():.4f} ≈ 1")

print(f"\n✅ Every feature now has ~zero mean, ~unit variance")
print(f"   γ and β let the network UNDO normalization if needed")
print(f"   This is what makes BN so powerful — it's adaptive")
`}
/>

### Why Batch Norm Works

1. **Reduces internal covariate shift** — layer inputs stay stable
2. **Allows higher learning rates** — normalized inputs prevent explosion
3. **Acts as mild regularizer** — batch statistics add noise
4. **Reduces sensitivity to initialization** — normalization compensates for bad init

<Callout type="warning" title="Training vs Inference">
During training: use batch statistics. During inference: use running mean/variance accumulated during training. Forgetting this distinction is the #1 batch norm bug.
</Callout>

### Batch Norm Placement

Convention: ==Conv → BN → ReLU== (not Conv → ReLU → BN). Normalizing before the nonlinearity keeps the pre-activation distribution stable.

## Combined: Conv Block Pattern

The standard building block of modern CNNs:

```python
def conv_block(X, kernels, bn, stride=1, pad=1, training=True):
    """Conv → BatchNorm → ReLU"""
    # Conv forward (from chapter 036)
    conv_out = conv2d_forward(X, kernels, stride=stride, pad=pad)
    # Batch norm
    bn_out = bn.forward(conv_out, training=training)
    # ReLU
    relu_out = np.maximum(0, bn_out)
    return relu_out
```

> [!IMPORTANT] Key Takeaways
> - **Max pooling**: downsamples spatially, routes gradients through max positions only
> - **Batch norm**: normalizes layer inputs, dramatically improves training stability
> - **Always use BN** in modern CNNs — there's almost no reason not to
> - **Global average pooling** before classifier replaces flatten+dense in many architectures

<Quiz
  chapterSlug="037-pooling-batchnorm"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does max pooling do?",
      options: ["Takes the maximum value in each window, downsampling spatial dimensions while keeping channels unchanged", "Averages all values in a window", "Selects random values", "Upsamples the feature map"],
      correctIndex: 0,
      explanation: "Max pool reduces H×W by the stride factor. Each output pixel = max of a local region. Captures strongest activation.",
      randomize: true,
    },
    {
      id: "q2",
      type: "shape-prediction",
      prompt: "Input (N, C, 28, 28), max pool 2×2, stride 2. Output shape?",
      options: ["(N, C, 14, 14)", "(N, C, 28, 28)", "(N, C*4, 14, 14)", "(N, C, 26, 26)"],
      correctIndex: 0,
      explanation: "Pooling halves spatial dims: 28/2 = 14. Channels stay the same. No learnable parameters.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In max pool backward, why do gradients only flow through max positions?",
      options: ["Changing non-max values doesn't change the output, so their partial derivatives are zero", "For computational efficiency only", "It's an approximation", "To act as regularization"],
      correctIndex: 0,
      explanation: "max(a,b,c,d) depends only on the largest value. Non-max positions have zero gradient. The saved mask routes gradients back correctly.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What must you save during max pool forward for the backward pass?",
      options: ["A mask indicating which positions were the max in each window", "The entire input", "Nothing — can recompute", "Only the output values"],
      correctIndex: 0,
      explanation: "The mask records where the max came from. Backward uses it to route dout only to those positions.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "How does average pooling backward differ from max pooling backward?",
      options: ["Every position in the window gets equal gradient (1/k²), not just the max position", "Same as max pool", "No gradient flows", "Gradient is doubled"],
      correctIndex: 0,
      explanation: "Average pool's gradient distributes equally: each input in the window contributed 1/k² to the output. All positions get gradient.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What problem does batch normalization solve?",
      options: ["Internal covariate shift — layer input distributions change as earlier layers update, destabilizing training", "Overfitting", "Vanishing gradients only", "Slow computation"],
      correctIndex: 0,
      explanation: "As weights update, each layer's input distribution shifts. Later layers constantly chase a moving target. BN stabilizes this.",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What does batch norm do to activations?",
      options: ["Normalizes to zero mean and unit variance per feature, then applies learnable scale (γ) and shift (β)", "Clips values to [-1, 1]", "Applies dropout", "Computes softmax"],
      correctIndex: 0,
      explanation: "x̂ = (x−μ)/√(σ²+ε), then y = γx̂ + β. γ and β let the network undo normalization if optimal.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the output?",
      code: "gamma, beta, x_norm = 2.0, 1.0, 0.5\nout = gamma * x_norm + beta\nprint(f'{out:.1f}')",
      options: ["2.0", "1.5", "3.0", "0.5"],
      correctIndex: 0,
      explanation: "y = γ·x̂ + β = 2×0.5 + 1 = 2.0. Learnable γ and β allow adaptive scaling/shifting.",
      randomize: false,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why does batch norm use running mean/variance at inference?",
      options: ["At inference, batch size may be 1, making batch statistics meaningless; running averages provide stable estimates", "Batch stats are always wrong", "Running stats are faster", "Prevents overfitting"],
      correctIndex: 0,
      explanation: "Single-sample batches have zero variance. Running stats accumulated during training give meaningful normalization at any batch size.",
      randomize: false,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What are γ (gamma) and β (beta) in batch norm?",
      options: ["Learnable parameters that scale and shift the normalized output, allowing the network to undo normalization if needed", "Fixed constants", "Learning rate parameters", "Regularization terms"],
      correctIndex: 0,
      explanation: "Without γ and β, BN would force all activations to N(0,1), which may not be optimal. These parameters restore representational power.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Where should batch norm be placed relative to activation?",
      options: ["Conv → BN → ReLU (normalize before nonlinearity)", "Conv → ReLU → BN", "BN → Conv → ReLU", "Doesn't matter"],
      correctIndex: 0,
      explanation: "Normalizing before ReLU keeps pre-activation distributions stable. This is the standard convention in modern architectures.",
      randomize: true,
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why does batch norm allow higher learning rates?",
      options: ["Normalized inputs prevent activation explosion and keep gradients well-scaled", "BN increases the effective batch size", "BN removes the need for backprop", "It doesn't — LR stays the same"],
      correctIndex: 0,
      explanation: "Without BN, large LR causes activations to grow/shrink wildly across layers. BN keeps them in a stable range, tolerating larger steps.",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "How does batch norm act as a mild regularizer?",
      options: ["Batch statistics add noise (different μ, σ per batch), acting like stochastic regularization similar to dropout", "It explicitly penalizes large weights", "It adds L2 penalty", "It doesn't regularize"],
      correctIndex: 0,
      explanation: "Each batch has slightly different mean/variance. This noise prevents the network from fitting too tightly to training data.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What is the #1 batch norm bug?",
      options: ["Using batch statistics during inference instead of running mean/variance", "Wrong learning rate", "Missing ReLU", "Wrong padding"],
      correctIndex: 0,
      explanation: "Training mode uses batch stats; inference mode uses running stats. Forgetting to switch causes terrible test performance.",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "For conv layers, which axes does batch norm normalize over?",
      options: ["axis=(0, 2, 3) — across batch and spatial dims, per channel", "axis=0 only", "axis=1 only", "All axes"],
      correctIndex: 0,
      explanation: "Per-channel normalization: compute μ and σ across all samples and spatial positions for each channel independently.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What is global average pooling?",
      options: ["Average over entire spatial dims (H×W) per channel, replacing flatten+dense before classifier", "Max over spatial dims", "Average over channels", "Average over batch"],
      correctIndex: 0,
      explanation: "GAP reduces (N,C,H,W) → (N,C). Fewer parameters than flatten+dense, acts as structural regularizer. Used in ResNet, EfficientNet.",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Does max pooling have any learnable parameters?",
      options: ["No — it's a fixed operation with no weights", "Yes — one weight per window", "Yes — shared weights like convolution", "Depends on implementation"],
      correctIndex: 0,
      explanation: "Pooling is parameter-free. It's a deterministic downsampling operation. No gradient w.r.t. pooling itself.",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Why does batch norm reduce sensitivity to initialization?",
      options: ["Normalization compensates for bad initial weight scales by forcing unit variance", "BN reinitializes weights each epoch", "BN uses Xavier init internally", "It doesn't affect initialization"],
      correctIndex: 0,
      explanation: "Even if weights produce activations with mean=100 or std=0.01, BN normalizes to ~N(0,1). Training becomes robust to init choices.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What happens to batch norm's running stats momentum parameter?",
      options: ["Controls how quickly running mean/var tracks batch stats: running = (1−m)×running + m×batch", "Controls learning rate", "Controls dropout probability", "Not used"],
      correctIndex: 0,
      explanation: "momentum=0.1 means running stats update slowly (stable). Higher momentum = faster tracking but noisier estimates.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Should you always use batch norm in modern CNNs?",
      options: ["Yes — there's almost no reason not to; it improves training stability and convergence speed", "No — it slows down inference", "Only for very deep networks", "Only with SGD, not Adam"],
      correctIndex: 0,
      explanation: "BN is standard in virtually all modern CNN architectures. Benefits far outweigh the small overhead.",
      randomize: true,
    }
  ]}
/>
