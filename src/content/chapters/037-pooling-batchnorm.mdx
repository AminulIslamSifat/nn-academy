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
      prompt: "In max pool backward, why do gradients only flow through max positions?",
      options: [
        "For computational efficiency",
        "Because max is non-differentiable at non-max positions — changing those values doesn't change the output",
        "It's an approximation",
        "To act as regularization"
      ],
      correctIndex: 1,
      explanation: "max(a,b,c,d) only depends on the largest value. Perturbing non-max values doesn't change the output, so their partial derivatives are zero. The gradient routes entirely through whichever position was selected.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does batch norm use running mean/variance at inference instead of batch statistics?",
      options: [
        "Batch statistics are always wrong",
        "At inference, you may have a single sample (batch size 1), making batch statistics meaningless. Running averages provide stable estimates from training.",
        "Running stats are faster to compute",
        "It prevents overfitting"
      ],
      correctIndex: 1,
      explanation: "With batch_size=1, variance is zero and mean equals the single sample. Running statistics accumulated over training provide meaningful normalization regardless of inference batch size.",
      randomize: false,
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "After batch norm with γ=2, β=1, what's the output for a normalized value of 0.5?",
      code: "gamma, beta, x_norm = 2.0, 1.0, 0.5\nout = gamma * x_norm + beta\nprint(f'{out:.1f}')",
      options: ["2.0", "1.5", "3.0", "0.5"],
      correctIndex: 0,
      explanation: "y = γ·x̂ + β = 2 × 0.5 + 1 = 2.0. The learnable γ and β allow the network to scale and shift the normalized distribution as needed.",
      randomize: false,
    }
  ]}
/>
