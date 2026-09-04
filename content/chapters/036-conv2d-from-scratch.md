---
title: "Convolutional Neural Networks from Scratch"
slug: "036-conv2d-from-scratch"
description: "MLPs ignore spatial structure. Build 2D convolution, stride, padding, and backprop through conv layers entirely in NumPy."
track: "nn-intermediate"
order: 1
read_time: 35
code_time: 30
execution_timeout: 30
prerequisites: ["030-backpropagation", "032-mnist-from-scratch"]
---

# Convolutional Neural Networks from Scratch

Your MNIST MLP treats each pixel independently. It doesn't know that pixel (10,10) is next to pixel (10,11). ==CNNs exploit spatial structure== by sharing weights across locations and detecting local patterns.

## What Is Convolution?

A small kernel slides across the image, computing dot products at every position:

<BlockMath latex="(I * K)_{i,j} = \sum_m \sum_n I_{i+m, j+n} \cdot K_{m,n}" />

The same kernel detects the same feature everywhere. A 3×3 edge detector finds edges whether they're in the top-left or bottom-right corner.

```python
import numpy as np

def conv2d_forward(X, kernel):
    """Simple 2D convolution, no padding, stride=1.
    X: (H, W), kernel: (kH, kW) → output: (H-kH+1, W-kW+1)
    """
    H, W = X.shape
    kH, kW = kernel.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = np.sum(X[i:i+kH, j:j+kW] * kernel)
    return output
```

<PyRunner
  cellId="036-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)

# Create a simple image with a vertical edge
img = np.zeros((8, 8))
img[:, 4:] = 1.0  # right half is white

# Vertical edge detector kernel
kernel = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

def conv2d(X, K):
    H, W = X.shape
    kH, kW = K.shape
    out = np.zeros((H-kH+1, W-kW+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = np.sum(X[i:i+kH, j:j+kW] * K)
    return out

result = conv2d(img, kernel)

print("Input image (vertical edge at column 4):")
for row in img.astype(int):
    print("  " + "".join("█" if v else "·" for v in row))

print(f"\nKernel (vertical edge detector):")
print(kernel)

print(f"\nConvolution output ({result.shape}):")
print(np.round(result, 1))
print(f"\n💡 Strong response at the edge location, zero elsewhere!")
`}
/>

## Multi-Channel Convolution

Real images have channels (RGB = 3, grayscale = 1). Each filter has depth matching the input:

<BlockMath latex="\text{output}_{i,j} = \sum_c \sum_m \sum_n X_{c, i+m, j+n} \cdot K_{c, m, n}" />

```python
def conv2d_multi_channel(X, kernels, stride=1, pad=0):
    """X: (C_in, H, W), kernels: (num_filters, C_in, kH, kW)
    Returns: (num_filters, out_H, out_W)
    """
    C_in, H, W = X.shape
    num_filters, _, kH, kW = kernels.shape
    
    # Pad input
    if pad > 0:
        X_padded = np.pad(X, ((0,0), (pad,pad), (pad,pad)), mode='constant')
    else:
        X_padded = X
    
    _, H_pad, W_pad = X_padded.shape
    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1
    
    output = np.zeros((num_filters, out_H, out_W))
    for f in range(num_filters):
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * stride
                w_start = j * stride
                patch = X_padded[:, h_start:h_start+kH, w_start:w_start+kW]
                output[f, i, j] = np.sum(patch * kernels[f])
    return output
```

<PyRunner
  cellId="036-cell-2"
  defaultCode={`import numpy as np

np.random.seed(42)

# Simulate a batch of 1 grayscale 8x8 image
X = np.random.rand(1, 8, 8)

# 2 filters, each 1x3x3
kernels = np.random.randn(2, 1, 3, 3) * 0.5

def conv2d_mc(X, K, stride=1, pad=0):
    C, H, W = X.shape
    nf, _, kH, kW = K.shape
    if pad > 0:
        X = np.pad(X, ((0,0),(pad,pad),(pad,pad)))
    _, Hp, Wp = X.shape
    oH = (Hp - kH)//stride + 1
    oW = (Wp - kW)//stride + 1
    out = np.zeros((nf, oH, oW))
    for f in range(nf):
        for i in range(oH):
            for j in range(oW):
                hs, ws = i*stride, j*stride
                out[f,i,j] = np.sum(X[:, hs:hs+kH, ws:ws+kW] * K[f])
    return out

out = conv2d_mc(X, kernels, pad=1)
print(f"Input shape:  {X.shape}  (channels, height, width)")
print(f"Kernels shape: {kernels.shape}  (filters, channels, kH, kW)")
print(f"Output shape:  {out.shape}  (filters, out_H, out_W)")
print(f"\nWith padding=1: output spatial size matches input ✅")
print(f"Without padding: output shrinks by (kH-1, kW-1)")
`}
/>

## Padding & Stride

Two knobs that control output size:

| Parameter | Effect | Formula |
|-----------|--------|---------|
| **Padding** | Adds zeros around border, preserves spatial size | <InlineMath latex="\text{out} = \lfloor(H + 2p - k)/s\rfloor + 1" /> |
| **Stride** | Skips positions, reduces output size | Same formula above |

> [!IMPORTANT] "Same" Padding
> To keep output size equal to input size with stride=1: `pad = (kernel_size - 1) // 2`. For 3×3 kernel: pad=1. For 5×5: pad=2.

## Vectorized Convolution (im2col)

The loop-based version above is correct but ==painfully slow==. The trick: reshape all patches into columns, then use matrix multiplication.

```python
def im2col(X, kH, kW, stride=1, pad=0):
    """Convert image patches to columns for vectorized convolution.
    X: (N, C, H, W) → cols: (N, C*kH*kW, out_H*out_W)
    """
    N, C, H, W = X.shape
    if pad > 0:
        X = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)))
    _, _, H_pad, W_pad = X.shape
    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1
    
    cols = np.zeros((N, C * kH * kW, out_H * out_W))
    for n in range(N):
        idx = 0
        for i in range(out_H):
            for j in range(out_W):
                hs, ws = i * stride, j * stride
                patch = X[n, :, hs:hs+kH, ws:ws+kW].ravel()
                cols[n, :, idx] = patch
                idx += 1
    return cols, out_H, out_W
```

<PyRunner
  cellId="036-cell-3"
  defaultCode={`import numpy as np

np.random.seed(42)
N, C, H, W = 2, 1, 8, 8
kH, kW = 3, 3
pad, stride = 1, 1

X = np.random.randn(N, C, H, W)

def im2col(X, kH, kW, stride=1, pad=0):
    N, C, H, W = X.shape
    if pad > 0:
        X = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)))
    _, _, Hp, Wp = X.shape
    oH = (Hp-kH)//stride+1
    oW = (Wp-kW)//stride+1
    cols = np.zeros((N, C*kH*kW, oH*oW))
    for n in range(N):
        idx = 0
        for i in range(oH):
            for j in range(oW):
                hs, ws = i*stride, j*stride
                cols[n, :, idx] = X[n, :, hs:hs+kH, ws:ws+kW].ravel()
                idx += 1
    return cols, oH, oW

cols, oH, oW = im2col(X, kH, kW, stride, pad)

# Now convolution is just matrix multiplication!
num_filters = 4
kernels = np.random.randn(num_filters, C*kH*kW) * 0.5

# (N, F, oH*oW) = (N, C*kH*kW, oH*oW).T @ (F, C*kH*kW).T ... reshaped
out = np.zeros((N, num_filters, oH, oW))
for n in range(N):
    out[n] = (kernels @ cols[n]).reshape(num_filters, oH, oW)

print(f"im2col transforms patches → columns")
print(f"  Input:   {X.shape}")
print(f"  Columns: {cols.shape}  (each column = one 3×3 patch)")
print(f"  Output:  {out.shape}")
print(f"\n✅ Convolution = matrix multiply after im2col")
print(f"   This is how real frameworks implement it!")
`}
/>

## Backprop Through Convolution

The key insight: if forward uses im2col + matmul, backward uses the same tools in reverse.

Given <InlineMath latex="\frac{\partial L}{\partial \text{output}}" />:

1. **Gradient w.r.t. kernels**: <InlineMath latex="\frac{\partial L}{\partial K} = \frac{\partial L}{\partial \text{out}} \cdot \text{cols}^T" />
2. **Gradient w.r.t. input**: col2im of <InlineMath latex="K^T \cdot \frac{\partial L}{\partial \text{out}}" />

```python
def conv2d_backward(dout, X_col, kernels, X_shape, kH, kW, stride, pad):
    """
    dout: (N, F, out_H, out_W)
    X_col: from im2col forward
    Returns: dX (same shape as original X), dkernels
    """
    N, F, out_H, out_W = dout.shape
    _, C_kk, _ = X_col.shape
    
    # Reshape dout for matmul
    dout_reshaped = dout.reshape(N, F, -1)  # (N, F, out_H*out_W)
    
    # Gradient for kernels
    dkernels = np.zeros((F, C_kk))
    for n in range(N):
        dkernels += dout_reshaped[n] @ X_col[n].T
    
    # Gradient for input (via col2im)
    dX_col = np.zeros_like(X_col)
    for n in range(N):
        dX_col[n] = kernels.T @ dout_reshaped[n]
    
    # col2im: scatter gradients back to image positions
    _, C, H, W = X_shape
    dX = np.zeros((N, C, H + 2*pad, W + 2*pad))
    oH_out = (H + 2*pad - kH) // stride + 1
    oW_out = (W + 2*pad - kW) // stride + 1
    
    for n in range(N):
        idx = 0
        for i in range(oH_out):
            for j in range(oW_out):
                hs, ws = i * stride, j * stride
                dX[n, :, hs:hs+kH, ws:ws+kW] += dX_col[n, :, idx].reshape(C, kH, kW)
                idx += 1
    
    if pad > 0:
        dX = dX[:, :, pad:-pad, pad:-pad]
    
    return dX, dkernels
```

> [!IMPORTANT] Shape Discipline
> Conv backprop is where most bugs happen. Always verify:
> - `dkernels` shape == `kernels` shape
> - `dX` shape == original `X` shape (before padding)
> - Print shapes at every step during debugging

## Putting It Together: Simple CNN

```
Input (1, 28, 28) → Conv2d(8 filters, 3×3, pad=1) → ReLU → MaxPool(2×2)
                  → Conv2d(16 filters, 3×3, pad=1) → ReLU → MaxPool(2×2)
                  → Flatten → Linear(256) → ReLU → Linear(10) → Softmax
```

This architecture should reach ~99% on MNIST, beating the MLP from chapter 032.

<Quiz
  chapterSlug="036-conv2d-from-scratch"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why do CNNs use weight sharing (same kernel across all positions)?",
      options: [
        "To reduce computation time only",
        "Translation invariance: a feature detected in one location should be detectable everywhere, plus massive parameter reduction",
        "Because GPUs require it",
        "To prevent overfitting"
      ],
      correctIndex: 1,
      explanation: "Weight sharing means the network learns WHAT to detect, not WHERE. An edge detector trained on the top-left also works on the bottom-right. Plus, a 3×3 kernel has 9 params regardless of image size vs. millions for a fully connected layer.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What's the output size for a 28×28 input with 3×3 kernel, stride=1, padding=1?",
      code: "H, k, s, p = 28, 3, 1, 1\nout = (H + 2*p - k) // s + 1\nprint(out)",
      options: ["28", "26", "27", "30"],
      correctIndex: 0,
      explanation: "(28 + 2×1 - 3) / 1 + 1 = 28. With 'same' padding (p = (k-1)//2 = 1 for k=3), output size equals input size.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does im2col accomplish?",
      options: [
        "Compresses the image for storage",
        "Converts convolution into matrix multiplication by extracting all patches as columns, enabling vectorized GPU-friendly computation",
        "Applies pooling automatically",
        "Normalizes the input"
      ],
      correctIndex: 1,
      explanation: "im2col rearranges overlapping patches into columns of a matrix. Then conv = matmul, which is highly optimized on modern hardware. The tradeoff is memory (patches overlap, so data is duplicated).",
      randomize: true,
    }
  ]}
/>
