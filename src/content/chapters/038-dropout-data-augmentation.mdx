---
title: "Dropout & Data Augmentation for CNNs"
slug: "038-dropout-data-augmentation"
description: "Spatial dropout, random crops, flips, and color jitter. Prevent CNN overfitting with proper regularization techniques implemented in NumPy."
track: "nn-intermediate"
order: 3
read_time: 18
code_time: 15
execution_timeout: 10
prerequisites: ["033-regularization", "037-pooling-batchnorm"]
---

# Dropout & Data Augmentation for CNNs

Standard dropout on conv feature maps is wasteful — it zeros individual pixels independently, but adjacent pixels are highly correlated. ==Spatial dropout drops entire channels==, forcing the network to not rely on any single feature map.

## Spatial Dropout

```python
import numpy as np

def spatial_dropout(X, p_drop, training=True):
    """X: (N, C, H, W). Drop entire channels, not individual pixels."""
    if not training or p_drop == 0:
        return X, None
    N, C, H, W = X.shape
    # One mask per channel per sample, broadcast across spatial dims
    mask = (np.random.rand(N, C, 1, 1) > p_drop).astype(float)
    return X * mask / (1.0 - p_drop), mask
```

<PyRunner
  cellId="038-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

X = np.ones((1, 4, 3, 3)) * 2.0  # 4 channels, all value 2
p = 0.5
mask = (np.random.rand(1, 4, 1, 1) > p).astype(float)
out = X * mask / (1 - p)

print("Spatial Dropout (p=0.5):")
for c in range(4):
    status = "ACTIVE" if mask[0,c,0,0] else "DROPPED"
    print(f"  Channel {c}: {status} → mean={out[0,c].mean():.1f}")

print(f"\n💡 Entire channels dropped, not individual pixels")
print(f"   Standard dropout would zero random pixels within each channel")
print(f"   Spatial dropout forces learning redundant FEATURE MAPS")
`}
/>

## Data Augmentation

Create new training samples by transforming existing ones. ==Free data, massive regularization.==

### Random Horizontal Flip

```python
def random_hflip(X, p=0.5):
    """X: (N, C, H, W) or (C, H, W)"""
    if np.random.rand() < p:
        return X[:, :, :, ::-1] if X.ndim == 4 else X[:, :, ::-1]
    return X
```

### Random Crop

```python
def random_crop(X, crop_size, padding=4):
    """Pad then randomly crop. X: (N, C, H, W)"""
    N, C, H, W = X.shape
    X_padded = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='reflect')
    crops = np.zeros((N, C, crop_size, crop_size))
    for n in range(N):
        top = np.random.randint(0, 2*padding + 1)
        left = np.random.randint(0, 2*padding + 1)
        crops[n] = X_padded[n, :, top:top+crop_size, left:left+crop_size]
    return crops
```

<PyRunner
  cellId="038-cell-2"
  defaultCode={`import numpy as np
np.random.seed(42)

# Create a simple test image
img = np.zeros((1, 1, 8, 8))
img[0, 0, 2:6, 2:6] = 1.0  # centered square

# Random crop with padding
def random_crop(X, crop_size, pad=2):
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='reflect')
    out = np.zeros((N, C, crop_size, crop_size))
    for n in range(N):
        t = np.random.randint(0, 2*pad+1)
        l = np.random.randint(0, 2*pad+1)
        out[n] = Xp[n, :, t:t+crop_size, l:l+crop_size]
    return out

crops = [random_crop(img, 8) for _ in range(4)]

print("Original 8×8 image (centered square):")
for row in img[0,0].astype(int):
    print("  " + "".join("█" if v else "·" for v in row))

print(f"\n4 random crops (pad=2, crop=8):")
for i, c in enumerate(crops):
    filled = int(c[0,0].sum())
    print(f"  Crop {i}: filled_pixels={filled}, square shifted")

print(f"\n✅ Each crop is a valid training sample")
print(f"   Network learns position-invariant features")
`}
/>

### Color Jitter (for RGB images)

```python
def color_jitter(X, brightness=0.2, contrast=0.2):
    """X: (N, 3, H, W), values in [0, 1]"""
    out = X.copy()
    # Brightness
    b = np.random.uniform(-brightness, brightness)
    out += b
    # Contrast
    c = np.random.uniform(1 - contrast, 1 + contrast)
    out *= c
    return np.clip(out, 0, 1)
```

> [!IMPORTANT] Augmentation Rules
> - Only augment **training** data, never test/validation
> - Choose transforms that make physical sense (don't flip digits upside down for MNIST)
> - Apply augmentations **on-the-fly** during training, don't pre-generate
> - Start conservative, increase if still overfitting

## Augmentation Pipeline

```python
def augment_batch(X, y, training=True):
    if not training:
        return X, y
    X = random_hflip(X, p=0.5)
    X = random_crop(X, crop_size=X.shape[-1], padding=4)
    if X.shape[1] == 3:  # RGB only
        X = color_jitter(X)
    return X, y
```

<Quiz
  chapterSlug="038-dropout-data-augmentation"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is spatial dropout better than standard dropout for CNNs?",
      options: [
        "It's faster to compute",
        "Adjacent pixels in feature maps are correlated; dropping individual pixels doesn't force redundancy. Dropping entire channels forces the network to learn backup feature detectors.",
        "It works better with batch normalization",
        "Standard dropout doesn't work with convolutions"
      ],
      correctIndex: 1,
      explanation: "In conv feature maps, nearby activations encode similar information. Standard dropout zeros scattered pixels that neighboring pixels compensate for. Spatial dropout removes entire feature maps, forcing genuine redundancy.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Should you apply data augmentation to validation/test data?",
      options: ["Yes, always", "Only to validation", "Never — evaluate on clean, unmodified data", "Only horizontal flips"],
      correctIndex: 2,
      explanation: "Augmentation creates artificial training diversity. Evaluation must measure real-world performance on unmodified data. Augmenting test data inflates or deflates metrics unpredictably.",
      randomize: false,
    }
  ]}
/>
