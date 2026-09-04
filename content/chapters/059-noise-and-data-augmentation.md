---
title: "Noise & Stochastic Augmentation"
slug: "059-noise-and-data-augmentation"
description: "Add Gaussian noise, salt-and-pepper noise, and stochastic transformations to improve model robustness. Understand denoising autoencoders conceptually."
track: "random"
order: 4
read_time: 15
code_time: 10
execution_timeout: 10
prerequisites: ["057-probability-distributions", "038-dropout-data-augmentation"]
---

# Noise & Stochastic Augmentation

Adding controlled noise during training makes models robust. ==A model that works on clean data but fails on slightly noisy input is fragile.== Noise injection is regularization through perturbation.

## Gaussian Noise

Add small random perturbations to inputs:

```python
def add_gaussian_noise(X, sigma=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return X + rng.normal(0, sigma, X.shape)
```

<PyRunner
  cellId="059-cell-1"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

# Simulate a clean signal
x = np.linspace(0, 2*np.pi, 50)
clean = np.sin(x)

# Add different noise levels
for sigma in [0.05, 0.1, 0.3]:
    noisy = clean + rng.normal(0, sigma, len(clean))
    snr = 10 * np.log10(np.var(clean) / np.var(noisy - clean))
    print(f"  σ={sigma:.2f}: SNR={snr:.1f} dB, max deviation={np.max(np.abs(noisy-clean)):.3f}")

print(f"\n💡 Small σ (0.01-0.1) acts as regularizer")
print(f"   Large σ forces the model to learn robust features")
`}
/>

## Salt-and-Pepper Noise

Randomly set pixels to 0 or 1:

```python
def salt_pepper(X, prob=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(X.shape)
    out = X.copy()
    out[mask < prob/2] = 0    # pepper
    out[mask > 1-prob/2] = 1  # salt
    return out
```

## Denoising Autoencoder Concept

Train a network to reconstruct clean input from corrupted input:

```
Clean X → Add Noise → Corrupted X̃ → Encoder → Decoder → Reconstructed X̂
Loss = ||X - X̂||²
```

The network learns to ignore noise and capture essential structure.

> [!IMPORTANT] Noise as Regularization
> Adding noise during training is equivalent to Tikhonov regularization under certain conditions. It smooths the decision boundary and prevents overfitting to exact training values.

<Quiz
  chapterSlug="059-noise-and-data-augmentation"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does adding noise during training improve generalization?",
      options: ["It makes training faster", "It prevents the model from memorizing exact training values, forcing it to learn robust features that survive perturbation", "It increases model capacity", "It replaces the need for validation data"],
      correctIndex: 1,
      explanation: "Noise creates infinite virtual training examples around each real example. The model can't memorize exact values because they're constantly perturbed. It must learn the underlying pattern.",
      randomize: true,
    }
  ]}
/>
