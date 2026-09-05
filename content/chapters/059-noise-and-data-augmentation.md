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
      prompt: "What is the purpose of adding noise during training?",
      options: ["To make training faster", "Regularization through perturbation — prevents memorization and forces learning of robust features that generalize beyond exact training values", "To increase model capacity", "To replace validation data"],
      correctIndex: 1,
      explanation: "Noise creates virtual training examples around each real sample. The model can't overfit to exact values when they're constantly perturbed.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does this function return?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nX = np.ones((2, 3)) * 5.0\nnoise = rng.normal(0, 0.1, X.shape)\nresult = X + noise\nprint(result.round(2))",
      options: ["Values close to 5.0 with small random perturbations", "Exactly [[5,5,5],[5,5,5]]", "Values between 0 and 1", "All zeros"],
      correctIndex: 0,
      explanation: "Gaussian noise with σ=0.1 adds small perturbations around the original value 5.0. Each element gets a different random offset drawn from N(0, 0.1).",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What range of sigma is typical for Gaussian noise regularization?",
      options: ["σ = 1.0-10.0", "σ = 0.01-0.1 relative to data scale — enough to regularize without corrupting the signal", "σ = 0 (no noise)", "σ = 100"],
      correctIndex: 1,
      explanation: "Too small has no effect; too large destroys the signal. σ should be a small fraction of the data's standard deviation. Start at 0.01 and increase if still overfitting.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "How does salt-and-pepper noise differ from Gaussian noise?",
      options: ["They're identical", "Salt-and-pepper sets random pixels to exactly 0 or 1 (sparse corruption); Gaussian adds continuous small perturbations to all pixels", "Salt-and-pepper is continuous", "Gaussian only affects edges"],
      correctIndex: 1,
      explanation: "Salt-and-pepper is sparse and extreme (binary corruption). Gaussian is dense and mild (continuous perturbation). Different noise types test different robustness properties.",
      randomize: true
    },
    {
      id: "q5",
      type: "fill-blank",
      prompt: "In salt-and-pepper noise with prob=0.05, approximately ___% of pixels are set to 0 (pepper) and ___% are set to 1 (salt).",
      options: ["2.5, 2.5", "5, 5", "0, 5", "5, 0"],
      correctIndex: 0,
      explanation: "prob/2 = 0.025 for each. Pepper: mask < 0.025. Salt: mask > 0.975. Total corrupted ≈ 5%, split equally between 0 and 1.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What is a denoising autoencoder?",
      options: ["A filter that removes noise from images", "A network trained to reconstruct clean input from corrupted (noisy) input — learning to ignore noise and capture essential structure", "A noise generation algorithm", "A type of GAN"],
      correctIndex: 1,
      explanation: "Clean X → add noise → X̃ → encoder → decoder → X̂. Loss = ||X - X̂||². The network must learn the data manifold to separate signal from noise.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "Why should noise only be added during training, not testing?",
      options: ["Testing is faster without noise", "Test evaluation must measure performance on clean, real-world inputs. Adding noise at test time would artificially degrade measured performance.", "NumPy doesn't allow it", "Noise causes NaN at test time"],
      correctIndex: 1,
      explanation: "Training noise is regularization. Testing measures real deployment conditions. If your deployment receives noisy inputs, include representative noise in test data — but don't add artificial noise.",
      randomize: true
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What does SNR measure?",
      code: "import numpy as np\nclean = np.sin(np.linspace(0, 2*np.pi, 100))\nnoise = np.random.default_rng(42).normal(0, 0.1, 100)\nsnr = 10 * np.log10(np.var(clean) / np.var(noise))\nprint(f'SNR = {snr:.1f} dB')",
      options: ["Signal-to-Noise Ratio: how much stronger the signal is compared to noise (higher = cleaner)", "Sample count", "Standard deviation", "Speed of computation"],
      correctIndex: 0,
      explanation: "SNR = 10·log₁₀(var(signal)/var(noise)). Higher SNR means less noise relative to signal. σ=0.1 on sin wave gives ~17 dB. σ=0.3 gives ~7 dB.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Adding noise during training is mathematically equivalent to what under certain conditions?",
      options: ["Batch normalization", "Tikhonov regularization — it smooths the decision boundary by penalizing sensitivity to input perturbations", "Dropout", "Weight decay on biases"],
      correctIndex: 1,
      explanation: "Bishop (1995) showed that training with additive Gaussian noise is equivalent to Tikhonov regularization. Both penalize model complexity and smooth the learned function.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "In the salt_pepper function, why use X.copy() instead of modifying X directly?",
      options: ["copy() is faster", "To avoid mutating the original input — augmented data should be a new array, preserving the clean original for other uses", "NumPy requires it", "copy() adds randomness"],
      correctIndex: 1,
      explanation: "Data augmentation should never destroy the original. Multiple augmentations of the same sample need the clean source. In-place modification would corrupt subsequent augmentations.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "A model achieves 99% accuracy on clean test data but 60% when 5% salt-and-pepper noise is added. What does this indicate?",
      options: ["The model is well-generalized", "The model memorized clean patterns without learning robust features — it's fragile to input perturbation", "Salt-and-pepper is too aggressive", "The test data is wrong"],
      correctIndex: 1,
      explanation: "Large accuracy drop under mild noise = overfitting to clean data. Solution: train with noise injection or augmentation to build robustness.",
      randomize: true
    },
    {
      id: "q12",
      type: "fill-blank",
      prompt: "The denoising autoencoder pipeline is: Clean X → ___ → Corrupted X̃ → Encoder → Decoder → Reconstructed ___. Loss = ||___ - X̂||².",
      options: ["Add Noise, X̂, X", "Remove Noise, X, X̂", "Add Noise, X, X", "Normalize, X̂, X̂"],
      correctIndex: 0,
      explanation: "Corrupt the clean input with noise, then train to reconstruct the ORIGINAL clean X (not the corrupted version). The loss compares reconstruction to clean ground truth.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Which noise type is most appropriate for medical imaging where sensor artifacts cause random dead/hot pixels?",
      options: ["Gaussian noise", "Salt-and-pepper noise — models dead pixels (0) and hot/saturated pixels (max value)", "Uniform noise", "Poisson noise"],
      correctIndex: 1,
      explanation: "Salt-and-pepper directly models binary sensor failures. Gaussian models thermal/electronic noise. Choose noise type matching the real-world corruption mechanism.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "How does noise injection relate to diffusion models?",
      options: ["No relation", "Diffusion models train by predicting noise added at various levels. Generation reverses this: iteratively removing noise from pure Gaussian to produce data.", "Diffusion models remove noise from training data", "Noise is only used for initialization"],
      correctIndex: 1,
      explanation: "Forward process: gradually add Gaussian noise to data. Reverse process: learn to predict/remove noise. The entire framework is built on controlled noise injection and denoising.",
      randomize: true
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "Approximately what fraction of pixels are corrupted?",
      code: "import numpy as np\nrng = np.random.default_rng(42)\nX = np.ones((100, 100)) * 0.5\nmask = rng.random(X.shape)\nout = X.copy()\nout[mask < 0.025] = 0\nout[mask > 0.975] = 1\ncorrupted = np.sum(out != 0.5) / out.size\nprint(f'Corrupted: {corrupted:.3f}')",
      options: ["≈ 0.05 (5%)", "≈ 0.025 (2.5%)", "≈ 0.10 (10%)", "≈ 0.50 (50%)"],
      correctIndex: 0,
      explanation: "2.5% pepper (mask < 0.025) + 2.5% salt (mask > 0.975) = 5% total corruption. Matches the prob parameter in the salt_pepper function.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why might you use different noise levels at different training stages?",
      options: ["You shouldn't — constant noise is best", "Start with higher noise for broad exploration of the loss landscape, then reduce for fine-tuning. Curriculum-style noise scheduling can improve convergence.", "Noise level is determined by batch size", "Hardware limitations require it"],
      correctIndex: 1,
      explanation: "Annealing noise mirrors learning rate schedules. Early high noise prevents premature convergence to sharp minima. Late low noise allows precise optimization.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What's the relationship between noise injection and dropout?",
      options: ["They're identical", "Both are stochastic regularization: dropout randomly zeros activations; noise injection perturbs inputs/activations with continuous noise. They target different layers of the network.", "Dropout replaces noise injection", "Noise injection only works on CNNs"],
      correctIndex: 1,
      explanation: "Dropout = multiplicative Bernoulli noise on hidden units. Input noise = additive Gaussian noise on data. Both prevent co-adaptation and improve generalization through stochasticity.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "For a model processing audio signals, which noise model is most physically realistic?",
      options: ["Salt-and-pepper", "Gaussian (thermal/electronic noise) or Poisson (counting/statistical noise in sensors)", "Uniform [0,1]", "Bernoulli"],
      correctIndex: 1,
      explanation: "Audio sensors have thermal noise (Gaussian) and quantization/counting noise (Poisson-like). Match noise model to physical source for effective augmentation.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "You train with Gaussian noise σ=0.5 but test accuracy drops vs training without noise. What went wrong?",
      options: ["Noise always hurts", "σ=0.5 is likely too large relative to data scale — the noise overwhelms the signal, preventing the model from learning useful patterns", "Gaussian noise doesn't work", "Need more epochs"],
      correctIndex: 1,
      explanation: "Noise scale must be calibrated to data. If data values are in [0,1] and σ=0.5, noise dominates. Reduce σ until training loss decreases while maintaining regularization benefit.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which statement best captures the philosophy behind noise-based regularization?",
      options: ["More noise is always better", "If the model can handle perturbed inputs, it has learned genuine structure rather than memorizing surface patterns — robustness implies understanding", "Noise replaces all other regularization", "Only works for image data"],
      correctIndex: 1,
      explanation: "Noise tests whether the model learned the underlying manifold or just interpolated training points. Robustness to perturbation is evidence of genuine feature learning.",
      randomize: true
    }
  ]}
/>
