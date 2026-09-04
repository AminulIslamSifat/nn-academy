---
title: "OCR: From Digit Recognition to Text Recognition"
slug: "040-ocr-text-recognition"
description: "Scale from single-digit MNIST to recognizing full words and lines. Learn CTC loss, sliding windows, and CRNN architecture concepts in NumPy."
track: "nn-intermediate"
order: 5
read_time: 28
code_time: 20
execution_timeout: 30
prerequisites: ["039-cnn-mnist"]
---

# OCR: From Digit Recognition to Text Recognition

MNIST classifies one digit. ==Real OCR reads entire words and sentences== of variable length. This requires three new ideas: handling variable-length sequences, predicting without knowing character boundaries, and training with misaligned labels.

## The OCR Problem

| Aspect | MNIST | Real OCR |
|--------|-------|----------|
| Input | Fixed 28×28 | Variable width images |
| Output | Single digit (0-9) | Variable-length string |
| Alignment | Known | Unknown (where does each char start?) |
| Classes | 10 | 26+ letters + digits + symbols |

## Approach 1: Sliding Window (Simple)

Slide a fixed-size window across the image, classify each patch independently:

```python
import numpy as np

def sliding_window_ocr(image, window_width=28, stride=7):
    """Classify overlapping patches across a text line image.
    image: (H, W) grayscale
    Returns: list of (position, predicted_char, confidence)
    """
    H, W = image.shape
    results = []
    for x in range(0, W - window_width + 1, stride):
        patch = image[:, x:x+window_width]
        # Resize to 28×28 if needed, then classify
        # prob = classifier.predict(patch)
        # results.append((x, argmax(prob), max(prob)))
        pass
    return results
```

Problems: doesn't handle variable-width characters, no context between predictions, needs post-processing to merge overlapping detections.

## Approach 2: CRNN Architecture (Modern)

==Convolutional Recurrent Neural Network== — the standard OCR architecture:

```
Image (H, W)
  → CNN feature extractor → (T, F) feature sequence
  → Bidirectional RNN → (T, num_classes) per-timestep predictions
  → CTC decoding → final text string
```

The CNN extracts visual features column-by-column. The RNN models sequential dependencies. CTC handles the alignment problem.

### Feature Extraction

```python
def cnn_feature_extractor(X):
    """X: (N, 1, H, W) → (N, T, F) where T = W//4 after pooling."""
    # Conv blocks reduce height to 1, width by factor of 4
    # Result: each column of features represents a horizontal slice
    # of the input image
    pass  # Uses conv layers from chapters 036-037
```

## CTC Loss: Training Without Alignment

The key innovation. ==CTC sums over ALL possible alignments== between predictions and labels:

Given predictions <InlineMath latex="y_t^c" /> at each timestep and target string <InlineMath latex="l" />:

<BlockMath latex="P(l|X) = \sum_{\pi: B(\pi)=l} P(\pi|X) = \sum_{\pi: B(\pi)=l} \prod_{t=1}^{T} y_t^{\pi_t}" />

where <InlineMath latex="B" /> collapses repeated characters and removes blanks.

The trick: dynamic programming computes this sum efficiently in <InlineMath latex="O(T \cdot |l|)" /> instead of enumerating exponentially many paths.

<PyRunner
  cellId="040-cell-1"
  defaultCode={`import numpy as np

# Demonstrate CTC concept with a simple example
# Predictions over 5 timesteps for chars {a, b, -}
# Target: "ab"

np.random.seed(42)
T = 5
chars = ['-', 'a', 'b']  # - is blank
logits = np.array([
    [2.0, 0.5, 0.1],  # t=0: mostly blank
    [0.1, 2.5, 0.2],  # t=1: mostly 'a'
    [1.5, 0.3, 0.3],  # t=2: blank-ish
    [0.2, 0.3, 2.0],  # t=3: mostly 'b'
    [1.0, 0.5, 0.8],  # t=4: mixed
])

# Softmax
shifted = logits - logits.max(axis=1, keepdims=True)
probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

print("Per-timestep probabilities:")
print(f"{'t':>3} | {'blank':>6} | {'a':>6} | {'b':>6} | Best")
print("─" * 38)
for t in range(T):
    best = chars[np.argmax(probs[t])]
    print(f"{t:3d} | {probs[t,0]:6.3f} | {probs[t,1]:6.3f} | {probs[t,2]:6.3f} | {best}")

print(f"\nValid alignments for target 'ab':")
alignments = [
    "-ab--", "-a-b-", "-a--b", "aab--", "a-ab-",
    "aa-b-", "-abb-", "-ab-b", "aabb-", "aab-b"
]
for a in alignments[:5]:
    prob = np.prod([probs[t, chars.index(c)] for t, c in enumerate(a)])
    print(f"  {a}: P={prob:.6f}")

print(f"\n💡 CTC sums ALL valid alignments automatically")
print(f"   No need to know where each character starts!")
`}
/>

### Simplified CTC Forward Algorithm

```python
def ctc_loss(log_probs, targets, input_lengths, target_lengths):
    """Simplified CTC loss using forward algorithm.
    log_probs: (T, N, C) log-softmax outputs
    targets: (N, S) target sequences (padded)
    Returns: scalar loss
    """
    T, N, C = log_probs.shape
    total_loss = 0.0
    
    for n in range(N):
        S = target_lengths[n]
        label = targets[n, :S]
        
        # Create extended label with blanks: [-, l1, -, l2, -, ...]
        ext_label = np.zeros(2 * S + 1, dtype=int)  # 0 = blank
        ext_label[1::2] = label
        
        L = len(ext_label)
        alpha = np.full((T, L), -np.inf)
        
        # Initialize
        alpha[0, 0] = log_probs[0, n, 0]  # blank
        if L > 1:
            alpha[0, 1] = log_probs[0, n, ext_label[1]]
        
        # Forward pass
        for t in range(1, T):
            for s in range(L):
                a = alpha[t-1, s]
                if s > 0:
                    a = np.logaddexp(a, alpha[t-1, s-1])
                if s > 1 and ext_label[s] != ext_label[s-2]:
                    a = np.logaddexp(a, alpha[t-1, s-2])
                alpha[t, s] = a + log_probs[t, n, ext_label[s]]
        
        # Total probability
        log_prob = np.logaddexp(alpha[T-1, L-1], alpha[T-1, L-2])
        total_loss -= log_prob
    
    return total_loss / N
```

> [!IMPORTANT] Why CTC Matters for OCR
> Without CTC, you'd need pixel-level character segmentation labels — extremely expensive to annotate. CTC trains on just the text string, learning alignment implicitly. This made modern OCR practical.

## Greedy CTC Decoding

At inference, collapse predictions:

```python
def ctc_greedy_decode(probs):
    """probs: (T, C). Returns decoded string."""
    preds = np.argmax(probs, axis=1)
    # Remove consecutive duplicates and blanks
    decoded = []
    prev = -1  # -1 = blank
    for p in preds:
        if p != prev and p != 0:  # not blank, not duplicate
            decoded.append(p)
        prev = p
    return decoded
```

<PyRunner
  cellId="040-cell-2"
  defaultCode={`import numpy as np

# Simulate CTC greedy decoding
np.random.seed(42)
chars = ['_', 'h', 'e', 'l', 'o']  # _ = blank

# Simulated predictions for "hello"
T = 12
preds = [0, 1, 1, 0, 2, 2, 3, 3, 3, 0, 4, 4]  # raw argmax

# Decode
decoded = []
prev = -1
for p in preds:
    if p != prev and p != 0:
        decoded.append(chars[p])
    prev = p

result = "".join(decoded)
print(f"Raw predictions: {[chars[p] for p in preds]}")
print(f"After CTC decode: '{result}'")
print(f"\n✅ Repeated chars collapsed, blanks removed")
print(f"   '_hh_eelll_o_' → 'hello'")
`}
/>

## OCR Pipeline Summary

```
Input Image → Preprocessing (binarize, deskew, resize)
           → CNN Feature Extractor (visual features per column)
           → Bidirectional RNN (sequential context)
           → CTC Loss (training) / Greedy Decode (inference)
           → Post-processing (language model, dictionary lookup)
```

<Quiz
  chapterSlug="040-ocr-text-recognition"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What problem does CTC solve that standard cross-entropy cannot?",
      options: [
        "CTC is faster to compute",
        "Standard CE requires aligned labels (knowing which timestep corresponds to which character). CTC handles unaligned input-output pairs by summing over all valid alignments.",
        "CTC works with larger vocabularies",
        "CTC prevents overfitting"
      ],
      correctIndex: 1,
      explanation: "In OCR, you know the output text but not where each character appears in the image. CTC marginalizes over all possible alignments, enabling training with only transcript-level labels.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does CTC use a blank token?",
      options: [
        "To pad short sequences",
        "To represent spaces between words",
        "To allow the model to output nothing at certain timesteps, enabling variable-length outputs and handling repeated characters",
        "It's optional and can be removed"
      ],
      correctIndex: 2,
      explanation: "Without blank, every timestep must emit a character. Blank lets the model 'wait' during transitions between characters and handles cases where the same character spans multiple timesteps (collapsed during decoding).",
      randomize: false,
    }
  ]}
/>
