---
title: "Capstone: A CRNN OCR Model"
slug: "045-crnn-ocr-capstone"
description: "Combine CNN feature extraction, sequence modeling, CTC decoding, preprocessing, and evaluation into a complete OCR system design."
track: "nn-intermediate"
order: 10
read_time: 30
code_time: 25
execution_timeout: 30
prerequisites: ["040-ocr-text-recognition", "042-lstm-gru", "044-seq2seq"]
---

# Capstone: A CRNN OCR Model

Now we build the full OCR pipeline: image in, text out. A practical OCR model is not just a classifier — it is a structured sequence model:

```
Raw image → preprocessing → CNN feature extractor → BiLSTM/GRU sequence model
          → per-timestep character probabilities → CTC decode → text
```

The goal is not to memorize one digit. The goal is to read a variable-width word or line without knowing where each character begins.

## 1. Image Preprocessing

OCR models are extremely sensitive to input cleanup. Before the neural network, normalize the visual problem:

```python
import numpy as np

def normalize_image(img):
    """img: grayscale array, arbitrary range."""
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def binarize(img, threshold=0.5):
    return (img > threshold).astype(np.float64)

def pad_to_height(img, target_h=32):
    h, w = img.shape
    if h >= target_h:
        return img[:target_h]
    top = (target_h - h) // 2
    bottom = target_h - h - top
    return np.pad(img, ((top, bottom), (0, 0)), mode="constant")
```

<PyRunner
  cellId="045-cell-1"
  defaultCode={`import numpy as np

np.random.seed(42)
img = np.random.rand(20, 60) * 80 + 120  # fake scanned grayscale range

norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
bin_img = (norm > 0.5).astype(float)

target_h = 32
pad_top = (target_h - bin_img.shape[0]) // 2
pad_bottom = target_h - bin_img.shape[0] - pad_top
padded = np.pad(bin_img, ((pad_top, pad_bottom), (0,0)))

print("OCR preprocessing pipeline:")
print(f"  Raw image:      shape={img.shape}, range=({img.min():.1f}, {img.max():.1f})")
print(f"  Normalized:     shape={norm.shape}, range=({norm.min():.1f}, {norm.max():.1f})")
print(f"  Binarized:      foreground ratio={bin_img.mean():.2f}")
print(f"  Height padded:  shape={padded.shape}")
print("\n✅ Network now receives stable input scale and fixed height")
`}
/>

## 2. CNN as Feature Extractor

For OCR, the CNN should preserve horizontal order. It can reduce height aggressively, but width becomes the time axis.

Example input: <InlineMath latex="(N, 1, 32, 128)" />

```
Conv/Pool → (N, 32, 16, 64)
Conv/Pool → (N, 64, 8, 32)
Conv      → (N, 128, 4, 32)
Collapse height → (N, 32, 512)
```

Now each of the 32 columns is one timestep with 512 visual features.

```python
def cnn_to_sequence(feature_map):
    """feature_map: (N, C, H, W) → sequence: (W, N, C*H)"""
    N, C, H, W = feature_map.shape
    seq = feature_map.transpose(3, 0, 1, 2).reshape(W, N, C * H)
    return seq
```

<PyRunner
  cellId="045-cell-2"
  defaultCode={`import numpy as np

N, C, H, W = 4, 128, 4, 32
features = np.random.randn(N, C, H, W)
seq = features.transpose(3, 0, 1, 2).reshape(W, N, C*H)

print("CNN feature map → sequence:")
print(f"  Feature map: {features.shape} = (batch, channels, height, width)")
print(f"  Sequence:    {seq.shape} = (timesteps, batch, features)")
print(f"\nEach timestep corresponds to one vertical slice of the image.")
print(f"Feature dimension = C × H = {C} × {H} = {C*H}")
`}
/>

## 3. Bidirectional Sequence Model

A character is easier to recognize with left and right context. For example, a vertical stroke could be `l`, `1`, or part of `h`. Bidirectional RNNs look both ways:

```python
def bidirectional_rnn(sequence, rnn_forward, rnn_backward):
    """sequence: (T, N, D)"""
    h_fwd = rnn_forward(sequence)
    h_bwd = rnn_backward(sequence[::-1])[::-1]
    return np.concatenate([h_fwd, h_bwd], axis=2)
```

Output shape: <InlineMath latex="(T, N, 2H)" />.

## 4. Character Projection

At every timestep, predict a distribution over characters plus blank:

```python
def character_logits(H_seq, W_out, b_out):
    """H_seq: (T, N, 2H), W_out: (2H, vocab_size)"""
    T, N, D = H_seq.shape
    return H_seq.reshape(T*N, D) @ W_out + b_out
```

The output is reshaped to <InlineMath latex="(T, N, V)" /> and trained with CTC.

## 5. Greedy CTC Decode

```python
def ctc_decode(prob_seq, alphabet):
    """prob_seq: (T, V), alphabet[0] is blank."""
    ids = np.argmax(prob_seq, axis=1)
    out = []
    prev = -1
    for i in ids:
        if i != prev and i != 0:
            out.append(alphabet[i])
        prev = i
    return "".join(out)
```

<PyRunner
  cellId="045-cell-3"
  defaultCode={`import numpy as np

alphabet = ['_', 'c', 'a', 't', 'r']
raw_ids = [0, 1, 1, 0, 2, 2, 0, 3, 3, 0]  # _cc_aa_tt_

out = []
prev = -1
for i in raw_ids:
    if i != prev and i != 0:
        out.append(alphabet[i])
    prev = i

print(f"Raw timestep argmax: {[alphabet[i] for i in raw_ids]}")
print(f"Decoded text:        {''.join(out)}")
print("\nCTC rule: collapse repeats, remove blanks")
`}
/>

## OCR Evaluation

Accuracy per word is too strict for debugging. Use character error rate:

<BlockMath latex="CER = \frac{\text{edit distance}(prediction, target)}{\text{len}(target)}" />

```python
def edit_distance(a, b):
    dp = np.zeros((len(a)+1, len(b)+1), dtype=int)
    dp[:,0] = np.arange(len(a)+1)
    dp[0,:] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j] + 1, dp[i,j-1] + 1, dp[i-1,j-1] + cost)
    return dp[-1,-1]
```

> [!IMPORTANT] OCR Debugging Checklist
> - If output is empty: blank class dominates; reduce blank bias or check CTC loss.
> - If repeated characters vanish incorrectly: inspect CTC collapsing rules.
> - If text is shifted or garbled: CNN width reduction may be too aggressive.
> - If training is unstable: normalize images and clip RNN gradients.

<Quiz
  chapterSlug="045-crnn-ocr-capstone"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does OCR convert CNN feature maps into a sequence along the width dimension?",
      options: ["Because text is naturally read left-to-right", "Because height contains no information", "Because CTC requires square inputs", "Because CNNs cannot output text"],
      correctIndex: 0,
      explanation: "After CNN feature extraction, each vertical column represents a slice of the word or line. Reading those columns left-to-right gives a natural timestep sequence for an RNN/CTC model.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What metric is better than exact word accuracy while debugging OCR?",
      options: ["MSE", "Character Error Rate", "AUC", "Perplexity"],
      correctIndex: 1,
      explanation: "CER reveals partial correctness. Predicting 'helo' for 'hello' is not totally wrong; CER captures the one missing character, while exact word accuracy treats it as a full failure.",
      randomize: false,
    }
  ]}
/>
