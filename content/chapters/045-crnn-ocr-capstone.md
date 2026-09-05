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
      options: [
        "Because text is naturally read left-to-right — each vertical column becomes a timestep representing a horizontal slice of the image",
        "Because height contains no useful information",
        "Because CTC requires square inputs",
        "Because CNNs cannot output text directly"
      ],
      correctIndex: 0,
      explanation: "After CNN processing, each column of the feature map corresponds to a horizontal position in the image. Reading columns left-to-right creates a temporal sequence that an RNN/CTC model can process naturally.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What metric is better than exact word accuracy for debugging OCR?",
      options: ["Mean Squared Error", "Character Error Rate (CER) — edit distance normalized by target length", "AUC-ROC", "Perplexity"],
      correctIndex: 1,
      explanation: "CER reveals partial correctness. Predicting 'helo' for 'hello' gives CER=0.2 (one edit), while word accuracy counts it as 100% wrong. CER guides incremental improvement during development.",
      randomize: false
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "If CNN outputs (N, 128, 4, 32) and you collapse height via transpose+reshape to sequence, what is the resulting shape?",
      options: ["(32, N, 512)", "(N, 32, 512)", "(128, N, 32)", "(4, N, 128)"],
      correctIndex: 0,
      explanation: "Transpose to (W, N, C, H) = (32, N, 128, 4), then reshape to (32, N, 128×4) = (32, N, 512). Width becomes timesteps, C×H becomes feature dimension per timestep.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why normalize images to [0,1] before feeding to the OCR network?",
      options: [
        "To reduce file size",
        "Scanned images have arbitrary intensity ranges. Normalization ensures consistent input scale regardless of scanner/camera settings, preventing activation saturation and gradient issues.",
        "Neural networks require binary inputs",
        "It makes binarization unnecessary"
      ],
      correctIndex: 1,
      explanation: "Raw pixel values might range from 0-255, 0-4095, or any scanner-dependent range. Without normalization, large inputs saturate activations and destabilize training. [0,1] provides a stable, predictable input distribution.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What does this preprocessing produce?",
      code: "import numpy as np\nimg = np.array([[100, 200], [50, 150]], dtype=float)\nnorm = (img - img.min()) / (img.max() - img.min() + 1e-8)\nprint(norm.round(2))",
      options: ["[[0.33 1.  ]\n [0.   0.67]]", "[[0.5 1.0]\n [0.25 0.75]]", "[[100 200]\n [50 150]]", "Error"],
      correctIndex: 0,
      explanation: "min=50, max=200, range=150. (100-50)/150=0.33, (200-50)/150=1.0, (50-50)/150=0.0, (150-50)/150=0.67. Min-max normalization maps to [0,1] preserving relative intensities.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Why use a bidirectional RNN instead of unidirectional for OCR?",
      options: [
        "It's twice as fast",
        "Characters are ambiguous in isolation — a vertical stroke could be 'l', '1', or part of 'h'. Seeing both left and right context disambiguates recognition.",
        "CTC requires bidirectional inputs",
        "It doubles the vocabulary size"
      ],
      correctIndex: 1,
      explanation: "OCR characters depend heavily on context. Forward pass sees what came before; backward pass sees what follows. Concatenating both gives each timestep full contextual awareness for better character discrimination.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "The CNN in CRNN should aggressively reduce ___ but preserve ___ since width becomes the time axis.",
      options: ["height, width", "width, height", "channels, spatial", "depth, batch"],
      correctIndex: 0,
      explanation: "Height is collapsed (via pooling or strided conv) to create a 1D sequence. Width must be preserved proportionally because each column maps to one timestep. Aggressive width reduction loses character-level resolution.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does pad_to_height accomplish in OCR preprocessing?",
      options: [
        "Makes all images the same width",
        "Ensures fixed height (e.g., 32px) so the CNN produces consistent feature map dimensions. Shorter images are centered with zero-padding.",
        "Increases resolution",
        "Removes noise"
      ],
      correctIndex: 1,
      explanation: "Neural networks expect fixed-size inputs (or at least fixed height for OCR). Padding shorter images to target height ensures the CNN always produces the same number of feature rows, enabling batched processing.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "In ctc_decode, why check `i != prev` before adding a character?",
      options: [
        "To skip blank tokens",
        "To collapse consecutive duplicate predictions — CTC emits the same character across multiple timesteps, which must be merged into a single output character",
        "To prevent index errors",
        "To handle uppercase letters"
      ],
      correctIndex: 1,
      explanation: "CTC models often predict the same character for several consecutive timesteps (e.g., 'lll' for 'l'). Without collapsing, you'd get 'lll' instead of 'l'. The blank token separates intentional repeats.",
      randomize: true
    },
    {
      id: "q10",
      type: "ordering",
      prompt: "Order the complete CRNN OCR pipeline:",
      items: ["Image preprocessing (normalize, binarize, pad)", "CNN feature extraction (reduce height, preserve width)", "Collapse to sequence (transpose + reshape)", "Bidirectional RNN (contextual features)", "Character projection + CTC decode"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Clean input → extract visual features → convert to temporal sequence → add sequential context → predict characters. Each stage transforms the representation toward the final text output.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Your OCR model outputs empty strings for every image. What's the most likely cause?",
      options: [
        "The CNN is too deep",
        "The blank class dominates predictions — the model learned to predict blank at every timestep. Check CTC loss implementation, reduce blank bias initialization, or verify label formatting.",
        "Images are too small",
        "The learning rate is too low"
      ],
      correctIndex: 1,
      explanation: "Predicting all-blanks minimizes CTC loss trivially if the loss isn't computed correctly or if the blank logit has a large positive bias. This is the most common CTC training failure mode.",
      randomize: true
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the edit distance between 'kitten' and 'sitting'?",
      code: "# Simplified: count edits needed\n# kitten → sitten (sub k→s)\n# sitten → sittin (sub e→i) \n# sittin → sitting (insert g)\nprint(3)",
      options: ["3", "2", "4", "7"],
      correctIndex: 0,
      explanation: "Edit distance = minimum insertions + deletions + substitutions to transform one string to another. kitten→sitting: substitute k→s, substitute e→i, insert g = 3 edits. CER = 3/7 ≈ 0.43.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Why does the bidirectional RNN output have dimension 2H instead of H?",
      options: [
        "It processes twice as many timesteps",
        "Forward hidden state (H) and backward hidden state (H) are concatenated, giving each timestep access to both past and future context in a 2H-dimensional vector",
        "It uses two separate LSTM layers",
        "The output is doubled for numerical stability"
      ],
      correctIndex: 1,
      explanation: "BiRNN runs forward (left→right) and backward (right→left), each producing H-dim hidden states. Concatenation gives 2H, doubling the feature richness at each timestep for the character projection layer.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "If your OCR output shows repeated characters vanishing incorrectly ('ee' → 'e'), what should you check?",
      options: [
        "The CNN kernel size",
        "CTC collapsing rules — the model may not be emitting blank tokens between repeated characters, causing them to merge during decoding",
        "The learning rate",
        "The image resolution"
      ],
      correctIndex: 1,
      explanation: "CTC requires blank emissions between identical consecutive characters. If the model never predicts blank between two 'e' regions, greedy decode collapses them. Verify blank predictions exist in the probability sequence.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "CER = edit_distance(prediction, target) / len(___). A CER of 0.0 means ___ match.",
      options: ["target, perfect", "prediction, partial", "target, no", "both, approximate"],
      correctIndex: 0,
      explanation: "CER normalizes edit distance by target length. CER=0 means zero edits needed — prediction exactly matches target. CER=1 means every character needs editing. Lower is better.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why is text shifted/garbled in OCR output when CNN width reduction is too aggressive?",
      options: [
        "The RNN can't process wide sequences",
        "Aggressive width pooling merges adjacent characters into single timesteps, destroying character boundaries. The model can no longer distinguish where one character ends and another begins.",
        "CTC doesn't support wide inputs",
        "The embedding layer overflows"
      ],
      correctIndex: 1,
      explanation: "Each timestep should correspond to roughly one character-width of the image. Excessive width reduction combines multiple characters per timestep, making individual character recognition impossible.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What role does the character projection layer (W_out, b_out) serve?",
      options: [
        "It extracts visual features",
        "It maps the 2H-dimensional RNN output at each timestep to vocabulary-sized logits — one score per character plus blank",
        "It performs CTC decoding",
        "It normalizes the input"
      ],
      correctIndex: 1,
      explanation: "The projection layer is a linear transformation: (T, N, 2H) → (T, N, V). It converts contextual RNN features into per-character scores that CTC loss trains and greedy decode consumes.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What shape does the character logits have after reshaping?",
      code: "T, N, D = 32, 4, 512\nV = 37  # alphabet + blank\nlogits_flat = np.zeros((T*N, V))\nlogits = logits_flat.reshape(T, N, V)\nprint(logits.shape)",
      options: ["(32, 4, 37)", "(4, 32, 37)", "(37, 32, 4)", "(128, 37)"],
      correctIndex: 0,
      explanation: "Reshape from (T×N, V) back to (T, N, V). Each of 32 timesteps, for each of 4 samples, has 37 character scores. This format is required for CTC loss computation.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Training is unstable with NaN losses. According to the debugging checklist, what should you try?",
      options: [
        "Increase the learning rate",
        "Normalize images to [0,1] and clip RNN gradients — both prevent extreme activations that cause numerical overflow",
        "Use a deeper CNN",
        "Remove batch normalization"
      ],
      correctIndex: 1,
      explanation: "NaN typically comes from exploding activations/gradients. Image normalization prevents large inputs; gradient clipping caps extreme updates. Both are essential stability measures for RNN-based OCR.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Why is the CRNN architecture particularly well-suited for OCR compared to pure CNN classification?",
      options: [
        "CRNNs are faster",
        "Pure CNNs require fixed-size inputs and fixed output classes. CRNN handles variable-width images via the sequence model and variable-length outputs via CTC, matching OCR's inherent variability.",
        "CNNs can't process grayscale images",
        "CRNNs don't need training data"
      ],
      correctIndex: 1,
      explanation: "OCR inputs vary in width (different word lengths) and outputs vary in length (different character counts). CRNN's sequence modeling + CTC naturally handles both variabilities without requiring segmentation or fixed-size constraints.",
      randomize: true
    }
  ]}
/>
