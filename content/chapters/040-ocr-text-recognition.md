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
      prompt: "What fundamental problem does CTC loss solve that standard cross-entropy cannot?",
      options: [
        "CTC is faster to compute than cross-entropy",
        "Standard CE requires aligned labels (knowing which timestep maps to which character). CTC handles unaligned input-output pairs by summing over all valid alignments.",
        "CTC supports larger vocabularies",
        "CTC automatically prevents overfitting"
      ],
      correctIndex: 1,
      explanation: "In OCR, you know the output text but not where each character appears in the image. CTC marginalizes over all possible alignments via dynamic programming, enabling training with only transcript-level labels.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does CTC introduce a blank token?",
      options: [
        "To pad short sequences to fixed length",
        "To represent whitespace between words",
        "To let the model output nothing at certain timesteps, enabling variable-length outputs and proper handling of repeated characters",
        "It's optional and can be safely removed"
      ],
      correctIndex: 2,
      explanation: "Without blank, every timestep must emit a character. Blank lets the model 'wait' during transitions and handles cases where the same character spans multiple timesteps — consecutive duplicates are collapsed during decoding.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In CTC greedy decoding, what happens to the raw prediction sequence [_, h, h, _, e, e, l, l, l, _, o, o]?",
      options: [
        "All tokens kept as-is",
        "Consecutive duplicates collapsed then blanks removed → 'hello'",
        "Only blanks removed → 'hhell oo'",
        "Randomly sampled to produce output"
      ],
      correctIndex: 1,
      explanation: "Greedy CTC decode: first collapse consecutive duplicates (_, h, _, e, l, _, o), then remove blanks (h, e, l, o) → 'hello'. The order matters — collapse first, then strip blanks.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the main limitation of the sliding window approach to OCR?",
      options: [
        "It's too slow for real-time applications",
        "It classifies each patch independently with no context between predictions, can't handle variable-width characters, and needs post-processing to merge overlapping detections",
        "It only works with digits",
        "It requires GPU acceleration"
      ],
      correctIndex: 1,
      explanation: "Sliding window treats each patch as an isolated classification problem. No sequential modeling means no context, no natural handling of variable character widths, and ambiguous overlapping predictions that need heuristic merging.",
      randomize: true
    },
    {
      id: "q5",
      type: "fill-blank",
      prompt: "The CRNN architecture consists of three stages: CNN feature extractor → ___ → CTC decoding.",
      options: ["Fully-connected classifier", "Bidirectional RNN", "Attention mechanism", "Max pooling"],
      correctIndex: 1,
      explanation: "CNN extracts visual features per column, BiRNN models left-right sequential context across columns, and CTC handles alignment-free training/decoding. Each stage addresses a different aspect of the OCR problem.",
      randomize: false
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "In the CTC forward algorithm, why is the label extended with blanks between every character (e.g., 'ab' → '-a-b-')?",
      options: [
        "To make all labels the same length",
        "To explicitly represent all valid alignment paths — the extended label encodes which transitions are allowed (blank→char, char→blank, char→different-char, but NOT char→same-char without intervening blank)",
        "To improve numerical stability",
        "It's a convention with no mathematical purpose"
      ],
      correctIndex: 1,
      explanation: "The extended label creates a state machine where valid CTC paths correspond to traversals through these states. Blanks between characters allow repetition of the same character (via blank intermediary) while preventing invalid direct repeats.",
      randomize: true
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "What does this CTC greedy decode produce?",
      code: "preds = [0, 1, 1, 2, 0, 2, 3, 3, 0]  # 0=blank\ndecoded = []\nprev = -1\nfor p in preds:\n    if p != prev and p != 0:\n        decoded.append(p)\n    prev = p\nprint(decoded)",
      options: ["[1, 2, 2, 3]", "[1, 2, 3]", "[1, 1, 2, 2, 3, 3]", "[0, 1, 2, 0, 2, 3, 0]"],
      correctIndex: 0,
      explanation: "Walk through: 0(skip blank), 1(add, prev=1), 1(same as prev, skip), 2(different, add, prev=2), 0(blank, prev=0), 2(different from 0, add, prev=2), 3(different, add, prev=3), 3(same, skip), 0(blank). Result: [1,2,2,3]. Note: two 2s because they're separated by a blank.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "Why does the CNN feature extractor reduce height to 1 but keep width proportional to input?",
      options: [
        "Height contains no useful information",
        "Each resulting column of features represents a horizontal slice of the image, creating a left-to-right sequence that the RNN can process temporally",
        "It's required by the CTC loss function",
        "To reduce computation"
      ],
      correctIndex: 1,
      explanation: "Text reads left-to-right. Collapsing height creates a 1D sequence where position t corresponds to horizontal location t in the image. This converts a 2D spatial problem into a 1D temporal one suitable for RNN processing.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "What is the time complexity of the CTC forward algorithm?",
      options: ["O(T × |l|) where T = timesteps, |l| = label length", "O(T²)", "O(2^T)", "O(T × C) where C = number of classes"],
      correctIndex: 0,
      explanation: "The forward algorithm uses dynamic programming over the extended label (length 2|l|+1) across T timesteps. Naive enumeration of all alignments would be exponential; DP makes it tractable at O(T × |l|).",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "In the CTC forward pass, when can alpha[t,s] receive contributions from alpha[t-1, s-2] (skip transition)?",
      options: [
        "Always",
        "Only when ext_label[s] ≠ ext_label[s-2] — i.e., the current and two-back labels are different characters (not the same character separated by blank)",
        "Only at even timesteps",
        "Never — only adjacent transitions are allowed"
      ],
      correctIndex: 1,
      explanation: "The skip transition (s-2 → s) jumps over a blank. It's only valid when the characters on either side differ. If ext_label[s] == ext_label[s-2], skipping would incorrectly merge two instances of the same character without an explicit blank emission between them.",
      randomize: true
    },
    {
      id: "q11",
      type: "ordering",
      prompt: "Order the complete OCR pipeline stages:",
      items: ["Image preprocessing (binarize, deskew, resize)", "CNN feature extraction (visual features per column)", "Bidirectional RNN (sequential context)", "CTC decoding (collapse + remove blanks)", "Post-processing (language model, dictionary)"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Preprocessing cleans the image → CNN extracts spatial features → BiRNN adds sequential context → CTC decodes raw predictions to text → Post-processing corrects errors using language knowledge.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why is log-space used in the CTC forward algorithm (log_probs, logaddexp)?",
      options: [
        "Log-space makes the math simpler to understand",
        "Multiplying many small probabilities causes underflow. Log-space converts products to sums, maintaining numerical precision across hundreds of timesteps",
        "GPUs compute logs faster than multiplication",
        "It's optional — regular probability space works fine"
      ],
      correctIndex: 1,
      explanation: "CTC multiplies probabilities across T timesteps. With T=100 and average prob=0.5, the product is ~10^-30 — below float32 range. Log-space keeps values in a manageable range (-70 to 0) and uses logaddexp for stable summation.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "How does a bidirectional RNN improve OCR compared to a unidirectional RNN?",
      options: [
        "It's twice as fast",
        "Each timestep has access to both past AND future context. When recognizing a character, seeing what comes after helps disambiguate visually similar characters.",
        "It doubles the number of output classes",
        "It eliminates the need for CTC"
      ],
      correctIndex: 1,
      explanation: "In OCR, future context is as valuable as past context. A partially visible 'm' followed by 'ing' is more confidently 'm' than 'rn'. BiRNN concatenates forward and backward hidden states, giving full contextual awareness.",
      randomize: true
    },
    {
      id: "q14",
      type: "fill-blank",
      prompt: "In CTC greedy decoding, the rule is: collapse consecutive ___ first, then remove all ___ tokens.",
      options: ["duplicates, blank", "blanks, duplicate", "predictions, zero", "characters, space"],
      correctIndex: 0,
      explanation: "First pass: merge consecutive identical predictions (aaa → a). Second pass: strip blank tokens. Order matters — if you remove blanks first, previously separated same-characters become adjacent and get incorrectly merged.",
      randomize: false
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What advantage does CTC provide for annotation cost in OCR datasets?",
      options: [
        "CTC generates synthetic labels automatically",
        "CTC only needs transcript-level labels (the text string), not pixel-level character bounding boxes or segmentation masks. This makes dataset creation dramatically cheaper.",
        "CTC reduces the number of training samples needed",
        "CTC works without any labels"
      ],
      correctIndex: 1,
      explanation: "Traditional OCR required precise character-level segmentation — expensive and error-prone. CTC's alignment-free training means annotators just type the text they see, reducing annotation cost by orders of magnitude.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "If your CTC model outputs [b, b, _, b, b] for target 'bb', what does greedy decoding produce?",
      options: ["'b'", "'bb'", "'bbb'", "'b_b'"],
      correctIndex: 1,
      explanation: "Collapse consecutive duplicates: [b, _, b]. Remove blanks: [b, b] → 'bb'. The blank between the two b-groups preserves them as separate characters. Without the blank, [b,b,b,b,b] would collapse to just 'b'.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why can't standard softmax + cross-entropy train an OCR model directly?",
      options: [
        "Softmax doesn't work with variable-length outputs",
        "CE requires a fixed mapping between input timesteps and target characters. In OCR, we don't know which image column corresponds to which character — the alignment is unknown.",
        "CE only works with binary classification",
        "Softmax can't handle more than 10 classes"
      ],
      correctIndex: 1,
      explanation: "Cross-entropy computes loss per-position: loss_t = -log(y_t^{target_t}). But in OCR, there's no target_t for each timestep — we only know the full string, not the alignment. CTC solves this by marginalizing over all possible alignments.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "For target 'ab' with extended label [-, a, -, b, -], how many states does the CTC forward algorithm track per timestep?",
      code: "target_len = 2  # 'ab'\nextended_len = 2 * target_len + 1\nprint(extended_len)",
      options: ["5", "4", "3", "2"],
      correctIndex: 0,
      explanation: "Extended label inserts blanks between and around every character: [-, a, -, b, -]. Length = 2S + 1 where S = original label length. For 'ab' (S=2): 2×2+1 = 5 states tracked at each timestep.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What role does post-processing (language model, dictionary lookup) play in the OCR pipeline?",
      options: [
        "It replaces CTC decoding entirely",
        "It corrects CTC errors using linguistic knowledge — 'rnake' → 'snake', 'thc' → 'the'. CTC produces character-level predictions; post-processing adds word-level intelligence.",
        "It speeds up inference",
        "It's only needed during training"
      ],
      correctIndex: 1,
      explanation: "CTC operates at the character level with no language understanding. Post-processing leverages dictionaries and language models to fix plausible errors, resolve ambiguities, and produce coherent text. It's the bridge from characters to meaning.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "A CTC-trained OCR model consistently merges repeated characters ('ee' → 'e'). What is the most likely cause?",
      options: [
        "The learning rate is too high",
        "The model isn't learning to emit blank tokens between repeated characters. Without blanks, consecutive same-character predictions collapse to a single character during decoding.",
        "The vocabulary is too small",
        "Greedy decoding is fundamentally broken"
      ],
      correctIndex: 1,
      explanation: "CTC requires blank emissions between repeated characters to preserve them after decoding. If the model never predicts blank between two 'e' regions, greedy decode collapses them. Fix: ensure sufficient training data with repeated characters and verify blank predictions exist.",
      randomize: true
    }
  ]}
/>
