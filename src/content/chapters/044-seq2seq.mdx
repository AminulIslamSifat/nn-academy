---
title: "Sequence-to-Sequence Models"
slug: "044-seq2seq"
description: "Map variable-length input sequences to variable-length output sequences. Build encoder-decoder architecture with teacher forcing in NumPy."
track: "nn-intermediate"
order: 9
read_time: 25
code_time: 20
execution_timeout: 15
prerequisites: ["042-lstm-gru", "043-word-embeddings"]
---

# Sequence-to-Sequence Models

Many tasks map sequences to sequences: translation, summarization, chat. ==Seq2seq uses an encoder to compress the input and a decoder to generate the output==, connected by a context vector.

## Architecture

```
Encoder (LSTM):  x₁ x₂ x₃ `[EOS]` → context vector c
Decoder (LSTM):  c → ŷ₁ ŷ₂ ŷ₃ `[EOS]`
```

The encoder reads the entire input into a fixed-size context vector. The decoder generates output one token at a time, conditioned on the context and previous outputs.

## Encoder

```python
import numpy as np

def encoder_forward(X, h0, c0, params):
    """Process input sequence, return final hidden state as context.
    X: (T_in, N, D)
    Returns: context_h (N, H), context_c (N, H)
    """
    T, N, D = X.shape
    H = params["Wf"].shape[1]
    
    h, c = h0, c0
    for t in range(T):
        h, c = lstm_cell(X[t], h, c, params)
    
    return h, c  # final state = context
```

## Decoder with Teacher Forcing

During training, feed the TRUE previous token (not the predicted one). This stabilizes early training:

```python
def decoder_forward(y_true, context_h, context_c, params, embed_W):
    """y_true: (T_out, N) target token IDs (shifted right).
    Returns: logits (T_out, N, V)
    """
    T, N = y_true.shape
    H = context_h.shape[1]
    V = embed_W.shape[0]
    
    h, c = context_h, context_c
    all_logits = []
    
    for t in range(T):
        # Embed the true previous token (teacher forcing)
        x_t = embed_W[y_true[t]]  # (N, D)
        
        # LSTM step
        h, c = lstm_cell(x_t, h, c, params)
        
        # Project to vocabulary
        logits = h @ params["Wy"] + params["by"]  # (N, V)
        all_logits.append(logits)
    
    return np.stack(all_logits)  # (T_out, N, V)
```

<PyRunner
  cellId="044-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

# Simulate seq2seq for simple reversal task
# Input: [1,2,3] → Output: [3,2,1]
V, D, H = 10, 4, 8

print("Seq2Seq Architecture:")
print(f"  Vocab size: {V}")
print(f"  Embedding dim: {D}")
print(f"  Hidden dim: {H}")

# Simulated encoding
T_in = 3
input_seq = [1, 2, 3]
print(f"\n  Input:  {input_seq}")

# Encoder processes input → context
context = np.random.randn(H) * 0.5  # simulated final hidden state
print(f"  Context vector shape: ({H},)")
print(f"  Context encodes entire input into fixed representation")

# Decoder generates output from context
T_out = 3
output_seq = [3, 2, 1]
print(f"  Output: {output_seq}")

print(f"\n💡 Bottleneck: ALL input info compressed into {H}-dim vector")
print(f"   This limits performance for long sequences")
print(f"   Solution: Attention mechanism (next chapter!)")
`}
/>

## Training with Cross-Entropy

```python
def seq2seq_loss(logits, y_true):
    """logits: (T, N, V), y_true: (T, N)
    Returns: scalar loss
    """
    T, N, V = logits.shape
    
    # Softmax
    shifted = logits - np.max(logits, axis=2, keepdims=True)
    exp_z = np.exp(shifted)
    probs = exp_z / np.sum(exp_z, axis=2, keepdims=True)
    
    # Cross-entropy
    loss = 0.0
    for t in range(T):
        for n in range(N):
            loss -= np.log(probs[t, n, y_true[t, n]] + 1e-7)
    
    return loss / (T * N)
```

> [!IMPORTANT] Teacher Forcing Trade-off
> Teacher forcing makes training faster and more stable, but creates a train/test mismatch: during inference, the decoder sees its OWN predictions, not ground truth. Solutions: scheduled sampling, professor forcing.

## Inference: Greedy vs Beam Search

### Greedy Decoding
Pick the most likely token at each step:

```python
def greedy_decode(encoder_context, decoder_params, embed_W, max_len=50, eos_id=0):
    h, c = encoder_context
    tokens = []
    token_id = 1  # start token
    
    for _ in range(max_len):
        x = embed_W[token_id]
        h, c = lstm_cell(x, h, c, decoder_params)
        logits = h @ decoder_params["Wy"] + decoder_params["by"]
        token_id = np.argmax(logits)
        if token_id == eos_id:
            break
        tokens.append(token_id)
    
    return tokens
```

### Beam Search
Maintain K best partial sequences. Better quality than greedy:

<PyRunner
  cellId="044-cell-2"
  defaultCode={`import numpy as np
np.random.seed(42)

# Demonstrate beam search concept
V = 6
beam_width = 3
max_len = 4

print("Beam Search (k=3) Demo:")
print(f"{'Step':>4} | Active Beams")
print("─" * 50)

# Simulated beam search
beams = [([1], 0.0)]  # (tokens, log_prob)

for step in range(max_len):
    candidates = []
    for tokens, score in beams:
        # Simulate next-token probabilities
        probs = np.random.dirichlet(np.ones(V))
        for v in range(V):
            if v == 0:  # EOS
                candidates.append((tokens + [v], score + np.log(probs[v])))
            else:
                candidates.append((tokens + [v], score + np.log(probs[v])))
    
    # Keep top-k
    candidates.sort(key=lambda x: -x[1])
    beams = candidates[:beam_width]
    
    display = [f"{b[0]}({b[1]:.2f})" for b in beams]
    print(f"{step+1:4d} | {', '.join(display)}")

best = beams[0]
print(f"\nBest sequence: {best[0]}, score: {best[1]:.2f}")
print(f"\n✅ Beam search explores multiple hypotheses")
print(f"   Greedy would only follow the single best path")
`}
/>

## The Context Vector Bottleneck

The biggest limitation: ==everything must fit in one fixed vector==. For long inputs, information is lost. This motivated attention mechanisms (chapter 046).

| Sequence Length | Encoder Capacity | Information Loss |
|----------------|-----------------|-----------------|
| Short (under 20) | Sufficient | Minimal |
| Medium (20-50) | Strained | Noticeable |
| Long (>50) | Insufficient | Severe |

<Quiz
  chapterSlug="044-seq2seq"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is teacher forcing and why is it used?",
      options: [
        "Forcing the model to predict correctly",
        "Feeding ground-truth previous tokens during training instead of predicted ones, preventing error accumulation in early training",
        "Using a stronger optimizer",
        "Training the encoder and decoder separately"
      ],
      correctIndex: 1,
      explanation: "Without teacher forcing, early-training predictions are garbage, and feeding garbage back produces more garbage. Teacher forcing provides clean inputs so the decoder learns correct patterns first.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the main limitation of basic seq2seq?",
      options: [
        "It can't handle variable-length sequences",
        "The fixed-size context vector is a bottleneck — long inputs lose information because everything must be compressed into one vector",
        "It requires too many parameters",
        "It only works for translation"
      ],
      correctIndex: 1,
      explanation: "A 256-dim context vector can't faithfully encode a 100-token sentence. Attention solves this by letting the decoder access ALL encoder states directly, not just the final one.",
      randomize: false,
    }
  ]}
/>
