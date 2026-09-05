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
        "Forcing the model to always predict correctly",
        "Feeding ground-truth previous tokens during training instead of predicted ones, preventing error accumulation in early training",
        "Using a stronger optimizer",
        "Training the encoder and decoder separately"
      ],
      correctIndex: 1,
      explanation: "Without teacher forcing, early-training predictions are garbage, and feeding garbage back produces more garbage. Teacher forcing provides clean inputs so the decoder learns correct generation patterns first.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the main limitation of basic seq2seq?",
      options: [
        "It can't handle variable-length sequences",
        "The fixed-size context vector is a bottleneck — long inputs lose information because everything must be compressed into one vector",
        "It requires too many parameters",
        "It only works for machine translation"
      ],
      correctIndex: 1,
      explanation: "A 256-dim context vector can't faithfully encode a 100-token sentence. Attention solves this by letting the decoder access ALL encoder states directly, not just the final one.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does the encoder's final hidden state represent in seq2seq?",
      options: [
        "The last input token's embedding",
        "A fixed-size context vector that should encode the entire input sequence's meaning — it's the sole information bridge between encoder and decoder",
        "The average of all input embeddings",
        "A random initialization"
      ],
      correctIndex: 1,
      explanation: "The encoder processes all input tokens sequentially. Its final (h, c) state theoretically summarizes everything. The decoder receives ONLY this vector, making it a critical information bottleneck.",
      randomize: true
    },
    {
      id: "q4",
      type: "shape-prediction",
      prompt: "If the decoder LSTM has hidden dim H=256 and vocabulary size V=10000, what shape are the output logits at each timestep?",
      options: ["(N, 10000)", "(N, 256)", "(T, N, 256)", "(N, 1)"],
      correctIndex: 0,
      explanation: "logits = h @ Wy + by where h is (N, H) and Wy is (H, V). Result: (N, V) = (N, 10000). Each sample gets a score for every vocabulary token.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What train/test mismatch does teacher forcing create?",
      options: [
        "None — training and inference are identical",
        "During training, the decoder sees perfect ground-truth inputs. During inference, it sees its own (possibly wrong) predictions. Errors compound at test time since the model never practiced recovering from mistakes.",
        "Teacher forcing uses different loss functions",
        "The encoder behaves differently"
      ],
      correctIndex: 1,
      explanation: "This exposure bias means the model may perform worse at inference than training metrics suggest. Solutions include scheduled sampling (gradually replace ground truth with predictions) and professor forcing.",
      randomize: true
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "In greedy decoding, what happens when argmax selects the EOS token?",
      code: "eos_id = 0\ntoken_id = 0  # EOS selected\ntokens = [5, 3, 7]\nif token_id == eos_id:\n    print('STOP')\nelse:\n    tokens.append(token_id)\nprint(f'Final: {tokens}')",
      options: ["STOP\nFinal: [5, 3, 7]", "Final: [5, 3, 7, 0]", "Error", "STOP\nFinal: []"],
      correctIndex: 0,
      explanation: "EOS (end-of-sequence) signals generation is complete. The decoder stops and returns accumulated tokens WITHOUT including EOS itself. Without EOS, decoding would continue until max_len.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "Beam search maintains ___ best partial sequences at each step, while greedy decoding follows only ___ path.",
      options: ["K, one", "one, K", "all, one", "K, all"],
      correctIndex: 0,
      explanation: "Beam search keeps K candidates (e.g., K=5), expanding each and retaining the top-K overall. Greedy picks only the single highest-probability token per step. Beam search finds better sequences at the cost of K× computation.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "Why use log probabilities in beam search instead of raw probabilities?",
      options: [
        "Log probabilities are faster to compute",
        "Multiplying many small probabilities causes underflow. Log-space converts products to sums, maintaining numerical precision across long sequences.",
        "Beam search requires negative values",
        "There's no difference"
      ],
      correctIndex: 1,
      explanation: "Sequence probability = product of per-step probabilities. For 20 steps with avg prob 0.3: 0.3^20 ≈ 3.5×10^-11, near float32 limits. Log-probs: 20 × log(0.3) ≈ -24, perfectly representable.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "At what input sequence length does the context vector bottleneck become severe?",
      options: [
        "Always severe regardless of length",
        "Short (<20): minimal loss. Medium (20-50): noticeable. Long (>50): severe information loss. This motivated attention mechanisms.",
        "Only above 1000 tokens",
        "Never — context vectors scale perfectly"
      ],
      correctIndex: 1,
      explanation: "A fixed-dim vector has finite capacity. Short sentences fit easily; long ones require compression that loses detail. Attention bypasses this by giving the decoder direct access to all encoder states.",
      randomize: true
    },
    {
      id: "q10",
      type: "ordering",
      prompt: "Order the seq2seq training pipeline:",
      items: ["Encoder processes input sequence → context vector", "Decoder receives context + teacher-forced targets", "Decoder outputs logits at each timestep", "Cross-entropy loss computed against true targets", "Backprop through both decoder and encoder"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Encode → decode with ground truth → compute per-timestep loss → backprop through entire encoder-decoder chain. Gradients flow from decoder loss back through context vector into encoder.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "How does greedy decoding differ from beam search in terms of output quality?",
      options: [
        "Greedy always produces better results",
        "Greedy makes locally optimal choices at each step but may miss globally better sequences. Beam search explores K hypotheses, often finding higher-probability complete sequences.",
        "They produce identical results",
        "Beam search is always worse due to exploration"
      ],
      correctIndex: 1,
      explanation: "Greedy picks the best token NOW without considering future consequences. A slightly lower-probability token at step 3 might enable much better tokens at steps 4-10. Beam search captures these non-greedy opportunities.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why does the decoder need an embedding layer?",
      options: [
        "To reduce memory usage",
        "Token IDs are integers with no inherent meaning. Embeddings convert them to dense vectors that the LSTM can process meaningfully, just like in the encoder.",
        "LSTMs require float inputs",
        "Embeddings prevent overfitting"
      ],
      correctIndex: 1,
      explanation: "Raw token IDs (0, 1, 2...) have no semantic structure. Embeddings map each ID to a learned D-dimensional vector where similar tokens can have similar representations.",
      randomize: true
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What shape does np.stack(all_logits) produce if all_logits contains T arrays of shape (N, V)?",
      code: "import numpy as np\nT, N, V = 5, 32, 1000\nall_logits = [np.zeros((N, V)) for _ in range(T)]\nstacked = np.stack(all_logits)\nprint(stacked.shape)",
      options: ["(5, 32, 1000)", "(32, 1000, 5)", "(5, 1000)", "(32, 5)"],
      correctIndex: 0,
      explanation: "np.stack creates a new axis at position 0. T arrays of (N, V) → (T, N, V). This gives per-timestep logits for all samples, ready for sequence-level cross-entropy loss.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What is scheduled sampling?",
      options: [
        "Sampling data points uniformly from the dataset",
        "Gradually replacing teacher-forced ground truth with the model's own predictions during training, bridging the train/test gap",
        "Sampling hyperparameters randomly",
        "Using different batch sizes each epoch"
      ],
      correctIndex: 1,
      explanation: "Start with 100% teacher forcing, gradually decrease to e.g., 50%. The model progressively practices with its own outputs, learning to recover from errors before deployment.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Why does beam search normalize scores by sequence length?",
      options: [
        "It doesn't need normalization",
        "Longer sequences accumulate more negative log-probs, making them unfairly penalized. Length normalization (score/length or score/length^α) enables fair comparison across different lengths.",
        "To prevent overflow",
        "To speed up computation"
      ],
      correctIndex: 1,
      explanation: "Without normalization, beam search prefers shorter sequences because they have fewer multiplicative factors reducing the total probability. Length normalization removes this bias toward brevity.",
      randomize: true
    },
    {
      id: "q16",
      type: "fill-blank",
      prompt: "During inference, the decoder generates tokens ___ at a time, using its own previous ___ as input for the next step.",
      options: ["one, prediction", "all, targets", "batch, embeddings", "randomly, states"],
      correctIndex: 0,
      explanation: "Autoregressive generation: predict token t, feed it as input for token t+1. This sequential dependency prevents parallelization during inference (unlike training with teacher forcing).",
      randomize: false
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "You're translating English→French with seq2seq. Input: 30 words. Output quality is poor. What's the most likely cause?",
      options: [
        "French has more words than English",
        "The context vector bottleneck — 30 words compressed into one fixed vector loses too much information. Solution: add attention.",
        "The LSTM hidden size is too large",
        "Teacher forcing is disabled"
      ],
      correctIndex: 1,
      explanation: "30-word sentences exceed basic seq2seq's effective range. The context vector can't preserve enough detail for accurate translation. Attention lets the decoder re-read specific encoder states as needed.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "What tasks can seq2seq handle beyond machine translation?",
      options: [
        "Only translation",
        "Any sequence-to-sequence mapping: summarization, chatbot responses, code generation, speech recognition, image captioning (with CNN encoder)",
        "Only classification tasks",
        "Only fixed-length output tasks"
      ],
      correctIndex: 1,
      explanation: "Seq2seq is a general framework for variable-length input → variable-length output. The encoder adapts to any input modality (text, audio, image), and the decoder generates any sequential output.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "With beam width K=1, beam search is equivalent to:",
      options: ["Random sampling", "Greedy decoding", "Exhaustive search", "Monte Carlo tree search"],
      correctIndex: 1,
      explanation: "K=1 means keeping only the single best candidate at each step — exactly what greedy decoding does. Larger K explores more alternatives at proportional computational cost.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Why must the decoder's initial hidden state come from the encoder rather than being zero-initialized?",
      options: [
        "Zero initialization causes NaN gradients",
        "The encoder's final state carries the encoded input meaning. Starting from zeros would mean the decoder generates output without knowing what the input was.",
        "It's a convention with no practical impact",
        "Zero initialization is actually preferred"
      ],
      correctIndex: 1,
      explanation: "The context vector IS the communication channel between encoder and decoder. Without it, the decoder has no information about the input and would generate generic, input-independent output.",
      randomize: true
    }
  ]}
/>
