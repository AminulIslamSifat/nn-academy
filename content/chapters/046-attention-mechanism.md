---
title: "Attention Mechanism"
slug: "046-attention-mechanism"
description: "Break the fixed-context bottleneck. Implement query-key-value attention, attention weights, and context vectors from scratch in NumPy."
track: "nn-advanced"
order: 1
read_time: 28
code_time: 22
execution_timeout: 15
prerequisites: ["044-seq2seq"]
---

# Attention Mechanism

Seq2seq compresses the whole input into one vector. That bottleneck destroys long-range information. ==Attention lets the decoder look back at every encoder state and decide what matters right now.==

Instead of one fixed context vector, we compute a fresh context vector at each decoder step.

## The Core Idea

For a decoder state <InlineMath latex="q" /> and encoder states <InlineMath latex="K,V" />:

<BlockMath latex="\text{score}_i = q \cdot k_i" />
<BlockMath latex="\alpha_i = \text{softmax}(\text{score})_i" />
<BlockMath latex="\text{context} = \sum_i \alpha_i v_i" />

In words:

- **Query**: what am I looking for?
- **Key**: what does each source position contain?
- **Value**: what information should I retrieve?

<Visualizer mode="matmul" shapeA={[8, 64]} shapeB={[64, 8]} title="Attention: Q(T×D) @ K(D×T) → scores(T×T)" />

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def dot_attention(query, keys, values):
    """query: (D,), keys: (T,D), values: (T,Dv)"""
    scores = keys @ query
    weights = softmax(scores)
    context = weights @ values
    return context, weights
```

<PyRunner
  cellId="046-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

sentence = ["the", "cat", "sat", "down"]
T, D = len(sentence), 6
keys = np.random.randn(T, D)
values = np.random.randn(T, D)
query = keys[1] + np.random.randn(D) * 0.2  # similar to "cat"

scores = keys @ query
weights = softmax(scores)
context = weights @ values

print("Attention weights:")
for word, score, w in zip(sentence, scores, weights):
    bar = "█" * int(w * 30)
    print(f"  {word:>5}: score={score:+6.3f}, weight={w:.3f} {bar}")

print(f"\nContext vector shape: {context.shape}")
print("💡 Query was similar to 'cat', so attention focuses there")
`}
/>

## Scaled Dot-Product Attention

For high-dimensional vectors, dot products grow large. Softmax saturates, gradients vanish. Scale by <InlineMath latex="\sqrt{d_k}" />:

<BlockMath latex="\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
```

<PyRunner
  cellId="046-cell-2"
  defaultCode={`import numpy as np
np.random.seed(0)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

seq_len, d_k, d_v = 5, 16, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

scores_unscaled = Q @ K.T
scores_scaled = scores_unscaled / np.sqrt(d_k)
w_unscaled = softmax(scores_unscaled, axis=1)
w_scaled = softmax(scores_scaled, axis=1)

print("Effect of scaling:")
print(f"  Unscaled score std: {scores_unscaled.std():.3f}")
print(f"  Scaled score std:   {scores_scaled.std():.3f}")
print(f"  Unscaled entropy:   {-np.mean(np.sum(w_unscaled*np.log(w_unscaled+1e-8), axis=1)):.3f}")
print(f"  Scaled entropy:     {-np.mean(np.sum(w_scaled*np.log(w_scaled+1e-8), axis=1)):.3f}")
print("\nScaling prevents attention from becoming too peaky too early.")
`}
/>

## Attention Masking

For generation, a token must not see future tokens. Causal mask:

```python
def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))
```

<PyRunner
  cellId="046-cell-3"
  defaultCode={`import numpy as np

T = 6
mask = np.tril(np.ones((T, T), dtype=int))
print("Causal mask (1 = allowed, 0 = blocked):")
print(mask)
print("\nRow t can only attend to columns <= t")
`}
/>

## Additive Attention

Older seq2seq models often used additive attention:

<BlockMath latex="e_{t,i} = v^T \tanh(W_q q_t + W_k k_i)" />

It uses a small neural network to compute compatibility instead of a dot product. Dot-product attention is faster and became standard in Transformers.

## Why Attention Changed Everything

Attention gives direct paths between any two positions. Distance no longer matters. In RNNs, information from token 1 to token 100 travels through 99 recurrent steps. With attention, token 100 can directly attend to token 1 in one operation.

> [!IMPORTANT] Key Takeaway
> Attention is differentiable memory retrieval. Queries ask questions, keys are addresses, values are content. Softmax turns similarity scores into a weighted read from memory.

<Quiz
  chapterSlug="046-attention-mechanism"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why divide QKᵀ by √d_k in scaled dot-product attention?",
      options: [
        "To reduce parameter count",
        "Dot product variance grows linearly with d_k; scaling keeps scores in a range where softmax produces non-saturated distributions with healthy gradients",
        "To make attention symmetric",
        "Because values must be normalized first"
      ],
      correctIndex: 1,
      explanation: "A dot product of two d_k-dim vectors has variance proportional to d_k. Without scaling, large d_k produces extreme scores → near-one-hot softmax → vanishing gradients. Dividing by √d_k normalizes variance to ~1.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a causal mask prevent?",
      options: [
        "Overfitting to training data",
        "A token attending to future tokens during autoregressive generation — it enforces left-to-right information flow",
        "Gradient explosion in deep networks",
        "Repeated word generation"
      ],
      correctIndex: 1,
      explanation: "In language generation, token t must only depend on tokens ≤ t. A causal mask sets future attention scores to -∞ before softmax, making their weights exactly zero.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In attention, what do Query, Key, and Value represent conceptually?",
      options: [
        "Input, hidden state, output",
        "Query = what am I looking for, Key = what does each position contain (address), Value = what information to retrieve (content)",
        "Encoder, decoder, context",
        "Weights, biases, activations"
      ],
      correctIndex: 1,
      explanation: "Analogy to database retrieval: query is the search term, keys are indexed addresses, values are stored content. Attention computes similarity between query and each key, then retrieves a weighted combination of values.",
      randomize: true
    },
    {
      id: "q4",
      type: "shape-prediction",
      prompt: "If Q has shape (T_q, D) and K has shape (T_k, D), what shape is the attention score matrix Q @ K.T?",
      options: ["(T_q, T_k)", "(D, D)", "(T_q, D)", "(T_k, T_q)"],
      correctIndex: 0,
      explanation: "Q(T_q, D) @ K.T(D, T_k) = (T_q, T_k). Each entry [i,j] is the similarity between query i and key j. This matrix determines how much each query attends to each key.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What does np.tril produce for T=4?",
      code: "import numpy as np\nprint(np.tril(np.ones((4,4), dtype=int)))",
      options: ["[[1 0 0 0]\n [1 1 0 0]\n [1 1 1 0]\n [1 1 1 1]]", "[[1 1 1 1]\n [0 1 1 1]\n [0 0 1 1]\n [0 0 0 1]]", "Identity matrix", "All ones"],
      correctIndex: 0,
      explanation: "np.tril creates a lower triangular matrix. Row t has 1s for columns ≤ t and 0s for columns > t. This is the causal mask: position t can attend to positions 0..t but not t+1..T.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "How does attention solve the context vector bottleneck of basic seq2seq?",
      options: [
        "It uses a larger context vector",
        "Instead of compressing everything into one fixed vector, attention computes a fresh context at each decoder step by directly accessing all encoder states. No information needs to be pre-compressed.",
        "It removes the encoder entirely",
        "It uses multiple context vectors in parallel"
      ],
      correctIndex: 1,
      explanation: "The decoder queries relevant encoder states at each generation step. Long-range dependencies are handled in O(1) instead of O(T) recurrent steps. Information flows directly from any encoder position to any decoder position.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "Attention weights α_i are computed via ___ over the raw scores, ensuring they sum to ___ and are all non-negative.",
      options: ["softmax, 1", "sigmoid, 0", "ReLU, 1", "normalization, 0"],
      correctIndex: 0,
      explanation: "Softmax converts arbitrary real-valued scores into a valid probability distribution: all positive, summing to 1. This makes attention weights interpretable as 'how much to focus on each position'.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What happens if you forget to apply the causal mask during autoregressive generation training?",
      options: [
        "Nothing — the model still works",
        "The model cheats by attending to future tokens it shouldn't see. Training loss looks great but inference fails because future tokens aren't available at generation time.",
        "Training becomes slower",
        "Gradients explode"
      ],
      correctIndex: 1,
      explanation: "Without masking, the model learns to copy future tokens directly rather than predicting from past context. This train/test mismatch causes catastrophic failure at inference when future tokens don't exist.",
      randomize: true
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is the effect of setting masked positions to -1e9 before softmax?",
      code: "import numpy as np\nscores = np.array([1.0, 2.0, -1e9, -1e9])\nshifted = scores - scores.max()\nprobs = np.exp(shifted) / np.exp(shifted).sum()\nprint(probs.round(4))",
      options: ["[0.2689 0.7311 0.     0.    ]", "[0.25 0.25 0.25 0.25]", "[0. 0. 0.5 0.5]", "Error"],
      correctIndex: 0,
      explanation: "exp(-1e9) ≈ 0, so masked positions get zero probability. The remaining positions share the probability mass normally. This is how causal/padding masks work in practice.",
      randomize: true
    },
    {
      id: "q10",
      type: "ordering",
      prompt: "Order the scaled dot-product attention computation:",
      items: ["Compute scores: Q @ K.T", "Scale by 1/√d_k", "Apply mask (if any)", "Softmax over scores", "Weighted sum: weights @ V"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Scores → scale → mask → softmax → aggregate. Scaling before softmax prevents saturation. Masking before softmax ensures blocked positions get zero weight. Final matmul retrieves content.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "How does additive attention differ from dot-product attention?",
      options: [
        "Additive attention is faster",
        "Additive uses a learned neural network (vᵀ·tanh(W_q·q + W_k·k)) to compute compatibility. Dot-product uses simple matrix multiplication. Dot-product is faster and became standard in Transformers.",
        "They're mathematically identical",
        "Additive attention doesn't use softmax"
      ],
      correctIndex: 1,
      explanation: "Additive attention (Bahdanau) learns a compatibility function via a small MLP. Dot-product attention assumes similarity is captured by inner product. Dot-product is computationally cheaper and scales better, winning out in Transformer architectures.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why is attention described as 'differentiable memory retrieval'?",
      options: [
        "It stores data in GPU memory",
        "Keys act as addresses, values as stored content, and softmax-weighted retrieval is fully differentiable — gradients flow through the attention weights back to Q, K, and V",
        "It replaces RAM with neural storage",
        "It's an analogy with no mathematical basis"
      ],
      correctIndex: 1,
      explanation: "Unlike hard memory lookup (non-differentiable), soft attention retrieves a weighted combination of all values. The weighting depends on learnable parameters, so the entire retrieval process can be trained end-to-end via backpropagation.",
      randomize: true
    },
    {
      id: "q13",
      type: "shape-prediction",
      prompt: "If attention weights have shape (T_q, T_k) and V has shape (T_k, D_v), what shape is the attention output (weights @ V)?",
      options: ["(T_q, D_v)", "(T_k, D_v)", "(T_q, T_k)", "(D_v, T_q)"],
      correctIndex: 0,
      explanation: "(T_q, T_k) @ (T_k, D_v) = (T_q, D_v). Each query position gets its own D_v-dimensional context vector — a weighted combination of all value vectors.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "In RNNs, information from token 1 to token 100 travels through 99 recurrent steps. How does attention change this?",
      options: [
        "It still requires 99 steps but processes them in parallel",
        "Token 100 can directly attend to token 1 in a single operation — distance becomes irrelevant. This eliminates the long-range dependency problem entirely.",
        "It reduces it to log(99) steps",
        "Attention doesn't help with long-range dependencies"
      ],
      correctIndex: 1,
      explanation: "Attention creates direct connections between any two positions regardless of distance. The path length from any token to any other is O(1), compared to O(T) for RNNs. This is why Transformers dominate sequence modeling.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "The context vector is computed as the ___ sum of value vectors, weighted by the attention ___.",
      options: ["weighted, probabilities", "simple, scores", "unweighted, keys", "cumulative, queries"],
      correctIndex: 0,
      explanation: "context = Σ α_i · v_i where α_i are softmax-normalized attention weights. High-weight positions contribute more to the context; low-weight positions contribute less. This is a soft, differentiable selection mechanism.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Your attention weights are nearly uniform (all ≈ 1/T) even after training. What might be wrong?",
      options: [
        "This is expected behavior",
        "The model isn't learning meaningful query-key relationships. Possible causes: poor initialization, learning rate too low, missing scaling factor causing gradient issues, or insufficient training data.",
        "Uniform attention is optimal",
        "The vocabulary is too small"
      ],
      correctIndex: 1,
      explanation: "Uniform attention means the model treats all positions equally — equivalent to simple averaging. Attention should learn selective focus patterns. Check gradient flow through Q/K projections and verify the scaling factor is applied.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is the computational complexity of self-attention for a sequence of length T with dimension D?",
      options: ["O(T × D)", "O(T² × D) — quadratic in sequence length due to the T×T score matrix", "O(T × D²)", "O(D³)"],
      correctIndex: 1,
      explanation: "Computing Q@K.T produces a T×T matrix (O(T²D)), and weights@V also costs O(T²D). This quadratic scaling is why Transformers struggle with very long sequences and motivated efficient attention variants.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "For d_k=64, what is the scaling factor 1/√d_k?",
      code: "import numpy as np\nd_k = 64\nprint(f'{1/np.sqrt(d_k):.4f}')",
      options: ["0.1250", "0.0156", "1.0000", "0.2500"],
      correctIndex: 0,
      explanation: "1/√64 = 1/8 = 0.125. This scales down dot products by 8× for d_k=64, preventing the variance from growing too large and keeping softmax in a well-behaved regime.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Why is self-attention called 'self'?",
      options: [
        "It attends to itself recursively",
        "Queries, keys, and values all come from the SAME sequence — each position attends to all other positions in its own input, unlike cross-attention where Q and K/V come from different sources",
        "It's a self-supervised technique",
        "It only works on single sentences"
      ],
      correctIndex: 1,
      explanation: "In self-attention, Q=K=V=X (same source). Each token looks at every other token in the same sequence. Cross-attention uses Q from one sequence and K,V from another (e.g., decoder attending to encoder).",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You're building a translation model. Where would you use causal masking vs no masking?",
      options: [
        "Causal mask everywhere",
        "Encoder self-attention: NO mask (bidirectional context). Decoder self-attention: CAUSAL mask (no future peeking). Encoder-decoder cross-attention: NO mask (decoder can see all encoder states).",
        "No mask anywhere",
        "Causal mask only in the encoder"
      ],
      correctIndex: 1,
      explanation: "Encoder reads the full input bidirectionally. Decoder generates left-to-right, needing causal masking. Cross-attention lets each decoder position access the complete encoded input. Different masking serves different information flow requirements.",
      randomize: true
    }
  ]}
/>
