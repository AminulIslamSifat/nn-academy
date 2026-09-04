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
      prompt: "Why divide QKᵀ by sqrt(d_k) in scaled dot-product attention?",
      options: ["To reduce parameter count", "Dot product variance grows with d_k; scaling keeps scores numerically stable and prevents softmax saturation", "To make attention symmetric", "Because values must be normalized"],
      correctIndex: 1,
      explanation: "A dot product sums d_k terms, so its variance grows with d_k. Large scores make softmax nearly one-hot, killing gradients. Dividing by sqrt(d_k) keeps the variance roughly constant.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a causal mask prevent?",
      options: ["Overfitting", "A token attending to future tokens during autoregressive generation", "Gradient explosion", "Repeated words"],
      correctIndex: 1,
      explanation: "In language generation, token t must only depend on tokens < t. A causal mask sets future attention scores to -inf, making their softmax probability zero.",
      randomize: false,
    }
  ]}
/>
