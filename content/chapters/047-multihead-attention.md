---
title: "Multi-Head Attention & Transformer Blocks"
slug: "047-multihead-attention"
description: "Why one attention head is not enough. Implement multi-head attention, residual connections, layer normalization, and the full transformer block in NumPy."
track: "nn-advanced"
order: 2
read_time: 30
code_time: 25
execution_timeout: 15
prerequisites: ["046-attention-mechanism"]
---

# Multi-Head Attention & Transformer Blocks

Single-head attention learns one kind of relationship. ==Multi-head attention runs several attention heads in parallel==, each learning a different pattern: syntax, coreference, position, semantics.

## Multi-Head Attention

Split Q, K, V into H heads, run attention independently, concatenate:

<BlockMath latex="\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) W^O" />

where each head uses <InlineMath latex="d_k = d_{model}/H" /> dimensions.

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """X: (N, T, D), W*: (D, D)"""
    N, T, D = X.shape
    d_k = D // num_heads
    
    Q = X @ Wq  # (N, T, D)
    K = X @ Wk
    V = X @ Wv
    
    # Reshape to (N, H, T, d_k)
    Q = Q.reshape(N, T, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(N, T, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(N, T, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Scaled dot-product per head
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    attn_out = weights @ V  # (N, H, T, d_k)
    
    # Concatenate heads
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(N, T, D)
    return attn_out @ Wo, weights
```

<PyRunner
  cellId="047-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

N, T, D, H = 2, 6, 16, 4
d_k = D // H
X = np.random.randn(N, T, D)
Wq = np.random.randn(D, D) * 0.3
Wk = np.random.randn(D, D) * 0.3
Wv = np.random.randn(D, D) * 0.3
Wo = np.random.randn(D, D) * 0.3

Q = (X @ Wq).reshape(N, T, H, d_k).transpose(0,2,1,3)
K = (X @ Wk).reshape(N, T, H, d_k).transpose(0,2,1,3)
scores = Q @ K.transpose(0,1,3,2) / np.sqrt(d_k)
weights = softmax(scores, axis=-1)

print(f"Multi-Head Attention shapes:")
print(f"  Input:     {X.shape}")
print(f"  Per head:  ({N}, {H}, {T}, {d_k})")
print(f"  Scores:    {scores.shape} = (batch, heads, query_pos, key_pos)")
print(f"  Weights:   {weights.shape}")

for h in range(H):
    entropy = -np.mean(np.sum(weights[0,h]*np.log(weights[0,h]+1e-8), axis=1))
    print(f"  Head {h} avg entropy: {entropy:.3f}")

print("\n💡 Different heads learn different attention patterns")
`}
/>

## Why Multiple Heads?

One head averages all relationships. Multiple heads specialize:

| Head Type | Learns |
|-----------|--------|
| Positional | Adjacent tokens, distance patterns |
| Syntactic | Subject-verb, modifier-head |
| Coreference | Pronoun → antecedent |
| Semantic | Related concepts |

> [!NOTE] Empirical Finding
> In trained transformers, some heads become interpretable (attending to previous token, matching brackets, etc.), while others remain distributed. More heads ≠ always better; diminishing returns past ~12-16 heads.

## The Transformer Block

Attention alone isn't enough. Each block adds:

1. **Residual connection**: <InlineMath latex="x + \text{sublayer}(x)" />
2. **Layer normalization**: stabilize training
3. **Feed-forward network**: per-position processing

```python
def layer_norm(X, gamma, beta, eps=1e-5):
    """X: (N, T, D)"""
    mu = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)
    X_norm = (X - mu) / np.sqrt(var + eps)
    return gamma * X_norm + beta

def feed_forward(X, W1, b1, W2, b2):
    """Two linear layers with ReLU."""
    return np.maximum(0, X @ W1 + b1) @ W2 + b2

def transformer_block(X, params, mask=None):
    # Self-attention sub-block
    attn_out, _ = multi_head_attention(
        X, params["Wq"], params["Wk"], params["Wv"], params["Wo"],
        params["num_heads"], mask
    )
    X = layer_norm(X + attn_out, params["ln1_g"], params["ln1_b"])
    
    # Feed-forward sub-block
    ff_out = feed_forward(X, params["W1"], params["b1"], params["W2"], params["b2"])
    X = layer_norm(X + ff_out, params["ln2_g"], params["ln2_b"])
    
    return X
```

<PyRunner
  cellId="047-cell-2"
  defaultCode={`import numpy as np
np.random.seed(42)

N, T, D = 2, 8, 32
X = np.random.randn(N, T, D) * 2.0

# Layer norm demo
gamma = np.ones(D)
beta = np.zeros(D)
mu = X.mean(axis=-1, keepdims=True)
var = X.var(axis=-1, keepdims=True)
X_norm = (X - mu) / np.sqrt(var + 1e-5)
out = gamma * X_norm + beta

print("Layer Normalization:")
print(f"  Before: mean={X[0,0].mean():+.3f}, std={X[0,0].std():.3f}")
print(f"  After:  mean={out[0,0].mean():+.6f}, std={out[0,0].std():.3f}")

# Residual connection demo
residual = X + out
print(f"\nResidual Connection:")
print(f"  Original signal preserved via addition")
print(f"  Gradient can flow through identity path even if sublayer fails")
print(f"  This is why deep transformers train at all!")
`}
/>

## Pre-Norm vs Post-Norm

- **Post-Norm** (original paper): `LN(x + sublayer(x))`
- **Pre-Norm** (modern default): `x + sublayer(LN(x))`

Pre-Norm is more stable for deep networks. Most modern implementations use it.

> [!IMPORTANT] Architecture Summary
> A Transformer = stack of identical blocks. Each block = multi-head self-attention + feed-forward, both wrapped in residual + layer norm. That's it. Everything else (BERT, GPT, T5) is just variations on this template.

<Quiz
  chapterSlug="047-multihead-attention"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why do transformers use multiple attention heads instead of one large head?",
      options: ["Faster computation", "Multiple heads can attend to different types of relationships simultaneously; one head would average everything together", "To reduce memory usage", "Because softmax requires it"],
      correctIndex: 1,
      explanation: "Each head has its own Q/K/V projections and learns a different attention pattern. One head would collapse all relationship types into a single averaged distribution.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What role does the residual connection play in transformer blocks?",
      options: ["It reduces parameter count", "It provides a direct gradient path, enabling training of very deep networks by preventing vanishing gradients through the sublayers", "It normalizes activations", "It implements dropout"],
      correctIndex: 1,
      explanation: "The identity shortcut means gradients always have an unimpeded path backward, regardless of how complex or degraded the sublayer becomes. This is essential for stacking dozens of blocks.",
      randomize: false,
    }
  ]}
/>
