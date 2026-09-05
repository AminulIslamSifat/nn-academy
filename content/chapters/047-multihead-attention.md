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
      options: [
        "Faster computation",
        "Multiple heads can attend to different types of relationships simultaneously (syntax, coreference, position); one head would average everything into a single blurred distribution",
        "To reduce memory usage",
        "Because softmax requires multiple passes"
      ],
      correctIndex: 1,
      explanation: "Each head has its own Q/K/V projections and learns a distinct attention pattern. One head collapses all relationship types into a single averaged distribution, losing the ability to represent diverse linguistic phenomena.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What role does the residual connection play in transformer blocks?",
      options: [
        "It reduces parameter count",
        "It provides a direct gradient path via identity addition, enabling training of very deep networks by ensuring gradients always flow even if sublayers degrade",
        "It normalizes activations",
        "It implements dropout"
      ],
      correctIndex: 1,
      explanation: "x + sublayer(x) creates an identity shortcut. Gradients flow through the addition unchanged, bypassing potentially problematic sublayers. This is why Transformers can be stacked 12-96 layers deep.",
      randomize: false
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "With D=512 and num_heads=8, what is d_k (dimension per head)?",
      options: ["64", "512", "8", "256"],
      correctIndex: 0,
      explanation: "d_k = D / num_heads = 512 / 8 = 64. Each head operates in a 64-dim subspace. Total computation is similar to single-head with D=512, but distributed across specialized subspaces.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "In multi-head attention, why reshape Q from (N, T, D) to (N, H, T, d_k)?",
      options: [
        "To reduce memory",
        "To separate the D-dimensional representation into H independent d_k-dimensional subspaces, allowing each head to compute attention independently in its own subspace",
        "NumPy requires this shape for matmul",
        "To enable batch processing"
      ],
      correctIndex: 1,
      explanation: "The reshape splits the full representation into H chunks of d_k dimensions each. Transposing to (N, H, T, d_k) puts heads as a batch dimension, enabling parallel attention computation across all heads simultaneously.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "After concatenating H heads of shape (N, H, T, d_k), what shape results from transpose+reshape?",
      code: "N, H, T, d_k = 2, 8, 10, 64\nout = np.zeros((N, H, T, d_k))\nconcat = out.transpose(0, 2, 1, 3).reshape(N, T, H * d_k)\nprint(concat.shape)",
      options: ["(2, 10, 512)", "(2, 8, 640)", "(10, 2, 512)", "(2, 10, 64)"],
      correctIndex: 0,
      explanation: "Transpose to (N, T, H, d_k), then reshape to (N, T, H×d_k) = (2, 10, 512). This merges all heads back into a single D-dimensional representation, ready for the output projection Wo.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What does layer normalization normalize over?",
      options: [
        "Across the batch dimension",
        "Across the feature dimension (last axis) — each position's D-dimensional vector is independently normalized to zero mean and unit variance",
        "Across the time dimension",
        "Across all dimensions simultaneously"
      ],
      correctIndex: 1,
      explanation: "LayerNorm computes mean and variance per-position across features: μ = mean(x_d), σ² = var(x_d). Unlike BatchNorm (normalizes across samples), LayerNorm is sample-independent, making it suitable for variable-length sequences.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "A transformer block consists of two sublayers: multi-head self-attention and ___, each wrapped in residual connection + ___.",
      options: ["feed-forward network, layer normalization", "convolution, batch normalization", "pooling, dropout", "embedding, softmax"],
      correctIndex: 0,
      explanation: "Attention captures inter-token relationships. The FFN (two linear layers + ReLU) processes each position independently, adding non-linearity and capacity. Both are stabilized by residual + LayerNorm.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What is the difference between pre-norm and post-norm in transformer blocks?",
      options: [
        "Pre-norm uses BatchNorm, post-norm uses LayerNorm",
        "Post-norm: LN(x + sublayer(x)). Pre-norm: x + sublayer(LN(x)). Pre-norm normalizes BEFORE the sublayer, providing more stable gradients for deep networks.",
        "They're mathematically identical",
        "Pre-norm is slower but more accurate"
      ],
      correctIndex: 1,
      explanation: "Pre-norm places LayerNorm before each sublayer, ensuring the sublayer receives normalized inputs. This prevents activation explosion in deep stacks and has become the modern default. Post-norm was used in the original paper.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why does the feed-forward network use TWO linear layers with ReLU instead of one?",
      options: [
        "Two layers are faster",
        "One linear layer is just another linear transformation — no additional expressiveness. Two layers with ReLU create a universal function approximator that can learn non-linear per-position transformations.",
        "It's required by the attention mechanism",
        "Single linear layers cause gradient explosion"
      ],
      correctIndex: 1,
      explanation: "A single linear layer composed with attention's linear operations is still linear overall. The ReLU nonlinearity between two linear layers enables the FFN to learn arbitrary non-linear functions of each position's representation.",
      randomize: true
    },
    {
      id: "q10",
      type: "ordering",
      prompt: "Order the operations in one transformer block (pre-norm variant):",
      items: ["Layer norm the input", "Multi-head self-attention", "Add residual (x + attn_out)", "Layer norm again", "Feed-forward network", "Add residual again"],
      correctOrder: [0, 1, 2, 3, 4, 5],
      explanation: "Pre-norm: LN → Attention → Residual → LN → FFN → Residual. Each sublayer receives normalized input and its output is added back to preserve the original signal.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What does the output projection Wo do after concatenating attention heads?",
      options: [
        "It's optional and can be removed",
        "It linearly combines information from all heads into a coherent D-dimensional output, learning how to integrate diverse attention patterns",
        "It applies softmax",
        "It normalizes the concatenated output"
      ],
      correctIndex: 1,
      explanation: "Concatenation simply stacks head outputs. Wo (D×D matrix) learns a weighted combination across heads, enabling the model to selectively emphasize or suppress information from different attention patterns.",
      randomize: true
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What does layer_norm produce for this input?",
      code: "import numpy as np\nx = np.array([2.0, 4.0, 6.0])\nmu = x.mean()\nvar = x.var()\nx_norm = (x - mu) / np.sqrt(var + 1e-5)\nprint(x_norm.round(3))",
      options: ["[-1.225  0.     1.225]", "[0. 0. 0.]", "[2. 4. 6.]", "[-1. 0. 1.]"],
      correctIndex: 0,
      explanation: "mean=4, var=8/3≈2.667, std≈1.633. (2-4)/1.633=-1.225, (4-4)/1.633=0, (6-4)/1.633=1.225. LayerNorm centers and scales each feature vector to zero mean, unit variance.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "How many parameters does multi-head attention have (excluding biases) with D=512?",
      options: ["512²", "4 × 512² (Wq + Wk + Wv + Wo, each D×D)", "8 × 512²", "512"],
      correctIndex: 1,
      explanation: "Four D×D projection matrices: Wq, Wk, Wv (input projections) and Wo (output projection). Total: 4 × D² = 4 × 262,144 = ~1M params. Number of heads doesn't change total params — it redistributes them.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why does increasing num_heads beyond ~12-16 show diminishing returns?",
      options: [
        "Hardware limitations",
        "Each head gets fewer dimensions (d_k = D/H). Too many heads means each operates in too small a subspace to capture meaningful patterns. There's an optimal trade-off between specialization and capacity per head.",
        "Softmax becomes unstable",
        "More heads always improve performance"
      ],
      correctIndex: 1,
      explanation: "With D=512 and H=32, each head has only d_k=16 dimensions — too few to represent complex relationships. With H=8, d_k=64 provides ample capacity. The sweet spot balances diversity of patterns with sufficient per-head expressiveness.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "The feed-forward network expands the representation to ___ × D (typically 4×), applies ReLU, then projects back to D.",
      options: ["4", "2", "1", "8"],
      correctIndex: 0,
      explanation: "FFN: D → 4D → D. The expansion provides a higher-dimensional space for non-linear processing before compressing back. The 4× factor is a standard hyperparameter balancing capacity and efficiency.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What happens if you remove residual connections from a 12-layer Transformer?",
      options: [
        "Nothing significant",
        "Training fails or degrades severely — gradients must pass through 12 sequential sublayers without shortcuts, causing vanishing/exploding gradients and making optimization nearly impossible",
        "The model trains faster",
        "Performance improves due to simpler architecture"
      ],
      correctIndex: 1,
      explanation: "Residual connections are essential for deep networks. Without them, each layer's gradient must flow through all subsequent layers multiplicatively. The identity path ensures stable gradient flow regardless of depth.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "In trained Transformers, some attention heads become interpretable. What patterns have researchers observed?",
      options: [
        "All heads learn identical patterns",
        "Specific heads specialize: attending to previous token, matching brackets, subject-verb agreement, positional distance. Others remain distributed with no clear interpretation.",
        "Heads only learn random patterns",
        "Interpretability is impossible for attention"
      ],
      correctIndex: 1,
      explanation: "Research (Vig 2019, Clark 2019) found heads specializing in syntactic relations, coreference, positional patterns, and more. However, many heads remain diffuse. Interpretability varies across models and layers.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Why does LayerNorm work better than BatchNorm for Transformers?",
      options: [
        "LayerNorm is faster",
        "BatchNorm normalizes across the batch dimension, which is problematic for variable-length sequences and small batches. LayerNorm normalizes per-sample across features, independent of batch composition.",
        "BatchNorm doesn't support GPUs",
        "There's no difference"
      ],
      correctIndex: 1,
      explanation: "Transformers process variable-length sequences with padding. BatchNorm statistics vary with batch composition and padding masks. LayerNorm's per-sample normalization is consistent regardless of batch contents or sequence lengths.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What is the total parameter count of one transformer block with D=512 and FFN hidden dim=2048?",
      options: [
        "~1M",
        "~7M: attention (4×512² ≈ 1M) + FFN (512×2048 + 2048×512 ≈ 2M) + LayerNorm (2×512×2 ≈ 2K) + biases",
        "~100K",
        "~50M"
      ],
      correctIndex: 1,
      explanation: "Attention: 4×D² = ~1M. FFN: D×4D + 4D×D = 2×D×4D = ~2M. LayerNorm: 2×2D = ~2K. Plus biases. Total ≈ 3M weights per block. A 12-layer model: ~36M params from blocks alone.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "BERT, GPT, and T5 are all based on the Transformer block. How do they differ?",
      options: [
        "They use completely different architectures",
        "Same building block, different configurations: BERT = encoder-only (bidirectional), GPT = decoder-only (causal), T5 = encoder-decoder. All stack the same attention+FFN+residual+LN template.",
        "Only GPT uses attention",
        "BERT uses RNNs internally"
      ],
      correctIndex: 1,
      explanation: "The Transformer block is universal. BERT stacks encoder blocks (no causal mask). GPT stacks decoder blocks (causal mask). T5 uses both. The 'everything is a Transformer' insight means understanding one block unlocks all modern NLP.",
      randomize: true
    }
  ]}
/>
