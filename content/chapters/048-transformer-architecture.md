---
title: "The Full Transformer Architecture"
slug: "048-transformer-architecture"
description: "Assemble encoder, decoder, positional encoding, and masking into the complete Transformer. Understand encoder-only, decoder-only, and encoder-decoder variants."
track: "nn-advanced"
order: 3
read_time: 28
code_time: 20
execution_timeout: 15
prerequisites: ["047-multihead-attention"]
---

# The Full Transformer Architecture

You have the building blocks. Now assemble them into the architecture that powers BERT, GPT, T5, and every modern LLM.

## Three Variants

| Variant | Architecture | Examples | Use Case |
|---------|-------------|----------|----------|
| Encoder-only | Stacked encoder blocks | BERT, RoBERTa | Classification, retrieval |
| Decoder-only | Stacked decoder blocks (causal) | GPT, LLaMA | Generation, chat |
| Encoder-Decoder | Encoder + cross-attention decoder | T5, BART | Translation, summarization |

==Decoder-only dominates modern LLMs== because generation is the most general task.

## Positional Encoding

Transformers have no inherent notion of order. Add position information explicitly:

### Sinusoidal (Original)

<BlockMath latex="PE_{pos,2i} = \sin(pos / 10000^{2i/d}), \quad PE_{pos,2i+1} = \cos(pos / 10000^{2i/d})" />

```python
import numpy as np

def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len).reshape(-1, 1)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe
```

### Rotary Position Embedding (RoPE)

Modern default. Encodes relative position directly into attention scores:

```python
def apply_rope(x, freqs):
    """x: (N, H, T, d), freqs: (T, d//2)"""
    d = x.shape[-1]
    x_pairs = x[..., :d//2] + 1j * x[..., d//2:]
    rotated = x_pairs * np.exp(1j * freqs)
    return np.concatenate([rotated.real, rotated.imag], axis=-1)
```

<PyRunner
  cellId="048-cell-1"
  defaultCode={`import numpy as np

def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len).reshape(-1, 1)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe

pe = sinusoidal_pe(20, 16)
print("Sinusoidal Positional Encoding (first 5 positions):")
print(np.round(pe[:5, :8], 3))

# Show that nearby positions have similar encodings
from numpy.linalg import norm
dist_1_2 = norm(pe[1] - pe[2])
dist_1_10 = norm(pe[1] - pe[10])
print(f"\nDistance pos(1)-pos(2): {dist_1_2:.3f}")
print(f"Distance pos(1)-pos(10): {dist_1_10:.3f}")
print("✅ Nearby positions are closer in embedding space")
`}
/>

## Decoder-Only Transformer (GPT-style)

This is what you need to understand for LLMs:

```python
class GPTBlock:
    def __init__(self, d_model, num_heads, ff_dim):
        self.d_model = d_model
        self.num_heads = num_heads
        # Pre-norm architecture
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02
        
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)
        self.W1 = np.random.randn(d_model, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, d_model) * 0.02
        self.b2 = np.zeros(d_model)
```

> [!IMPORTANT] Key Design Choices in Modern LLMs
> - **Pre-Norm** over post-norm
> - **RoPE** over sinusoidal PE
> - **SwiGLU** activation in FFN instead of ReLU
> - **RMSNorm** instead of LayerNorm
> - **No bias** in linear layers
> These are engineering refinements; the core architecture is unchanged since 2017.

## Scaling Laws

Performance predictably improves with compute, data, and parameters:

<BlockMath latex="L(N) \propto N^{-\alpha}" />

More parameters + more data + more compute = lower loss. This predictable scaling is why companies bet billions on larger models.

<Quiz
  chapterSlug="048-transformer-architecture"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why do decoder-only models dominate modern LLMs over encoder-decoder?",
      options: ["They're smaller", "Generation is the most general task; classification and retrieval can be framed as generation, but not vice versa", "They train faster", "They don't need positional encoding"],
      correctIndex: 1,
      explanation: "A decoder-only model can generate text, answer questions, classify, summarize, and translate — all as text generation. Encoder-only models can't generate. Encoder-decoder adds complexity without proportional benefit for general-purpose use.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why do transformers need positional encoding?",
      options: ["To encode word meaning", "Self-attention is permutation-invariant; without PE, the model cannot distinguish token order", "To reduce computation", "For regularization"],
      correctIndex: 1,
      explanation: "Attention computes pairwise interactions regardless of position. Without explicit position signals, 'cat sat' and 'sat cat' would produce identical representations.",
      randomize: false,
    }
  ]}
/>
