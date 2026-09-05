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
      options: [
        "They're smaller and faster",
        "Generation is the most general task — classification, retrieval, summarization, and translation can all be framed as text generation, but not vice versa",
        "They train faster per epoch",
        "They don't need positional encoding"
      ],
      correctIndex: 1,
      explanation: "A decoder-only model handles every NLP task as conditional text generation. Encoder-only models can't generate. Encoder-decoder adds architectural complexity without proportional benefit for general-purpose use.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why do transformers need positional encoding?",
      options: [
        "To encode word meaning",
        "Self-attention is permutation-invariant — without PE, the model cannot distinguish token order ('cat sat' = 'sat cat')",
        "To reduce computation",
        "For regularization purposes"
      ],
      correctIndex: 1,
      explanation: "Attention computes pairwise interactions regardless of position. The same set of tokens in any order produces identical attention outputs. Positional encoding breaks this symmetry by injecting order information.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What are the three main Transformer variants?",
      options: [
        "Small, medium, large",
        "Encoder-only (BERT — classification/retrieval), Decoder-only (GPT — generation/chat), Encoder-Decoder (T5 — translation/summarization)",
        "CNN, RNN, Transformer",
        "Supervised, unsupervised, reinforcement"
      ],
      correctIndex: 1,
      explanation: "Each variant uses different masking strategies: encoder = bidirectional (no mask), decoder = causal (lower-triangular mask), encoder-decoder = bidirectional encoder + causal decoder with cross-attention bridge.",
      randomize: true
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What shape does sinusoidal_pe(100, 64) return?",
      code: "import numpy as np\nmax_len, d_model = 100, 64\npe = np.zeros((max_len, d_model))\nprint(pe.shape)",
      options: ["(100, 64)", "(64, 100)", "(100,)", "(64,)"],
      correctIndex: 0,
      explanation: "Positional encoding has shape (max_len, d_model). Each of the 100 positions gets a 64-dim vector that's added to the corresponding token embedding. The encoding is fixed (not learned) in the sinusoidal variant.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "How does RoPE differ from sinusoidal positional encoding?",
      options: [
        "RoPE is added to embeddings like sinusoidal PE",
        "RoPE encodes relative position directly into attention scores via rotation, rather than adding absolute position vectors to embeddings. This makes attention inherently position-aware.",
        "RoPE uses learned parameters instead of fixed functions",
        "There's no meaningful difference"
      ],
      correctIndex: 1,
      explanation: "Sinusoidal PE adds position vectors to token embeddings (absolute). RoPE rotates Q and K vectors based on position, so the dot product Q·K naturally depends on relative distance. This gives better length generalization.",
      randomize: true
    },
    {
      id: "q6",
      type: "fill-blank",
      prompt: "In sinusoidal PE, even dimensions use ___ and odd dimensions use ___, creating unique position signatures via different frequencies.",
      options: ["sin, cos", "cos, sin", "tan, cot", "exp, log"],
      correctIndex: 0,
      explanation: "PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)). Sin/cos pairs at different frequencies create unique encodings where nearby positions are similar and distant positions differ.",
      randomize: false
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What modern refinements distinguish current LLMs from the original 2017 Transformer?",
      options: [
        "The core architecture is completely different",
        "Pre-Norm, RoPE, SwiGLU activation, RMSNorm, no bias in linear layers — engineering improvements that stabilize training and improve performance without changing the fundamental block structure",
        "They use RNNs internally",
        "They removed attention entirely"
      ],
      correctIndex: 1,
      explanation: "The attention+FFN+residual+norm template is unchanged. Refinements like RMSNorm (faster than LayerNorm), SwiGLU (better than ReLU), and RoPE (better than sinusoidal) are incremental improvements on the same foundation.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does the scaling law L(N) ∝ N^(-α) tell us?",
      options: [
        "Loss increases with more parameters",
        "Loss predictably decreases as a power law of parameters/data/compute. This predictable relationship enables reliable forecasting of model performance before training.",
        "Loss is independent of model size",
        "Scaling only works up to 1B parameters"
      ],
      correctIndex: 1,
      explanation: "Kaplan et al. (2020) showed loss follows a smooth power law across orders of magnitude. This predictability is why companies invest billions — they can estimate returns before spending compute.",
      randomize: true
    },
    {
      id: "q9",
      type: "shape-prediction",
      prompt: "If token embeddings have shape (N, T, D=768) and positional encoding has shape (T, D), what shape results from adding them?",
      options: ["(N, T, 768)", "(T, 768)", "(N, 768)", "Error — shapes incompatible"],
      correctIndex: 0,
      explanation: "PE (T, D) broadcasts against embeddings (N, T, D) along the batch dimension. Each sample gets the same positional signal added to its token embeddings. Output retains (N, T, D) shape.",
      randomize: true
    },
    {
      id: "q10",
      type: "ordering",
      prompt: "Order the components in a decoder-only Transformer forward pass:",
      items: ["Token embedding lookup", "Add positional encoding", "Stack of transformer blocks (causal self-attention + FFN)", "Final layer norm", "Output projection to vocabulary logits"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Embed tokens → inject position → process through N blocks → normalize → project to vocab. Each block refines representations using causal attention (past-only) and position-wise FFN.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why does BERT use encoder-only architecture instead of decoder-only?",
      options: [
        "BERT was designed before decoders existed",
        "BERT's tasks (classification, NER, retrieval) need bidirectional context — seeing both left AND right of each token. Causal masking in decoder-only would prevent this.",
        "Encoder-only is always better",
        "BERT generates text"
      ],
      correctIndex: 1,
      explanation: "Understanding tasks benefit from full context. 'The ___ sat on the mat' needs both sides to fill 'cat'. Encoder-only's bidirectional attention provides this. Generation tasks need causal masking, making decoder-only appropriate.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What is RMSNorm and how does it differ from LayerNorm?",
      options: [
        "They're identical",
        "RMSNorm skips mean-centering and only divides by root-mean-square. Fewer operations, similar effectiveness. Modern LLMs prefer it for computational efficiency.",
        "RMSNorm normalizes across the batch",
        "RMSNorm uses exponential moving averages"
      ],
      correctIndex: 1,
      explanation: "LayerNorm: subtract mean, divide by std. RMSNorm: just divide by RMS (sqrt of mean of squares). Removing mean-centering saves computation with negligible quality impact. Most modern LLMs (LLaMA, etc.) use RMSNorm.",
      randomize: true
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "In sinusoidal PE with d_model=8, what frequency divisor applies to dimension pair (0,1)?",
      code: "import numpy as np\nd_model = 8\ni = 0  # first pair\ndivisor = 10000 ** (2*i / d_model)\nprint(f'{divisor:.1f}')",
      options: ["1.0", "10000.0", "100.0", "0.0001"],
      correctIndex: 0,
      explanation: "For i=0: 10000^(0/8) = 10000^0 = 1.0. The first dimension pair uses frequency 1/1 = 1.0 (highest frequency). Higher i values use larger divisors (lower frequencies), creating a multi-scale position representation.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why do modern LLMs remove bias terms from linear layers?",
      options: [
        "Biases cause gradient explosion",
        "Biases add parameters without proportional benefit when LayerNorm/RMSNorm already provides learnable shift parameters. Removing them simplifies the model and reduces parameter count.",
        "GPUs can't compute biases efficiently",
        "Biases are incompatible with attention"
      ],
      correctIndex: 1,
      explanation: "LayerNorm's beta parameter already serves as a per-layer bias. Adding separate linear biases is redundant. Removing them saves ~1% of parameters and simplifies implementation with no measurable quality loss.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "SwiGLU replaces ReLU in the FFN. It computes SwiGLU(x) = Swish(xW₁) ⊙ (___), providing smoother gradients than ReLU.",
      options: ["xW₂", "xW₃", "sigmoid(x)", "tanh(x)"],
      correctIndex: 1,
      explanation: "SwiGLU(x, W₁, W₃) = Swish(xW₁) ⊙ (xW₃). The gating mechanism (element-wise product of two projections) provides smoother, more expressive non-linearity than ReLU. Most modern LLMs use SwiGLU.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "An encoder-decoder Transformer uses cross-attention in the decoder. What does this attend to?",
      options: [
        "Previous decoder tokens",
        "All encoder output states — allowing each decoder position to read relevant parts of the encoded input",
        "Random positions",
        "Only the last encoder state"
      ],
      correctIndex: 1,
      explanation: "Cross-attention: Q comes from decoder, K and V come from encoder. This is how the decoder accesses encoded input information. Self-attention handles intra-decoder dependencies; cross-attention bridges encoder and decoder.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why does sinusoidal PE use decreasing frequencies across dimensions?",
      options: [
        "It's arbitrary",
        "Low dimensions capture fine-grained local position differences (high frequency), while high dimensions capture coarse global position (low frequency). This creates a multi-scale position representation.",
        "To prevent overflow",
        "Higher frequencies are computationally expensive"
      ],
      correctIndex: 1,
      explanation: "Dimension 0-1 oscillate rapidly (distinguish adjacent positions). Higher dimensions oscillate slowly (distinguish distant positions). Together, they create unique encodings where similarity decays smoothly with distance.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "What advantage does RoPE provide over sinusoidal PE for long-context models?",
      options: [
        "RoPE is faster to compute",
        "RoPE encodes relative position, enabling better extrapolation to sequence lengths not seen during training. Sinusoidal PE encodes absolute position and degrades beyond trained length.",
        "RoPE uses fewer parameters",
        "There's no advantage"
      ],
      correctIndex: 1,
      explanation: "Since RoPE affects attention scores via relative rotation, the model learns position-independent patterns. At inference, longer sequences use the same relative encoding. Sinusoidal PE assigns specific vectors to specific positions, failing beyond max trained length.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "You want to build a document classifier. Which Transformer variant is most appropriate?",
      options: [
        "Decoder-only",
        "Encoder-only — bidirectional context captures full document semantics, and the [CLS] token or pooled output provides a fixed-size representation for classification",
        "Encoder-decoder",
        "None — use an RNN"
      ],
      correctIndex: 1,
      explanation: "Classification needs understanding, not generation. Encoder-only's bidirectional attention captures complete context. The final hidden state of a special token serves as the document representation for a classification head.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "If scaling laws predict that doubling compute reduces loss by 5%, what happens when you double again?",
      options: [
        "Loss reduces by another 5% (linear scaling)",
        "Loss reduces by approximately 5% again (power law scaling continues smoothly across orders of magnitude, though with diminishing absolute returns)",
        "No further improvement",
        "Loss increases due to overfitting"
      ],
      correctIndex: 1,
      explanation: "Power law L ∝ C^(-α) means constant relative improvement per multiplicative increase. Each doubling yields similar percentage reduction. However, absolute improvement shrinks (5% of 2.0 > 5% of 1.9), motivating ever-larger investments.",
      randomize: true
    }
  ]}
/>
