---
title: "Building GPT from Scratch"
slug: "050-gpt-from-scratch"
description: "Implement a complete decoder-only transformer language model in NumPy: embeddings, causal attention, positional encoding, and autoregressive generation."
track: "nn-advanced"
order: 5
read_time: 35
code_time: 30
execution_timeout: 60
prerequisites: ["047-multihead-attention", "048-transformer-architecture", "049-tokenization"]
---

# Building GPT from Scratch

==This is where everything comes together.== You'll build a mini-GPT: a decoder-only transformer that generates text autoregressively, entirely in NumPy.

## Architecture Overview

```
Token IDs → Embedding + Positional Encoding
         → N × Transformer Block (causal self-attention + FFN)
         → LayerNorm → Linear(vocab_size) → Logits
         → Sample next token → repeat
```

## Complete Implementation

```python
import numpy as np

class MiniGPT:
    def __init__(self, vocab_size, d_model=64, num_heads=4, 
                 num_layers=2, max_len=128, ff_dim=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        
        # Token + position embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_len, d_model) * 0.02
        
        # Transformer blocks
        self.blocks = []
        for _ in range(num_layers):
            block = {
                "ln1_g": np.ones(d_model), "ln1_b": np.zeros(d_model),
                "Wq": np.random.randn(d_model, d_model) * 0.02,
                "Wk": np.random.randn(d_model, d_model) * 0.02,
                "Wv": np.random.randn(d_model, d_model) * 0.02,
                "Wo": np.random.randn(d_model, d_model) * 0.02,
                "ln2_g": np.ones(d_model), "ln2_b": np.zeros(d_model),
                "W1": np.random.randn(d_model, ff_dim) * 0.02,
                "b1": np.zeros(ff_dim),
                "W2": np.random.randn(ff_dim, d_model) * 0.02,
                "b2": np.zeros(d_model),
            }
            self.blocks.append(block)
        
        # Output projection
        self.ln_f_g = np.ones(d_model)
        self.ln_f_b = np.zeros(d_model)
        self.head = np.random.randn(d_model, vocab_size) * 0.02
```

### Forward Pass

```python
    def forward(self, token_ids):
        """token_ids: (N, T) → logits (N, T, vocab_size)"""
        N, T = token_ids.shape
        
        # Embeddings
        x = self.token_emb[token_ids] + self.pos_emb[:T]
        
        # Causal mask
        mask = np.tril(np.ones((T, T), dtype=bool))
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention (pre-norm)
            h = layer_norm(x, block["ln1_g"], block["ln1_b"])
            attn_out, _ = multi_head_attention(
                h, block["Wq"], block["Wk"], block["Wv"], block["Wo"],
                self.num_heads, mask
            )
            x = x + attn_out
            
            # Feed-forward (pre-norm)
            h = layer_norm(x, block["ln2_g"], block["ln2_b"])
            ff_out = np.maximum(0, h @ block["W1"] + block["b1"]) @ block["W2"] + block["b2"]
            x = x + ff_out
        
        # Output
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.head
        return logits
```

<PyRunner
  cellId="050-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

# Mini demo: show the architecture flow
vocab_size, d_model, T = 100, 32, 8
token_ids = np.array([[5, 12, 3, 42, 7, 1, 99, 28]])

token_emb = np.random.randn(vocab_size, d_model) * 0.02
pos_emb = np.random.randn(T, d_model) * 0.02

x = token_emb[token_ids[0]] + pos_emb
print("MiniGPT Forward Pass:")
print(f"  Token IDs:     {token_ids[0]}")
print(f"  Token emb:     {token_emb[token_ids[0]].shape}")
print(f"  + Pos emb:     {pos_emb.shape}")
print(f"  Input to blocks: {x.shape}")
print(f"  After N blocks:  {x.shape} (same!)")
print(f"  Final LN + head: ({T}, {vocab_size}) logits")

mask = np.tril(np.ones((T, T), dtype=int))
print(f"\n  Causal mask shape: {mask.shape}")
print(f"  Each position attends only to itself and earlier positions")

params = vocab_size*d_model + T*d_model + 2*(d_model*d_model*4 + d_model*256 + 256*d_model) + d_model*vocab_size
print(f"\n📊 Estimated params (2 layers, d=32): ~{params:,}")
`}
/>

## Autoregressive Generation

Generate one token at a time, feeding each prediction back as input:

```python
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0):
        """Autoregressive generation."""
        ids = list(prompt_ids)
        
        for _ in range(max_new_tokens):
            # Forward pass on current sequence
            x = np.array([ids[-self.max_len:]])  # truncate if too long
            logits = self.forward(x)
            
            # Get logits for last position
            next_logits = logits[0, -1] / temperature
            
            # Softmax sampling
            probs = np.exp(next_logits - np.max(next_logits))
            probs /= probs.sum()
            next_token = np.random.choice(len(probs), p=probs)
            
            ids.append(next_token)
        
        return ids
```

<PyRunner
  cellId="050-cell-2"
  defaultCode={`import numpy as np
np.random.seed(42)

# Demonstrate temperature effect on sampling
logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])

print("Effect of temperature on sampling distribution:")
print(f"{'Temp':>6} | {'Top prob':>8} | {'Entropy':>8} | Distribution")
print("─" * 55)

for temp in [0.1, 0.5, 1.0, 2.0]:
    scaled = logits / temp
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    bar = "".join(f"{'█' * int(p*20)}" for p in probs)
    print(f"{temp:6.1f} | {probs.max():8.3f} | {entropy:8.3f} | {bar}")

print(f"\nLow temp → deterministic (greedy-like)")
print(f"High temp → diverse (more random)")
print(f"temp=1.0 → original distribution")
`}
/>

## Training Loop

Standard next-token prediction with cross-entropy:

```python
def train_step(model, token_ids, lr=0.001):
    """One training step: predict next token given previous tokens."""
    N, T = token_ids.shape
    
    # Forward
    logits = model.forward(token_ids[:, :-1])  # predict from position 0..T-2
    targets = token_ids[:, 1:]                  # targets are positions 1..T-1
    
    # Cross-entropy loss
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(N).reshape(-1,1), 
                               np.arange(T-1).reshape(1,-1), targets])
    
    # Backward pass would go here (omitted for brevity)
    # In practice, use autograd or manual backprop through transformer
    
    return loss
```

> [!IMPORTANT] What Makes This Hard
> The forward pass above works. The backward pass through a multi-layer transformer with attention is complex. In production, use PyTorch/JAX autograd. Building it manually is an excellent learning exercise but expect weeks of debugging.

## From Mini-GPT to Real LLMs

| Component | Mini-GPT | GPT-3/4 |
|-----------|---------|---------|
| Layers | 2-4 | 96+ |
| d_model | 64-128 | 12288 |
| Heads | 4-8 | 96 |
| Context | 128-512 | 4096-128K |
| Vocab | 1K-10K | 50K+ |
| Parameters | under 1M | 175B+ |
| Training data | KB | TB |

The architecture is identical. Scale is the difference.

<Quiz
  chapterSlug="050-gpt-from-scratch"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "In autoregressive generation, why do we only use the last position's logits?",
      options: [
        "It's computationally faster",
        "Each position predicts the NEXT token; only the last position's prediction gives us the next token to append to the sequence",
        "Other positions are masked out",
        "Only the last position receives gradients"
      ],
      correctIndex: 1,
      explanation: "Position t predicts token t+1. During generation, we've generated tokens 0..T-1 and want token T. Only position T-1's output predicts token T. Earlier positions' predictions were already consumed.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does temperature control in text generation?",
      options: [
        "Model size and parameter count",
        "Sharpness of the probability distribution: low temp → greedy/deterministic, high temp → diverse/random",
        "Training convergence speed",
        "GPU memory usage"
      ],
      correctIndex: 1,
      explanation: "Dividing logits by T before softmax: T<1 sharpens (more confident, repetitive), T>1 flattens (more uniform, creative). T→0 approaches argmax, T→∞ approaches uniform random sampling.",
      randomize: false
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "If token_ids has shape (4, 64) and d_model=128, what shape is x after embedding + positional encoding?",
      options: ["(4, 64, 128)", "(4, 128)", "(64, 128)", "(4, 64)"],
      correctIndex: 0,
      explanation: "token_emb[token_ids]: (4, 64, 128). pos_emb[:64]: (64, 128) broadcasts to (4, 64, 128). Sum preserves shape. Each of 4 samples has 64 positions with 128-dim representations.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why does GPT use learned positional embeddings instead of sinusoidal PE?",
      options: [
        "Sinusoidal PE doesn't work with causal attention",
        "Learned embeddings can adapt to the specific positional patterns needed by the model. Modern practice shows both work similarly, but learned PE is simpler to implement.",
        "Learned PE uses fewer parameters",
        "There's no difference in practice"
      ],
      correctIndex: 1,
      explanation: "Both approaches work. Learned PE lets the model discover optimal position representations during training. Sinusoidal PE provides fixed multi-scale encoding. Modern LLMs often use RoPE instead of either.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "With temperature=0.5, what happens to these logits before softmax?",
      code: "import numpy as np\nlogits = np.array([2.0, 1.0, 0.5])\nscaled = logits / 0.5\nprint(scaled)",
      options: ["[4. 2. 1.]", "[1. 0.5 0.25]", "[2. 1. 0.5]", "[0.4 0.2 0.1]"],
      correctIndex: 0,
      explanation: "Dividing by 0.5 doubles the logits. Larger differences between logits → sharper softmax distribution → more deterministic sampling. Low temperature amplifies the model's confidence.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "In the training loop, why are targets shifted by one position (targets = token_ids[:, 1:])?",
      options: [
        "To reduce sequence length",
        "Next-token prediction: position t's output should predict token t+1. Inputs are tokens 0..T-2, targets are tokens 1..T-1. The model learns P(x_{t+1} | x_0..x_t).",
        "To prevent data leakage",
        "It's arbitrary — any alignment works"
      ],
      correctIndex: 1,
      explanation: "Language modeling objective: predict the next token given all previous tokens. Shifting creates input-target pairs where each position's prediction is evaluated against the actual next token.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "During generation, the model feeds its own ___ back as input for the next step. This is called ___ generation.",
      options: ["prediction, autoregressive", "target, supervised", "embedding, parallel", "gradient, recurrent"],
      correctIndex: 0,
      explanation: "Autoregressive = self-feeding. Each generated token becomes part of the input context for predicting the next. This sequential dependency is why generation can't be parallelized (unlike training with teacher forcing).",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What happens if you set temperature=0 during generation?",
      options: [
        "The model generates random tokens",
        "Division by zero error. In practice, T→0 makes softmax approach argmax — always picking the highest-probability token, producing deterministic but potentially repetitive output.",
        "Generation stops immediately",
        "Temperature=0 is the recommended default"
      ],
      correctIndex: 1,
      explanation: "As T→0, logits/T → ±∞, making softmax concentrate all probability on the maximum logit. Equivalent to greedy decoding. Safe implementations clamp T to a small positive value.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why does MiniGPT truncate the input to max_len during generation?",
      options: [
        "To save memory",
        "The positional embedding matrix has fixed size max_len. Longer sequences would index out of bounds. Also, causal attention cost grows quadratically.",
        "Truncation improves generation quality",
        "GPUs require fixed-length inputs"
      ],
      correctIndex: 1,
      explanation: "pos_emb has shape (max_len, d_model). Indexing beyond max_len fails. Additionally, attention is O(T²), so unbounded growth quickly becomes intractable even with sufficient memory.",
      randomize: true
    },
    {
      id: "q10",
      type: "ordering",
      prompt: "Order one step of autoregressive generation:",
      items: ["Forward pass on current token sequence", "Extract logits at last position", "Apply temperature scaling", "Sample next token from softmax distribution", "Append sampled token to sequence"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Process full context → get next-token prediction → adjust confidence via temperature → sample probabilistically → extend sequence. Repeat until EOS or max length.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "How does MiniGPT differ architecturally from GPT-3/4?",
      options: [
        "Completely different architecture",
        "Identical architecture at different scale: same decoder-only transformer blocks, same causal attention, same autoregressive generation. Difference is layers (2 vs 96+), dimensions (64 vs 12288), and data (KB vs TB).",
        "MiniGPT uses RNNs internally",
        "GPT-3 uses encoder-decoder architecture"
      ],
      correctIndex: 1,
      explanation: "This is the key insight: architecture is identical, scale differs. Understanding MiniGPT means understanding GPT-4. The engineering challenges are about scaling, not architectural innovation.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why is the backward pass through a transformer 'hard' compared to the forward pass?",
      options: [
        "Backprop doesn't work with attention",
        "Gradients must flow through multiple interacting components: attention scores depend on Q and K, which depend on inputs. Chain rule through softmax, matmuls, residuals, and LayerNorm across N layers requires careful bookkeeping.",
        "NumPy doesn't support backpropagation",
        "The backward pass is actually simpler than forward"
      ],
      correctIndex: 1,
      explanation: "Each transformer block has ~10 operations with interdependent gradients. Attention alone requires derivatives through softmax(QK^T/√d)V w.r.t. Q, K, V. Multiplied across N layers with residual connections, manual backprop is extremely error-prone.",
      randomize: true
    },
    {
      id: "q13",
      type: "code-output",
      prompt: "What shape are the output logits for batch size 2, sequence length 16, vocab size 5000?",
      code: "N, T, V = 2, 16, 5000\nlogits_shape = (N, T, V)\nprint(logits_shape)",
      options: ["(2, 16, 5000)", "(16, 5000)", "(2, 5000)", "(2, 16)"],
      correctIndex: 0,
      explanation: "Logits: (batch, sequence_length, vocab_size). Each position in each sample gets a score for every vocabulary token. Cross-entropy loss compares these against the target token ID at each position.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why does the final layer norm (ln_f) exist before the output projection?",
      options: [
        "It's optional and can be removed",
        "After N transformer blocks, activations may have drifted from stable distributions. Final LayerNorm ensures the output head receives well-conditioned inputs for reliable logits.",
        "It replaces softmax",
        "It encodes positional information"
      ],
      correctIndex: 1,
      explanation: "Pre-norm architectures accumulate un-normalized residual additions. The final LN stabilizes the representation before the linear projection to vocabulary space, preventing extreme logits.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "The output head projects from d_model dimensions to ___ dimensions, producing one logit per vocabulary token.",
      options: ["vocab_size", "d_model", "num_heads", "ff_dim"],
      correctIndex: 0,
      explanation: "head: (d_model, vocab_size). Each position's d_model-dim representation maps to vocab_size scores. Softmax over these scores gives the next-token probability distribution.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Your generated text repeats the same phrase endlessly. What should you adjust?",
      options: [
        "Increase model size",
        "Increase temperature to add diversity, or implement repetition penalty to discourage recently generated tokens",
        "Decrease temperature",
        "Remove the causal mask"
      ],
      correctIndex: 1,
      explanation: "Low temperature + no repetition penalty → model confidently repeats high-probability patterns. Higher temperature adds exploration. Repetition penalty explicitly reduces logits for recently used tokens.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why is cross-entropy the standard loss for language modeling?",
      options: [
        "It's the only loss that works with softmax",
        "Cross-entropy directly measures how well the predicted distribution matches the true next-token distribution. Minimizing CE = maximizing likelihood of the training data.",
        "MSE works better but is slower",
        "It prevents overfitting automatically"
      ],
      correctIndex: 1,
      explanation: "Language modeling = estimating P(x_{t+1} | x_1..x_t). Cross-entropy loss = -log P(true_token), which is exactly the negative log-likelihood. Minimizing it maximizes the probability assigned to correct tokens.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "During generation, why recompute the full forward pass for each new token instead of just the last step?",
      options: [
        "It's wasteful but necessary without KV caching. With KV cache, you'd only compute the new token's attention against cached past keys/values.",
        "Each token needs different weights",
        "The causal mask changes each step",
        "NumPy requires full recomputation"
      ],
      correctIndex: 0,
      explanation: "Without KV caching, every generation step recomputes all previous attention. Real implementations cache K,V from previous steps, reducing generation from O(T²) to O(T) per new token. Critical for practical inference speed.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What determines the maximum context length of MiniGPT?",
      options: [
        "Available RAM only",
        "The positional embedding matrix size (max_len) and quadratic attention cost. Both impose hard limits on sequence length.",
        "The vocabulary size",
        "Number of transformer layers"
      ],
      correctIndex: 1,
      explanation: "pos_emb has shape (max_len, d_model) — indexing beyond max_len fails. Even with RoPE (no fixed PE), attention cost O(T²) makes very long sequences computationally prohibitive without efficient attention variants.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You trained MiniGPT on Shakespeare and it generates plausible-sounding but nonsensical text. Why?",
      options: [
        "The model implementation is broken",
        "Small models (under 1M params, KB of data) learn surface-level patterns (rhythm, common phrases) but lack capacity for coherent long-range reasoning. Scale brings coherence.",
        "Shakespeare is too complex for transformers",
        "The tokenizer is wrong"
      ],
      correctIndex: 1,
      explanation: "Language modeling at small scale captures statistical regularities without true understanding. Coherent generation emerges at scale when the model has enough capacity and data to learn world knowledge and reasoning patterns.",
      randomize: true
    }
  ]}
/>
