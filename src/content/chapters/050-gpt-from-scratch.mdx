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
      options: ["It's faster", "Each position predicts the NEXT token; only the last position's prediction gives us the next token to append", "Other positions are masked", "Only the last position has gradients"],
      correctIndex: 1,
      explanation: "Position t predicts token t+1. During generation, we've already generated tokens 0..T-1 and want token T. Only position T-1's output predicts token T.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does temperature control in generation?",
      options: ["Model size", "Sharpness of the probability distribution: low=greedy/deterministic, high=diverse/random", "Training speed", "Memory usage"],
      correctIndex: 1,
      explanation: "Dividing logits by T before softmax: T less than 1 sharpens (more confident), T>1 flattens (more uniform). T→0 approaches argmax, T→∞ approaches uniform random.",
      randomize: false,
    }
  ]}
/>
