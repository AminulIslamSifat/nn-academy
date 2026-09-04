---
title: "Capstone: Training a Mini-LLM"
slug: "054-mini-llm-capstone"
description: "Put everything together: tokenize data, build a decoder-only transformer, pretrain on text, and generate. A complete LLM pipeline in NumPy."
track: "nn-advanced"
order: 9
read_time: 35
code_time: 30
execution_timeout: 60
prerequisites: ["049-tokenization", "050-gpt-from-scratch", "051-pretraining-finetuning"]
---

# Capstone: Training a Mini-LLM

==This is the final project.== You'll train a tiny language model from scratch using everything from chapters 046-053. It won't write essays, but it will generate coherent text and demonstrate every concept you've learned.

## Project Structure

```
mini_llm/
├── tokenizer.py      # BPE tokenizer
├── model.py          # Decoder-only transformer
├── train.py          # Training loop
├── generate.py       # Autoregressive generation
└── data/
    └── corpus.txt    # Training text
```

## Step 1: Prepare Data

```python
import numpy as np
from collections import Counter

# Load and tokenize
corpus = open("data/corpus.txt").read()

# Train BPE (from chapter 049)
def train_bpe(text, num_merges=500):
    tokens = list(text)
    merges = []
    for _ in range(num_merges):
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs: break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        merged = "".join(best)
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                new_tokens.append(merged); i += 2
            else:
                new_tokens.append(tokens[i]); i += 1
        tokens = new_tokens
    vocab = sorted(set(tokens))
    tok2id = {t: i for i, t in enumerate(vocab)}
    return tok2id, merges
```

## Step 2: Build the Model

Use the MiniGPT class from chapter 050 with these recommended sizes:

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| d_model | 64 | Trainable on CPU |
| num_heads | 4 | d_k = 16 |
| num_layers | 2 | Fast iteration |
| ff_dim | 256 | 4× d_model |
| max_len | 128 | Short contexts |
| vocab_size | ~2000 | From BPE |

## Step 3: Training Loop

```python
def train(model, token_ids, epochs=20, batch_size=32, lr=0.001):
    N = len(token_ids)
    seq_len = 128
    
    for epoch in range(epochs):
        # Create random subsequences
        indices = np.random.randint(0, N - seq_len, batch_size)
        batch = np.array([token_ids[i:i+seq_len] for i in indices])
        
        # Forward
        logits = model.forward(batch[:, :-1])
        targets = batch[:, 1:]
        
        # Loss
        log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        loss = -np.mean(log_probs[:, np.arange(seq_len-1), targets])
        
        # Backward + update (simplified)
        # In practice, implement full backprop or use autograd
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: loss={loss:.3f}")
```

<PyRunner
  cellId="054-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

# Simulate training progress
text = "the cat sat on the mat. the dog ran in the park. " * 200
vocab = sorted(set(text))
vocab_size = len(vocab)

print(f"Mini-LLM Training Setup:")
print(f"  Corpus length: {len(text)} chars")
print(f"  Vocabulary: {vocab_size} unique characters")
print(f"  Model: 2-layer transformer, d=64, h=4")
print(f"  Context: 128 tokens")
print(f"  Parameters: ~{64*64*8*2 + vocab_size*64 + 64*vocab_size:,}")

# Simulated training curve
losses = 3.5 * np.exp(-np.arange(20) * 0.12) + 1.2
print(f"\nSimulated Training:")
for e in range(0, 20, 5):
    print(f"  Epoch {e:2d}: loss={losses[e]:.3f}")

print(f"\n💡 With real backprop, this model would learn basic English patterns")
print(f"   Expected output: grammatical but nonsensical short sentences")
`}
/>

## Step 4: Generation

```python
def generate(model, prompt, max_tokens=50, temperature=0.8):
    ids = [tok2id.get(c, 0) for c in prompt]
    for _ in range(max_tokens):
        x = np.array([ids[-128:]])
        logits = model.forward(x)[0, -1] / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        next_id = np.random.choice(len(probs), p=probs)
        ids.append(next_id)
    return "".join(id2tok[i] for i in ids)
```

## What to Expect

A mini-LLM trained on a few KB of text for 20 epochs will:
- ✅ Generate grammatically plausible text
- ✅ Repeat common phrases from training data
- ✅ Show basic word-level coherence
- ❌ NOT produce factual content
- ❌ NOT follow instructions
- ❌ NOT reason

==That's expected.== Scale is what unlocks those abilities. Your mini-LLM demonstrates the ARCHITECTURE; real LLMs demonstrate the SCALE.

> [!IMPORTANT] Learning Outcomes
> After completing this capstone, you understand:
> - Tokenization (BPE)
> - Transformer architecture (attention, FFN, residuals, norms)
> - Positional encoding
> - Autoregressive generation
> - Next-token prediction training
> - Temperature sampling
> - Why scale matters
>
> You've built an LLM from scratch in NumPy. That puts you ahead of 99% of ML practitioners.

<Quiz
  chapterSlug="054-mini-llm-capstone"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does a mini-LLM generate grammatical but nonsensical text?",
      options: ["The architecture is wrong", "It learned local patterns (grammar, common phrases) from limited data but lacks the scale for global coherence, facts, and reasoning", "Temperature is too high", "Need more layers"],
      correctIndex: 1,
      explanation: "Grammar is a local pattern learnable from small data. Factual knowledge and long-range reasoning require seeing billions of tokens. Architecture is correct; scale is missing.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What's the most important lesson from building a mini-LLM?",
      options: ["NumPy is fast enough for production", "The same architecture that generates nonsense at 100K params generates Shakespeare at 100B params — scale transforms capability", "Transformers are too complex", "Language models need PyTorch"],
      correctIndex: 1,
      explanation: "Architecture alone doesn't create intelligence. The identical transformer block, scaled up with sufficient data and compute, produces qualitatively different behavior. Understanding both the architecture AND the scaling is essential.",
      randomize: false,
    }
  ]}
/>
