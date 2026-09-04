---
title: "Word Embeddings"
slug: "043-word-embeddings"
description: "From one-hot vectors to dense semantic representations. Implement embedding layers, Word2Vec skip-gram, and understand why embeddings capture meaning in NumPy."
track: "nn-intermediate"
order: 8
read_time: 22
code_time: 18
execution_timeout: 15
prerequisites: ["041-rnn-from-scratch"]
---

# Word Embeddings

One-hot encoding treats every word as completely unrelated. ==Embeddings map words to dense vectors where similar words have similar vectors.== This is the foundation of all modern NLP.

## Why Not One-Hot?

With vocabulary size V=50,000:
- One-hot vector: 50,000 dimensions, 49,999 zeros
- "cat" and "dog" are as distant as "cat" and "the"
- No notion of similarity or semantics
- Massive sparse matrices everywhere

Embeddings solve all of this: each word gets a learned D-dimensional vector (typically 50–300) where geometry encodes meaning.

## Embedding Layer

Just a lookup table:

```python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
    
    def forward(self, indices):
        """indices: (N, T) integer token IDs → (N, T, D) vectors"""
        return self.W[indices]
    
    def backward(self, dout, indices):
        """dout: (N, T, D). Accumulate gradients into embedding matrix."""
        dW = np.zeros_like(self.W)
        np.add.at(dW, indices, dout)
        return dW
```

<PyRunner
  cellId="043-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

vocab_size, embed_dim = 100, 8
W = np.random.randn(vocab_size, embed_dim) * 0.1

# Lookup
indices = np.array([[5, 12, 3, 42]])  # batch of 1, sequence of 4
embedded = W[indices]  # (1, 4, 8)

print(f"Vocabulary: {vocab_size} words")
print(f"Embedding dim: {embed_dim}")
print(f"\nToken IDs: {indices[0]}")
print(f"Embedded shape: {embedded.shape}")
print(f"\nEach token → {embed_dim}-dim dense vector:")
for i, idx in enumerate(indices[0]):
    print(f"  Token {idx}: [{embedded[0,i,:4].round(3)}...]")

print(f"\n💡 One-hot would be {vocab_size}-dim sparse")
print(f"   Embedding is {embed_dim}-dim dense ({vocab_size/embed_dim:.0f}× compression)")
`}
/>

> [!NOTE] `np.add.at` for Backward
> Standard indexing (`dW[indices] += dout`) doesn't work when indices repeat — it only adds once. `np.add.at` correctly accumulates gradients for duplicate token IDs.

## Word2Vec: Learning Embeddings from Co-occurrence

The key insight: ==words that appear in similar contexts have similar meanings.== Word2Vec learns embeddings by predicting context words (skip-gram) or target words (CBOW).

### Skip-Gram Model

Given a center word, predict surrounding context words:

<BlockMath latex="P(w_{context} | w_{center}) = \frac{\exp(v_{center}^T u_{context})}{\sum_w \exp(v_{center}^T u_w)}" />

Training on "the cat sat on the mat" with window=2:
- (cat, the), (cat, sat), (cat, on)
- (sat, cat), (sat, on), (sat, the)
- etc.

<PyRunner
  cellId="043-cell-2"
  defaultCode={`import numpy as np
np.random.seed(42)

# Mini Word2Vec demo
text = "the cat sat on the mat the cat likes the mat"
words = text.split()
vocab = sorted(set(words))
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
D = 6  # tiny embedding dim for demo

print(f"Vocabulary: {vocab}")
print(f"Size: {V}, Embedding dim: {D}")

# Generate skip-gram pairs
window = 2
pairs = []
for i, word in enumerate(words):
    for j in range(max(0, i-window), min(len(words), i+window+1)):
        if i != j:
            pairs.append((word_to_idx[word], word_to_idx[words[j]]))

print(f"\nSkip-gram pairs (first 10):")
for c, ctx in pairs[:10]:
    print(f"  {vocab[c]} → {vocab[ctx]}")
print(f"Total pairs: {len(pairs)}")

# Train embeddings
W_center = np.random.randn(V, D) * 0.1
W_context = np.random.randn(V, D) * 0.1
lr = 0.5

for epoch in range(100):
    loss = 0
    for center, context in pairs:
        # Dot product similarity
        score = W_center[center] @ W_context[context]
        # Simplified: push similar pairs closer
        pred = 1 / (1 + np.exp(-score))
        loss -= np.log(pred + 1e-8)
        
        grad = (pred - 1)
        W_center[center] -= lr * grad * W_context[context] / len(pairs)
        W_context[context] -= lr * grad * W_center[center] / len(pairs)
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch:3d}: loss={loss/len(pairs):.3f}")

# Check similarities
def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print(f"\nCosine similarities:")
for w1 in ['cat', 'mat', 'the']:
    sims = [(w2, cosine_sim(W_center[word_to_idx[w1]], W_center[word_to_idx[w2]])) 
            for w2 in vocab if w2 != w1]
    sims.sort(key=lambda x: -x[1])
    top3 = ", ".join(f"{w}({s:.2f})" for w, s in sims[:3])
    print(f"  {w1} ≈ {top3}")
`}
/>

## Embedding Arithmetic

The famous property: ==vector arithmetic captures semantic relationships==.

<BlockMath latex="\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}" />

This emerges naturally from training — the model learns that gender and royalty are independent axes in embedding space.

> [!IMPORTANT] Practical Notes
> - Pre-trained embeddings (GloVe, Word2Vec, FastText) are available for free — don't train from scratch unless your domain is specialized
> - Modern models (BERT, GPT) use contextual embeddings where the same word gets different vectors depending on context
> - Embedding dimension: 50-300 for Word2Vec, 768+ for transformers

<Quiz
  chapterSlug="043-word-embeddings"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why do embeddings capture semantic similarity?",
      options: [
        "They're manually designed to encode meaning",
        "Words appearing in similar contexts get pushed to similar vectors during training, encoding distributional semantics",
        "They use alphabetical ordering",
        "Neural networks always learn semantics"
      ],
      correctIndex: 1,
      explanation: "The distributional hypothesis: words in similar contexts have similar meanings. Training objectives (predict context, predict target) force co-occurring words to have high dot products, creating geometric structure in embedding space.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why use np.add.at instead of dW[indices] += dout for embedding backward?",
      options: [
        "It's faster",
        "When the same token appears multiple times in a batch, standard indexing only adds the gradient once; np.add.at correctly accumulates all occurrences",
        "It handles negative indices",
        "There's no difference"
      ],
      correctIndex: 1,
      explanation: "NumPy fancy indexing with duplicates is undefined behavior for +=. If token 5 appears 3 times, dW[5] += dout only adds one gradient. np.add.at(dW, indices, dout) correctly sums all three.",
      randomize: false,
    }
  ]}
/>
