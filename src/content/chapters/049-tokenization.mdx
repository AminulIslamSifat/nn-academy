---
title: "Tokenization: BPE & SentencePiece"
slug: "049-tokenization"
description: "How LLMs convert text to numbers. Implement byte-pair encoding from scratch, understand vocabulary design, and see why tokenization matters for model behavior."
track: "nn-advanced"
order: 4
read_time: 22
code_time: 18
execution_timeout: 10
prerequisites: ["043-word-embeddings", "048-transformer-architecture"]
---

# Tokenization: BPE & SentencePiece

LLMs don't see characters or words. They see ==token IDs==. Tokenization is the bridge between raw text and the model's vocabulary. Bad tokenization wastes capacity; good tokenization compresses language efficiently.

## Why Not Characters or Words?

| Approach | Vocab Size | Problem |
|----------|-----------|---------|
| Character-level | ~100 | Sequences too long, slow training |
| Word-level | ~100K+ | Rare/unseen words → OOV, huge embeddings |
| Subword (BPE) | ~30K-100K | ==Sweet spot: compact, no OOV, morphologically aware== |

## Byte-Pair Encoding (BPE)

Start with individual bytes. Iteratively merge the most frequent adjacent pair:

```python
import numpy as np
from collections import Counter

def train_bpe(texts, num_merges=1000):
    """Train BPE vocabulary from list of strings."""
    # Initialize with character-level tokens
    vocab = set()
    tokenized = []
    for text in texts:
        tokens = list(text)
        tokenized.append(tokens)
        vocab.update(tokens)
    
    merges = []
    for _ in range(num_merges):
        # Count all adjacent pairs
        pair_counts = Counter()
        for tokens in tokenized:
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i+1])] += 1
        
        if not pair_counts:
            break
        
        # Merge most frequent pair
        best_pair = pair_counts.most_common(1)[0][0]
        merges.append(best_pair)
        
        # Apply merge to all tokenized texts
        new_tokenized = []
        merged_token = "".join(best_pair)
        for tokens in tokenized:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokenized.append(new_tokens)
        tokenized = new_tokenized
        vocab.add(merged_token)
    
    return vocab, merges
```

<PyRunner
  cellId="049-cell-1"
  defaultCode={`from collections import Counter

texts = [
    "low lower lowest",
    "new newer newest",
    "wide wider widest",
] * 100

# Simple BPE training
tokenized = [list(t) for t in texts]
merges = []

for step in range(15):
    pairs = Counter()
    for tokens in tokenized:
        for i in range(len(tokens)-1):
            pairs[(tokens[i], tokens[i+1])] += 1
    
    if not pairs:
        break
    best = pairs.most_common(1)[0][0]
    merges.append(best)
    merged = "".join(best)
    
    new_tok = []
    for tokens in tokenized:
        nt = []; i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                nt.append(merged); i += 2
            else:
                nt.append(tokens[i]); i += 1
        new_tok.append(nt)
    tokenized = new_tok

print("BPE Merges (first 15):")
for i, (a, b) in enumerate(merges):
    print(f"  Step {i+1:2d}: '{a}' + '{b}' → '{a}{b}'")

print(f"\nSample tokenization after merging:")
print(f"  'lowest' → {tokenized[0][:6]}")
print(f"\n💡 BPE discovers morphemes: low+est, new+est, wid+est")
`}
/>

## How BPE Works in Practice

Modern tokenizers (GPT, LLaMA) use byte-level BPE:

1. Encode text as UTF-8 bytes (no Unicode issues)
2. Train BPE on byte sequences
3. Map merged tokens to integer IDs
4. Special tokens: `[BOS]`, `[EOS]`, `[PAD]`, `[UNK]` (rarely needed)

> [!IMPORTANT] Why Tokenization Matters
> - Vocabulary size determines embedding matrix size
> - Token count determines sequence length → attention cost (O(n²))
> - Multilingual models need more tokens per word in non-Latin scripts
> - Code/math tokenize differently than prose

## Encoding & Decoding

```python
def encode(text, merges, vocab_to_id):
    """Greedy left-to-right encoding."""
    tokens = list(text)
    for a, b in merges:
        merged = a + b
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - len(merged.split()) + 1:
                candidate = "".join(tokens[i:i+len(merged)])
                if candidate == merged:
                    new_tokens.append(merged)
                    i += len(merged)
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return [vocab_to_id.get(t, 0) for t in tokens]
```

<PyRunner
  cellId="049-cell-2"
  defaultCode={`# Show how different texts tokenize
test_words = ["unhappiness", "internationalization", "cat", "cats"]

# Simulated BPE vocab (typical patterns)
vocab_demo = {
    "un": 100, "happi": 101, "ness": 102,
    "inter": 200, "nation": 201, "al": 202, "ization": 203,
    "cat": 300, "s": 301, "▁cat": 302, "▁cats": 303,
}

print("Subword tokenization examples:")
print(f"{'Word':<25} | Tokens")
print("─" * 50)
splits = {
    "unhappiness": ["un", "happi", "ness"],
    "internationalization": ["inter", "nation", "al", "ization"],
    "cat": ["▁cat"],
    "cats": ["▁cats"],
}
for word, toks in splits.items():
    ids = [vocab_demo.get(t, 0) for t in toks]
    print(f"{word:<25} | {toks} → {ids}")

print(f"\n✅ Morphology-aware: 'unhappiness' = un + happi + ness")
print(f"   Common words stay whole: 'cat' = single token")
print(f"   Rare words decompose gracefully")
`}
/>

## Practical Considerations

| Factor | Recommendation |
|--------|---------------|
| Vocab size | 32K (English), 100K+ (multilingual) |
| Max token length | Usually 4-6 chars for English |
| Pre-trained tokenizer | Always reuse when fine-tuning |
| Custom domain | Retrain if domain vocabulary differs significantly |

<Quiz
  chapterSlug="049-tokenization"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does BPE avoid the out-of-vocabulary problem?",
      options: ["It uses a very large vocabulary", "Every word can be decomposed into subwords down to individual bytes, so any string is representable", "It uses character-level fallback", "It adds an UNK token"],
      correctIndex: 1,
      explanation: "BPE starts from bytes/characters and merges upward. Even completely unseen words decompose into known subword units. There is always a valid tokenization.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does tokenization affect model performance?",
      options: ["It doesn't, only architecture matters", "Token count determines sequence length (O(n²) attention cost) and vocabulary size determines embedding parameters; inefficient tokenization wastes both", "Larger vocabularies always improve quality", "Tokenization only affects inference speed"],
      correctIndex: 1,
      explanation: "More tokens per word = longer sequences = quadratic attention cost. Smaller vocab = smaller embeddings but less expressive. The right balance directly impacts training efficiency and model quality.",
      randomize: false,
    }
  ]}
/>
