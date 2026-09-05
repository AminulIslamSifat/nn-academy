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
      options: [
        "It uses a very large vocabulary",
        "Every word can be decomposed into subwords down to individual bytes/characters, so any string is representable without needing UNK tokens",
        "It uses character-level fallback at runtime",
        "It adds an UNK token for unknown words"
      ],
      correctIndex: 1,
      explanation: "BPE starts from bytes/characters and merges upward during training. Even completely unseen words decompose into known subword units. There is always a valid tokenization — no OOV possible.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does tokenization directly affect model performance?",
      options: [
        "It doesn't — only architecture matters",
        "Token count determines sequence length (O(n²) attention cost) and vocabulary size determines embedding matrix size; inefficient tokenization wastes both compute and parameters",
        "Larger vocabularies always improve quality",
        "Tokenization only affects inference speed"
      ],
      correctIndex: 1,
      explanation: "More tokens per word = longer sequences = quadratic attention cost. Smaller vocab = smaller embeddings but less expressive tokens. The right balance directly impacts training efficiency and model quality.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the fundamental tradeoff in choosing vocabulary size?",
      options: [
        "Speed vs accuracy",
        "Small vocab: fewer embedding params but more tokens per word (longer sequences). Large vocab: fewer tokens but larger embedding matrix and more rare/unseen tokens.",
        "Training time vs inference time",
        "Memory vs disk space"
      ],
      correctIndex: 1,
      explanation: "Vocab size sits at the intersection of parameter efficiency (embedding matrix) and sequence efficiency (tokens per sentence). Typical sweet spot: 32K for English, 100K+ for multilingual.",
      randomize: true
    },
    {
      id: "q4",
      type: "ordering",
      prompt: "Order the BPE training algorithm steps:",
      items: ["Initialize with character/byte-level tokens", "Count all adjacent token pairs across corpus", "Merge the most frequent pair into a new token", "Repeat until desired vocabulary size"],
      correctOrder: [0, 1, 2, 3],
      explanation: "BPE is a greedy bottom-up merge algorithm. Start atomic, iteratively combine the most common neighbors. Each merge adds one token to the vocabulary. Stop when vocab reaches target size.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "After BPE merges ('l','o')→'lo' then ('lo','w')→'low', how is 'lower' tokenized?",
      code: "text = 'lower'\n# After merges: lo+w=low, so 'low' is one token\n# Remaining: 'e', 'r'\ntokens = ['low', 'e', 'r']\nprint(tokens)",
      options: ["['low', 'e', 'r']", "['l', 'o', 'w', 'e', 'r']", "['lower']", "['lo', 'wer']"],
      correctIndex: 0,
      explanation: "BPE applies merges greedily left-to-right. 'low' was merged first, consuming l-o-w. Remaining 'e' and 'r' weren't merged further. Result: 3 tokens instead of 5 characters.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Why do modern tokenizers use byte-level BPE instead of character-level?",
      options: [
        "Bytes are faster than characters",
        "UTF-8 bytes handle ALL Unicode without special casing. Character-level needs separate handling for every script, emoji, and special symbol. Byte-level is universal.",
        "Characters are too large for embeddings",
        "Byte-level produces shorter sequences"
      ],
      correctIndex: 1,
      explanation: "Unicode has 140K+ codepoints. UTF-8 uses only 256 byte values. Byte-level BPE works identically for English, Chinese, Arabic, emoji — no language-specific preprocessing needed.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "BPE discovers ___ structure automatically: 'unhappiness' → un + happi + ness, capturing meaningful linguistic units without explicit supervision.",
      options: ["syntactic", "morphological", "phonetic", "semantic"],
      correctIndex: 1,
      explanation: "Frequent prefixes (un-, re-), suffixes (-ness, -ing), and roots emerge naturally from frequency-based merging. BPE approximates morphology without linguistic knowledge — a key reason it outperforms character-level models.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "How does BPE encoding work at inference time?",
      options: [
        "Randomly splits words",
        "Greedy left-to-right: apply learned merges in training order, always matching the longest possible merged token first",
        "Uses dynamic programming for optimal splitting",
        "Looks up whole words in a dictionary"
      ],
      correctIndex: 1,
      explanation: "Encoding replays the merge list sequentially. At each step, replace all occurrences of the merged pair. This greedy approach is deterministic and matches the training procedure exactly.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why do multilingual models need larger vocabularies (100K+) than English-only models (32K)?",
      options: [
        "More languages means more words",
        "Non-Latin scripts (Chinese, Japanese, Arabic) don't share subword units with English. Each script needs its own set of frequent subwords, multiplying vocabulary requirements.",
        "Multilingual models are less efficient",
        "Unicode requires more tokens"
      ],
      correctIndex: 1,
      explanation: "English subwords (th, ing, tion) don't transfer to Chinese characters or Arabic roots. A multilingual vocab must cover frequent patterns in ALL supported languages, requiring proportionally more tokens.",
      randomize: true
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "If 'internationalization' tokenizes as ['inter', 'nation', 'al', 'ization'], how many tokens is this vs character-level?",
      code: "subword_tokens = 4\nchar_tokens = len('internationalization')\nprint(f'subword={subword_tokens}, char={char_tokens}, ratio={char_tokens/subword_tokens:.1f}x')",
      options: ["subword=4, char=20, ratio=5.0x", "subword=4, char=4, ratio=1.0x", "subword=20, char=4, ratio=0.2x", "subword=1, char=20, ratio=20.0x"],
      correctIndex: 0,
      explanation: "Subword tokenization compresses 20 characters into 4 tokens — 5× compression. This directly reduces sequence length and attention cost. Common morphemes stay intact while rare words decompose gracefully.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why should you always reuse the pre-trained tokenizer when fine-tuning a model?",
      options: [
        "It's faster than training a new one",
        "The model's embeddings were trained on specific token IDs. Changing the tokenizer changes the ID mapping, making all pre-trained embeddings meaningless.",
        "Pre-trained tokenizers are always optimal",
        "Fine-tuning doesn't involve tokenization"
      ],
      correctIndex: 1,
      explanation: "Embedding matrix row i corresponds to token ID i. A different tokenizer assigns different IDs to the same text, so the pre-trained embeddings no longer align with the input. Always match tokenizer to checkpoint.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "What happens to code/math text when tokenized with a prose-trained BPE?",
      options: [
        "It tokenizes identically to prose",
        "Code/math uses different character distributions (brackets, operators, variable names). Prose-trained BPE produces inefficient tokenizations — more tokens per concept, wasting sequence length.",
        "Code can't be tokenized",
        "Math symbols are ignored"
      ],
      correctIndex: 1,
      explanation: "BPE learns frequent patterns from its training data. Code has different frequent patterns (def, return, ==, {}) than prose. Domain-mismatched tokenization increases tokens per unit of meaning.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What are special tokens and why are they needed?",
      options: [
        "They're optional decorations",
        "[BOS] marks sequence start, [EOS] marks end, [PAD] fills batches to uniform length. These structural markers aren't in natural text but are essential for training and inference.",
        "They replace punctuation",
        "They encode capitalization"
      ],
      correctIndex: 1,
      explanation: "Special tokens provide structural signals the model can't infer from content alone. BOS tells the decoder where to start generating. EOS signals completion. PAD enables batched processing of variable-length sequences.",
      randomize: true
    },
    {
      id: "q14",
      type: "fill-blank",
      prompt: "BPE is a ___ algorithm that merges the most frequent adjacent pair at each step, building vocabulary from the bottom up.",
      options: ["greedy", "dynamic programming", "random", "supervised"],
      correctIndex: 0,
      explanation: "BPE makes locally optimal choices (most frequent pair) without considering global optimality. Despite being greedy, it produces effective vocabularies because frequent pairs correlate with meaningful linguistic units.",
      randomize: false
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "How does SentencePiece differ from original BPE?",
      options: [
        "They're identical",
        "SentencePiece operates directly on raw text (not pre-tokenized by whitespace), handles multiple languages uniformly, and includes unigram language model as an alternative to BPE",
        "SentencePiece only works for Japanese",
        "SentencePiece doesn't support subwords"
      ],
      correctIndex: 1,
      explanation: "Original BPE requires pre-tokenization (usually whitespace splitting), which fails for languages without spaces (Chinese, Japanese). SentencePiece treats input as a raw byte stream, making it truly language-agnostic.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Your LLM generates broken output for Bengali text but works fine for English. What's the likely tokenization issue?",
      options: [
        "The model doesn't support Bengali",
        "The tokenizer was trained primarily on English. Bengali characters/conjuncts get split into many byte-level tokens, producing excessively long sequences that exceed context limits or lose coherence.",
        "Bengali is inherently harder",
        "The attention mechanism fails for non-Latin scripts"
      ],
      correctIndex: 1,
      explanation: "English-optimized BPE allocates most vocab to English subwords. Bengali text may require 3-5× more tokens per word, consuming context budget and making the model's job harder. Solution: multilingual tokenizer or Bengali-specific vocab.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why does BPE typically produce tokens of 4-6 characters for English?",
      options: [
        "It's hardcoded",
        "This length corresponds to common morphemes and frequent word fragments in English. Shorter tokens waste compression; longer tokens become too rare to learn reliably from finite data.",
        "GPUs process 4-6 chars optimally",
        "It's arbitrary"
      ],
      correctIndex: 1,
      explanation: "BPE merge frequency follows Zipf's law. Common morphemes (prefixes, suffixes, roots) are typically 3-6 characters. The algorithm naturally converges to this range because these are the most frequently co-occurring adjacent units.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "In BPE, if pair ('t','h') appears 500 times and ('h','e') appears 300 times, which gets merged first?",
      code: "from collections import Counter\npairs = Counter({('t','h'): 500, ('h','e'): 300})\nbest = pairs.most_common(1)[0]\nprint(f\"{best[0]} → count={best[1]}\")",
      options: ["('t', 'h') → count=500", "('h', 'e') → count=300", "Both simultaneously", "Random selection"],
      correctIndex: 0,
      explanation: "BPE always merges the MOST frequent pair first. ('t','h') with 500 occurrences beats ('h','e') with 300. After merging 'th', pair counts are recomputed for the next iteration.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "When should you retrain a tokenizer instead of reusing a pre-trained one?",
      options: [
        "Always retrain for best results",
        "When your domain vocabulary differs significantly from the pre-trained tokenizer's training data (e.g., medical, legal, code). Otherwise, reuse to maintain embedding compatibility.",
        "Never retrain",
        "Only for non-English languages"
      ],
      correctIndex: 1,
      explanation: "Domain-specific terminology (drug names, legal terms, API calls) may tokenize poorly with general-purpose BPE. Custom tokenization improves efficiency. But if fine-tuning an existing model, you MUST keep its tokenizer.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "How does tokenization choice affect the effective context window of an LLM?",
      options: [
        "It doesn't — context window is fixed in tokens",
        "Context window is measured in tokens, not characters. Efficient tokenization (fewer tokens per word) means more actual content fits in the same token budget. Poor tokenization wastes context on subword fragmentation.",
        "Larger vocabularies increase context window",
        "Context window adapts automatically"
      ],
      correctIndex: 1,
      explanation: "A 4K-token window holds ~3000 words with good English tokenization but maybe ~1000 words with poor multilingual tokenization. Tokenization efficiency directly determines how much real content the model can process.",
      randomize: true
    }
  ]}
/>
