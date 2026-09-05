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
      prompt: "What is the overall goal of the mini-LLM capstone project?",
      options: ["Build a production chatbot", "Demonstrate every concept from chapters 046-053 by training a tiny language model from scratch in NumPy", "Beat GPT-4 on benchmarks", "Learn PyTorch"],
      correctIndex: 1,
      explanation: "The capstone integrates tokenization, transformer architecture, training, and generation into one complete pipeline to solidify understanding of how LLMs work end-to-end.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does a mini-LLM generate grammatically plausible but factually nonsensical text?",
      options: ["The architecture is wrong", "It learned local patterns (grammar, common phrases) from limited data but lacks the scale for global coherence, facts, and reasoning", "Temperature is too high", "Needs more attention heads"],
      correctIndex: 1,
      explanation: "Grammar is a local statistical pattern learnable from small data. Factual knowledge and long-range reasoning require seeing billions of tokens across diverse domains.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In the training loop, why do we slice batch[:, :-1] for inputs and batch[:, 1:] for targets?",
      options: ["To remove padding", "Next-token prediction: input is tokens 0..T-1, target is tokens 1..T. Each position predicts the next token.", "To reduce memory usage", "To separate prompt from response"],
      correctIndex: 1,
      explanation: "Causal LM training shifts the sequence by one position. Position t receives tokens 0..t-1 as input and must predict token t. This is the core pretraining objective.",
      randomize: true
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What shape are the logits after model.forward(batch[:, :-1]) if batch is (32, 128)?",
      code: "import numpy as np\nbatch = np.random.randint(0, 100, (32, 128))\ninputs = batch[:, :-1]\nprint('Input shape:', inputs.shape)\n# Assuming vocab_size=2000\nlogits_shape = (inputs.shape[0], inputs.shape[1], 2000)\nprint('Logits shape:', logits_shape)",
      options: ["(32, 127, 2000)", "(32, 128, 2000)", "(32, 127)", "(127, 2000)"],
      correctIndex: 0,
      explanation: "batch[:, :-1] removes the last token → shape (32, 127). The model outputs logits of shape (batch, seq_len, vocab_size) = (32, 127, 2000).",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Why is d_model=64 recommended for the mini-LLM instead of 512 or 1024?",
      options: ["Larger models can't learn language", "d_model=64 keeps the model trainable on CPU within reasonable time while still demonstrating all architectural concepts", "64 is the only valid dimension", "Attention doesn't work with larger dimensions"],
      correctIndex: 1,
      explanation: "The goal is educational, not competitive. d=64 allows full forward+backward passes on CPU in seconds. Larger dimensions would make iteration painfully slow without adding conceptual value.",
      randomize: true
    },
    {
      id: "q6",
      type: "fill-blank",
      prompt: "With d_model=64 and num_heads=4, each head operates on vectors of dimension d_k = ___.",
      options: ["16", "64", "4", "256"],
      correctIndex: 0,
      explanation: "d_k = d_model / num_heads = 64 / 4 = 16. Multi-head attention splits the representation into smaller subspaces per head.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "Why is ff_dim typically set to 4× d_model?",
      options: ["It's a hardware requirement", "Empirical convention: provides enough capacity for non-linear transformations while keeping parameter count manageable", "Mathematical proof requires it", "Attention needs it"],
      correctIndex: 1,
      explanation: "The feed-forward network expands to 4× d_model then projects back. This bottleneck design gives the model room for non-linear feature processing. 4× is standard from the original Transformer paper.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "During generation, why do we use ids[-128:] when max_len=128?",
      options: ["To speed up computation", "The model was trained with max context length 128. Feeding more tokens would exceed positional encoding bounds.", "To save memory", "128 is always optimal"],
      correctIndex: 1,
      explanation: "Positional encodings are defined for positions 0..max_len-1. Feeding more tokens than max_len causes out-of-bounds errors or undefined behavior. We keep only the most recent 128 tokens.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "What role does temperature play in the generate function?",
      code: "logits = model.forward(x)[0, -1] / temperature",
      options: ["It changes the model weights", "Dividing logits by temperature > 1 makes the distribution softer (more random); < 1 makes it sharper (more deterministic)", "It controls batch size", "It adjusts learning rate"],
      correctIndex: 1,
      explanation: "Temperature scales logits before softmax. T>1 flattens the distribution (more exploration), T<1 sharpens it (more greedy), T=1 leaves it unchanged.",
      randomize: true
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What happens to probabilities when temperature approaches 0?",
      code: "import numpy as np\nlogits = np.array([1.0, 2.0, 3.0])\nfor temp in [1.0, 0.1, 0.01]:\n    scaled = logits / temp\n    probs = np.exp(scaled - np.max(scaled))\n    probs /= probs.sum()\n    print(f'T={temp:.2f}: {probs.round(3)}')",
      options: ["Distribution becomes uniform", "Distribution concentrates on the highest-logit token (argmax-like)", "All probabilities become 0", "Error: division by zero"],
      correctIndex: 1,
      explanation: "As T→0, the largest logit dominates exponentially. T=0.01 gives ~[0, 0, 1]. This is equivalent to greedy decoding. Note: T=0 exactly would cause division by zero.",
      randomize: true
    },
    {
      id: "q11",
      type: "ordering",
      prompt: "Order the steps of autoregressive generation:",
      items: ["Tokenize the prompt into IDs", "Take the last max_len tokens as input", "Run forward pass through the model", "Get logits for the last position", "Apply temperature scaling and softmax", "Sample next token from probability distribution", "Append sampled token to the sequence"],
      correctOrder: [0, 1, 2, 3, 4, 5, 6],
      explanation: "Generation is a loop: tokenize → truncate context → forward → extract last position logits → scale & normalize → sample → append → repeat from step 2.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why use np.logaddexp.reduce instead of naive log(sum(exp(logits))) for computing log_probs?",
      options: ["It's faster", "Numerical stability: exp of large logits overflows. logaddexp uses the log-sum-exp trick internally to prevent overflow/underflow", "It gives different results", "NumPy doesn't support sum(exp())"],
      correctIndex: 1,
      explanation: "log_probs = logits - log(sum(exp(logits))). If any logit is large (e.g., 1000), exp(1000) = inf. logaddexp.reduce subtracts the max first, keeping values in safe range.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "How are training subsequences created from the corpus?",
      code: "indices = np.random.randint(0, N - seq_len, batch_size)\nbatch = np.array([token_ids[i:i+seq_len] for i in indices])",
      options: ["Sequential chunks from start to end", "Random starting positions are sampled, creating overlapping subsequences of length seq_len from anywhere in the corpus", "Only non-overlapping windows", "Every possible subsequence is used"],
      correctIndex: 1,
      explanation: "Random sampling creates stochastic batches. Overlapping subsequences provide more training signal than sequential chunking. This is standard practice for language model training.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What is the most important takeaway from building a mini-LLM from scratch?",
      options: ["NumPy is production-ready for LLMs", "The same architecture that generates nonsense at small scale produces remarkable capabilities at large scale — architecture + scale together create emergent abilities", "Transformers are too complex to understand", "You need PyTorch to build LLMs"],
      correctIndex: 1,
      explanation: "Understanding the architecture demystifies LLMs. The mini-LLM proves you understand every component. Scale (data + parameters + compute) is what transforms this architecture into something powerful.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "A mini-LLM trained on 10KB of text for 20 epochs will NOT be able to:",
      options: ["Generate grammatically plausible text", "Repeat common phrases from training data", "Follow instructions or reason about novel problems", "Show basic word-level coherence"],
      correctIndex: 2,
      explanation: "Instruction following and reasoning are emergent properties requiring massive scale (billions of tokens, billions of parameters) plus fine-tuning. A tiny model on tiny data learns statistics, not reasoning.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why is BPE preferred over character-level tokenization for the mini-LLM?",
      options: ["BPE is simpler to implement", "BPE merges frequent character sequences into single tokens, reducing sequence length and allowing the model to capture word-level patterns more efficiently", "Character-level doesn't work with transformers", "BPE requires less memory"],
      correctIndex: 1,
      explanation: "Character-level means 'understanding' takes many more tokens. BPE compresses common words/patterns into single tokens, making the 128-token context window much more expressive.",
      randomize: true
    },
    {
      id: "q17",
      type: "fill-blank",
      prompt: "With d_model=64, num_layers=2, num_heads=4, ff_dim=256, the approximate parameter count per transformer layer is ___ × d_model² + d_model × ff_dim × 2 ≈ ___.",
      options: ["roughly 100K-200K total for the whole model", "exactly 64", "over 1 billion", "zero"],
      correctIndex: 0,
      explanation: "Each layer has attention (~4×d²) + FFN (~2×d×ff_dim) + norms. With these hyperparameters, total model params are roughly 100K-200K including embeddings. Tiny compared to real LLMs.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "In the loss computation, what does log_probs[:, np.arange(seq_len-1), targets] achieve?",
      options: ["Selects random positions", "Advanced indexing: for each batch element and each position t, selects the log probability assigned to the actual target token at that position", "Computes the mean of all logits", "Filters out padding tokens"],
      correctIndex: 1,
      explanation: "targets has shape (N, seq_len-1). For each (batch_idx, position), we index into the vocab dimension to get the log prob of the correct next token. This is the cross-entropy numerator.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Which concept from earlier chapters is NOT directly used in the mini-LLM pipeline?",
      options: ["Multi-head attention (ch 047)", "Batch normalization (ch 037)", "Positional encoding (ch 048)", "Autoregressive generation (ch 050)"],
      correctIndex: 1,
      explanation: "Transformers use Layer Normalization, not Batch Normalization. BN was covered in the CNN chapters (037) but isn't part of the standard transformer architecture used in the mini-LLM.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "If your mini-LLM's loss plateaus at a high value and generated text is gibberish, what should you check FIRST?",
      options: ["Add more layers immediately", "Verify the tokenizer maps tokens consistently between training and generation, check that targets are correctly shifted by 1, and ensure the loss computation indexes the right positions", "Increase temperature", "Switch to character-level tokenization"],
      correctIndex: 1,
      explanation: "Common bugs: off-by-one in target shifting, inconsistent tokenization between train/generate, wrong indexing in loss computation. Always verify correctness of the pipeline before scaling up.",
      randomize: true
    }
  ]}
/>
