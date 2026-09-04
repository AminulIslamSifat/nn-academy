---
title: "Pretraining & Fine-Tuning"
slug: "051-pretraining-finetuning"
description: "How LLMs learn: unsupervised pretraining on massive text, then supervised fine-tuning for specific tasks. Understand the two-phase paradigm in NumPy terms."
track: "nn-advanced"
order: 6
read_time: 22
code_time: 15
execution_timeout: 10
prerequisites: ["050-gpt-from-scratch"]
---

# Pretraining & Fine-Tuning

Modern LLMs don't train from scratch for each task. They use a ==two-phase paradigm==: learn language generally, then specialize.

## Phase 1: Pretraining

Train on massive unlabeled text with next-token prediction:

<BlockMath latex="\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{\lt t})" />

No labels needed. The text IS the label. Every token predicts the next one.

| Aspect | Detail |
|--------|--------|
| Data | TB of web text, books, code |
| Objective | Next-token prediction (causal LM) |
| Compute | Weeks-months on thousands of GPUs |
| Result | Base model: knows language, facts, reasoning patterns |

```python
# Pretraining is just standard GPT training at scale
def pretrain_step(model, batch_token_ids):
    """batch_token_ids: (N, T+1)"""
    inputs = batch_token_ids[:, :-1]
    targets = batch_token_ids[:, 1:]
    
    logits = model.forward(inputs)
    # Cross-entropy loss over all positions
    loss = cross_entropy(logits, targets)
    return loss
```

> [!IMPORTANT] Why Pretraining Works
> Predicting the next token forces the model to learn grammar, facts, reasoning, and world knowledge. You can't predict "Paris" after "The capital of France is" without knowing geography. The objective implicitly teaches everything.

## Phase 2: Fine-Tuning

Adapt the pretrained model to specific tasks with labeled data:

### Supervised Fine-Tuning (SFT)

Train on instruction-response pairs:

```
User: What is photosynthesis?
Assistant: Photosynthesis is the process by which plants convert...
```

Only compute loss on the assistant's tokens, not the user prompt:

```python
def sft_loss(model, input_ids, response_ids, response_mask):
    """response_mask: 1 for response tokens, 0 for prompt tokens."""
    full_ids = np.concatenate([input_ids, response_ids], axis=1)
    logits = model.forward(full_ids[:, :-1])
    targets = full_ids[:, 1:]
    
    # Masked cross-entropy: only loss on response tokens
    mask = response_mask[:, 1:]  # shift to match targets
    loss = -np.sum(mask * log_probs) / np.sum(mask)
    return loss
```

<PyRunner
  cellId="051-cell-1"
  defaultCode={`import numpy as np

# Demonstrate masked loss concept
prompt_len = 5
response_len = 8
total = prompt_len + response_len

mask = np.zeros(total)
mask[prompt_len:] = 1  # only response tokens count

print("Supervised Fine-Tuning Loss Masking:")
print(f"  Prompt tokens:   {prompt_len} (loss=0, no gradient)")
print(f"  Response tokens: {response_len} (loss computed)")
print(f"  Mask: {mask.astype(int)}")
print(f"\n💡 Model learns to GENERATE responses, not predict prompts")
print(f"   This is what turns a base model into a chat model")
`}
/>

### Key Differences: Pretrain vs Fine-Tune

| Aspect | Pretraining | Fine-Tuning |
|--------|------------|-------------|
| Data volume | TB | KB-MB |
| Learning rate | Higher | Lower (10-100× smaller) |
| Epochs | 1 pass over data | Multiple epochs |
| Loss masking | All tokens | Response tokens only |
| Goal | Learn language | Learn behavior/format |

> [!NOTE] Why Lower LR for Fine-Tuning?
> The pretrained model already knows language. Fine-tuning adjusts behavior, not knowledge. High LR would destroy pretrained knowledge (catastrophic forgetting).

## Adapter Methods (Parameter-Efficient Fine-Tuning)

Full fine-tuning updates ALL parameters. Adapters update under 1%:

- **LoRA**: Low-rank decomposition of weight updates
- **Prefix tuning**: Learnable prefix vectors prepended to keys/values
- **Prompt tuning**: Learnable soft prompts

```python
# LoRA concept: ΔW = BA where B is (d,r), A is (r,d), r << d
def lora_forward(x, W, A, B, alpha=1.0):
    """Original weights frozen; only A and B are trained."""
    return x @ W + alpha * (x @ A) @ B
```

<Quiz
  chapterSlug="051-pretraining-finetuning"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does next-token prediction teach so much?",
      options: ["It's a simple objective", "Predicting the next token requires understanding context, facts, grammar, and reasoning — the objective implicitly encodes all of language understanding", "Because models are large", "Because of attention"],
      correctIndex: 1,
      explanation: "You cannot accurately predict 'is' in 'The result of 2+2 ___ 4' without arithmetic. You cannot predict 'Shakespeare' after 'Hamlet was written by' without cultural knowledge. The prediction objective is a proxy for general understanding.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why mask prompt tokens during SFT loss computation?",
      options: ["Prompts are too long", "We want the model to learn to generate responses, not predict user inputs; computing loss on prompts would teach the wrong behavior", "To save memory", "Prompts have different tokenization"],
      correctIndex: 1,
      explanation: "If you compute loss on prompt tokens, the model learns to predict user messages instead of generating helpful responses. Masking ensures gradients only flow through the assistant's output.",
      randomize: false,
    }
  ]}
/>
