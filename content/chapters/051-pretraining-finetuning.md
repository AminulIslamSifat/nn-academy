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
      prompt: "What is the training objective during pretraining?",
      options: ["Minimize perplexity on a benchmark", "Next-token prediction: predict each token from all previous tokens", "Classify text into categories", "Generate summaries of input text"],
      correctIndex: 1,
      explanation: "Pretraining uses causal language modeling: given tokens x₁...xₜ₋₁, predict xₜ. The text itself provides the labels — no human annotation needed.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why does next-token prediction implicitly teach world knowledge?",
      options: ["It doesn't — models only learn syntax", "Predicting the next token requires understanding context, facts, grammar, and reasoning. You can't predict 'Paris' after 'The capital of France is' without knowing geography.", "Because the dataset is labeled with facts", "Because attention memorizes everything"],
      correctIndex: 1,
      explanation: "The prediction objective is a proxy for understanding. Accurate prediction of domain-specific tokens requires internalizing the underlying knowledge.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In supervised fine-tuning (SFT), why are prompt tokens masked from the loss?",
      options: ["Prompts are too long to compute loss on", "We want the model to learn to generate responses, not predict user inputs. Computing loss on prompts teaches the wrong behavior.", "To save GPU memory", "Prompts use different tokenization"],
      correctIndex: 1,
      explanation: "If loss is computed on prompt tokens, the model learns to reproduce user messages instead of generating helpful assistant responses. Masking ensures gradients only flow through the response portion.",
      randomize: true
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What does this mask look like for prompt_len=3, response_len=5?",
      code: "import numpy as np\nprompt_len = 3\nresponse_len = 5\ntotal = prompt_len + response_len\nmask = np.zeros(total)\nmask[prompt_len:] = 1\nprint(mask.astype(int))",
      options: ["[0 0 0 1 1 1 1 1]", "[1 1 1 0 0 0 0 0]", "[0 0 0 0 0 1 1 1]", "[1 1 1 1 1 0 0 0]"],
      correctIndex: 0,
      explanation: "First 3 positions (prompt) are 0, last 5 positions (response) are 1. The mask zeros out loss contribution from prompt tokens.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "How does the learning rate for fine-tuning typically compare to pretraining?",
      options: ["Same learning rate", "10-100× smaller than pretraining", "10-100× larger than pretraining", "Learning rate doesn't matter for fine-tuning"],
      correctIndex: 1,
      explanation: "Fine-tuning adjusts behavior, not foundational knowledge. A high LR would destroy pretrained weights (catastrophic forgetting). Smaller LR makes gentle adjustments.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What is catastrophic forgetting in the context of fine-tuning?",
      options: ["Running out of GPU memory", "The model loses pretrained general knowledge when fine-tuned with too high a learning rate or too many epochs on narrow data", "Forgetting to save checkpoints", "Tokenization errors during fine-tuning"],
      correctIndex: 1,
      explanation: "Catastrophic forgetting occurs when gradient updates overwrite pretrained weights that encoded useful general knowledge. Lower LR and fewer epochs mitigate this.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "During pretraining, the loss is computed over ___ tokens. During SFT, the loss is computed over ___ tokens only.",
      options: ["all, response", "response, all", "prompt, response", "all, prompt"],
      correctIndex: 0,
      explanation: "Pretraining computes loss on every token (each predicts the next). SFT masks prompt tokens and only computes loss on response tokens.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What does LoRA stand for and what does it do?",
      options: ["Low-Rank Adaptation: decomposes weight updates into two small matrices ΔW = BA where r << d, keeping original weights frozen", "Long-Range Attention: extends the context window", "Loss-Regularized Alignment: adds regularization to SFT", "Layer-wise Optimization Rate Adjustment: per-layer learning rates"],
      correctIndex: 0,
      explanation: "LoRA freezes the pretrained W and learns small matrices A (d×r) and B (r×d) with rank r << d. This reduces trainable parameters to under 1% while preserving pretrained knowledge.",
      randomize: true
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is the output shape of lora_forward?",
      code: "import numpy as np\nx = np.random.randn(4, 512)   # (batch, d_model)\nW = np.random.randn(512, 512) # frozen weights\nA = np.random.randn(512, 8)   # r=8\nB = np.random.randn(8, 512)\nout = x @ W + 1.0 * (x @ A) @ B\nprint(out.shape)",
      options: ["(4, 512)", "(4, 8)", "(512, 512)", "(4, 512, 8)"],
      correctIndex: 0,
      explanation: "x @ W → (4, 512). (x @ A) → (4, 8), then @ B → (4, 512). Sum preserves shape (4, 512). LoRA adds a low-rank correction to the original linear transform.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Why does LoRA preserve pretrained knowledge better than full fine-tuning?",
      options: ["It uses more parameters", "Original weights W are frozen and never updated. Only the small adapter matrices A and B change, limiting how much the model can deviate from its pretrained state.", "It uses a higher learning rate", "It adds noise during training"],
      correctIndex: 1,
      explanation: "By freezing W, LoRA guarantees the pretrained function x@W is always present. The low-rank update BA can only make small directional changes, acting as a built-in regularization.",
      randomize: true
    },
    {
      id: "q11",
      type: "ordering",
      prompt: "Order the typical LLM development pipeline:",
      items: ["Collect massive unlabeled text corpus", "Pretrain with next-token prediction", "Supervised fine-tuning on instruction-response pairs", "RLHF alignment with human preferences", "Deploy as chat model"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Pretraining builds base knowledge → SFT teaches instruction-following format → RLHF aligns outputs with human values → deployment. Each phase builds on the previous.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "A base model after pretraining can complete text but follows instructions poorly. What phase fixes this?",
      options: ["More pretraining", "Supervised fine-tuning (SFT)", "Increasing model size", "Changing the tokenizer"],
      correctIndex: 1,
      explanation: "Base models are completion engines. SFT on instruction-response pairs teaches the model the format and behavior expected of an assistant. Pretraining alone doesn't teach instruction following.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "In the SFT loss formula, what happens if response_mask is all zeros?",
      code: "loss = -np.sum(mask * log_probs) / np.sum(mask)",
      options: ["Loss equals zero", "Division by zero error", "Loss equals the mean of log_probs", "The model trains normally"],
      correctIndex: 1,
      explanation: "np.sum(mask) = 0 when mask is all zeros, causing division by zero. In practice, every training example must have at least one response token.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "How many epochs does pretraining typically use vs fine-tuning?",
      options: ["Both use many epochs", "Pretraining: ~1 pass over massive data; Fine-tuning: multiple epochs over small data", "Pretraining: many epochs; Fine-tuning: 1 pass", "Both use exactly 1 epoch"],
      correctIndex: 1,
      explanation: "Pretraining data is so large that 1 epoch suffices (seeing each example once). Fine-tuning data is small, so multiple passes are needed to learn the task format.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "Which statement about adapter methods (LoRA, prefix tuning) is correct?",
      options: ["They update all model parameters", "They update fewer than 1% of parameters while keeping the base model frozen", "They require more GPU memory than full fine-tuning", "They only work for classification tasks"],
      correctIndex: 1,
      explanation: "Parameter-efficient fine-tuning methods like LoRA add small trainable modules while freezing the vast majority of pretrained weights. This enables fine-tuning on consumer GPUs.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why is pretraining described as 'unsupervised'?",
      options: ["No GPU is used", "No human-labeled data is needed — the text itself serves as both input and target via next-token prediction", "No loss function is used", "The model architecture is unknown"],
      correctIndex: 1,
      explanation: "In causal LM pretraining, every token in the sequence is automatically a label for the previous tokens. No manual annotation required — just raw text.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "You fine-tune a base model on medical QA but notice it forgot how to write code. What likely happened?",
      options: ["Medical data was corrupted", "Catastrophic forgetting: the fine-tuning LR was too high or too many epochs overwrote general capabilities", "The tokenizer changed", "Medical QA is incompatible with coding"],
      correctIndex: 1,
      explanation: "Narrow fine-tuning with aggressive updates can erase broad pretrained knowledge. Solutions: lower LR, fewer epochs, LoRA, or mixing in general data during fine-tuning.",
      randomize: true
    },
    {
      id: "q18",
      type: "fill-blank",
      prompt: "In LoRA with d_model=512 and rank r=8, the number of trainable parameters per adapter is ___ compared to ___ for the full weight matrix.",
      options: ["8,192 (2×512×8), 262,144 (512×512)", "262,144, 8,192", "4,096, 262,144", "512, 8"],
      correctIndex: 0,
      explanation: "LoRA: A is (512×8)=4096 params, B is (8×512)=4096 params, total 8192. Full W: 512×512=262,144. That's ~3.1% of parameters — massive efficiency gain.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What distinguishes a 'base model' from a 'chat model'?",
      options: ["Different architectures", "Base models are pretrained only (text completion); chat models are additionally fine-tuned to follow instructions and engage in dialogue", "Chat models are smaller", "Base models can't generate text"],
      correctIndex: 1,
      explanation: "Base models predict the next token. Chat models have been SFT'd (and often RLHF'd) to understand instructions, maintain conversation format, and be helpful/safe.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which best explains why fine-tuning requires less data than pretraining?",
      options: ["Fine-tuning uses better hardware", "Pretraining must learn language from scratch; fine-tuning only needs to adjust existing knowledge toward a specific behavior or format", "Fine-tuning data is higher quality", "Pretraining wastes data"],
      correctIndex: 1,
      explanation: "The pretrained model already understands language, facts, and reasoning. Fine-tuning just redirects these capabilities toward a specific interaction pattern, requiring far fewer examples.",
      randomize: true
    }
  ]}
/>
