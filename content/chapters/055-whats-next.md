---
title: "What's Next: Beyond the Basics"
slug: "055-whats-next"
description: "Where to go from here: efficient architectures, multimodal models, agents, and the frontier of LLM research. Your roadmap for continued learning."
track: "nn-advanced"
order: 10
read_time: 15
code_time: 5
execution_timeout: 5
prerequisites: ["054-mini-llm-capstone"]
---

# What's Next: Beyond the Basics

You've built neural networks from scratch, from MNIST to mini-GPT. ==Here's where the field is going== and how to keep learning.

## Efficient Architectures

The O(n²) attention bottleneck is being attacked:

| Approach | Idea | Status |
|----------|------|--------|
| Sparse Attention | Attend to subset of tokens | Production (Longformer) |
| Linear Attention | Replace softmax with kernel trick | Research |
| State Space Models (Mamba) | Recurrent-like linear-time sequence modeling | Production-ready |
| Ring Attention | Distribute attention across devices | Training infrastructure |
| KV Cache Quantization | Compress cached keys/values | Inference optimization |

> [!NOTE] Mamba/SSM
> State space models achieve near-transformer quality with linear-time inference. Hybrid architectures (Transformer + SSM) may be the next dominant paradigm.

## Multimodal Models

Text + images + audio + video in one model:

- **Vision-Language**: CLIP, LLaVA, GPT-4V
- **Audio**: Whisper, AudioLM
- **Video**: Sora, Gen-3
- **Unified**: Gemini, GPT-4o

Key insight: align different modalities into shared embedding space, then let the transformer handle everything.

## Agents & Tool Use

LLMs that act, not just generate:

```
User: What's the weather in Dhaka?
Agent: [calls weather API] → "32°C, humid, chance of rain"
User: Book a flight to Rajshahi
Agent: [searches flights] → [compares prices] → "Best option: BG-452, ৳3200"
```

Research frontiers: planning, self-correction, multi-step reasoning, memory.

## Open Problems

| Problem | Why It Matters |
|---------|---------------|
| Hallucination | Models confidently generate falsehoods |
| Long context | Beyond 128K tokens, quality degrades |
| Reasoning | Still brittle compared to humans |
| Alignment | Ensuring models do what we actually want |
| Efficiency | Making LLMs accessible without datacenter-scale compute |
| Evaluation | We don't fully know how to measure intelligence |

## Your Learning Roadmap

### Immediate Next Steps
1. **Rebuild this curriculum in PyTorch** — translate NumPy implementations to framework code
2. **Read key papers**: Attention Is All You Need, GPT-2/3, Chinchilla, DPO
3. **Fine-tune an open model**: LLaMA-3, Mistral, Qwen on your own data
4. **Build something**: Chatbot, code assistant, document QA, personal agent

### Deeper Study
- **Math**: Linear algebra, probability, optimization theory
- **Systems**: CUDA, distributed training, inference optimization
- **Research**: Follow arxiv, attend conferences, reproduce papers
- **Community**: Contribute to open-source models, share your work

> [!IMPORTANT] Final Words
> You now understand neural networks from first principles. Not from a library, not from a tutorial — from the math up. That foundation will serve you regardless of how the field evolves. Frameworks change. Architectures evolve. But the fundamentals you've built here are permanent.
>
> Now go build something that matters. 💛

<Quiz
  chapterSlug="055-whats-next"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What's the best way to solidify what you've learned in this curriculum?",
      options: ["Read more textbooks", "Rebuild key components in a framework like PyTorch and apply them to a real project", "Memorize formulas", "Watch more tutorials"],
      correctIndex: 1,
      explanation: "Active application cements understanding. Translating NumPy implementations to PyTorch bridges theory and practice. Building a real project reveals gaps that passive study cannot.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why might state space models (Mamba) complement transformers?",
      options: ["They're simpler to implement", "They offer linear-time sequence processing vs transformers' quadratic attention, enabling much longer contexts efficiently", "They don't need training", "They replace attention entirely"],
      correctIndex: 1,
      explanation: "Transformers excel at parallel training and rich attention patterns but scale quadratically. SSMs process sequences in linear time. Hybrids could get the best of both worlds.",
      randomize: false,
    }
  ]}
/>
