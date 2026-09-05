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
      prompt: "What is the primary computational bottleneck of standard transformer attention?",
      options: ["Feed-forward layers are too large", "Self-attention scales as O(n²) with sequence length, making long contexts expensive", "Embedding lookups are slow", "Normalization adds overhead"],
      correctIndex: 1,
      explanation: "Attention computes pairwise scores between all token pairs: n×n matrix. Doubling context length quadruples computation. This motivates efficient attention alternatives.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How do State Space Models (like Mamba) address the attention bottleneck?",
      options: ["They use larger GPUs", "They achieve near-transformer quality with linear-time O(n) sequence processing instead of quadratic attention", "They skip the feed-forward layer", "They only work on short sequences"],
      correctIndex: 1,
      explanation: "SSMs process tokens sequentially like RNNs but with modern training tricks. Linear scaling enables much longer contexts. Hybrid Transformer+SSM architectures may become the next dominant paradigm.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is sparse attention?",
      options: ["Using fewer attention heads", "Attending to only a subset of tokens (e.g., local windows, strided patterns) instead of all n² pairs, reducing computation while preserving most information", "Using float16 for attention", "Removing attention entirely"],
      correctIndex: 1,
      explanation: "Sparse attention selects which token pairs to attend to using predefined patterns (local, global, strided). Longformer uses this approach for production-scale long documents.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the key insight behind multimodal models?",
      options: ["Use separate models for each modality", "Align different modalities (text, image, audio) into a shared embedding space, then let the transformer handle everything uniformly", "Convert all inputs to text first", "Train each modality independently"],
      correctIndex: 1,
      explanation: "CLIP aligns text and images in shared space. LLaVA feeds visual tokens alongside text tokens. The unified representation lets one transformer reason across modalities.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What distinguishes an LLM agent from a plain chatbot?",
      options: ["Agents are larger models", "Agents can take actions — calling APIs, using tools, executing multi-step plans — rather than just generating text responses", "Agents use different tokenizers", "Agents don't use transformers"],
      correctIndex: 1,
      explanation: "A chatbot generates text. An agent reasons about goals, selects tools, executes actions, observes results, and iterates. Tool use transforms language models into general-purpose problem solvers.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What is hallucination in LLMs?",
      options: ["Visual artifacts in generated images", "Models confidently generate plausible-sounding but factually incorrect or fabricated information", "Memory overflow errors", "Training loss oscillation"],
      correctIndex: 1,
      explanation: "LLMs predict likely next tokens, not verified facts. They can produce fluent, confident text that is entirely wrong. This is a fundamental limitation of the prediction objective.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "KV cache quantization compresses cached ___ and ___ during inference to reduce memory usage for long-context generation.",
      options: ["keys, values", "queries, keys", "weights, biases", "inputs, outputs"],
      correctIndex: 0,
      explanation: "During autoregressive generation, past keys and values are cached to avoid recomputation. Quantizing these caches (e.g., to int8 or int4) dramatically reduces memory for long sequences.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "Why is 'alignment' listed as an open problem?",
      options: ["Models can't be aligned at all", "Ensuring models reliably do what humans actually want (not just what they say) is unsolved — current methods (RLHF, DPO) are imperfect and brittle", "Alignment is already solved by RLHF", "Only applies to vision models"],
      correctIndex: 1,
      explanation: "Alignment includes honesty, harmlessness, helpfulness, and following intent. Current techniques improve behavior but don't guarantee it. Misalignment risks grow with model capability.",
      randomize: true
    },
    {
      id: "q9",
      type: "ordering",
      prompt: "Order the recommended learning roadmap steps:",
      items: ["Complete this NumPy-from-scratch curriculum", "Rebuild key components in PyTorch", "Fine-tune an open model on custom data", "Build a real project (chatbot, QA system, agent)", "Read foundational papers and follow research"],
      correctOrder: [0, 1, 2, 3, 4],
      explanation: "Foundation first (NumPy), then framework fluency (PyTorch), then practical skills (fine-tuning), then application (projects), then staying current (research). Each step builds on the previous.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Which paper introduced the transformer architecture?",
      options: ["GPT-3: Language Models are Few-Shot Learners", "Attention Is All You Need (Vaswani et al., 2017)", "BERT: Pre-training of Deep Bidirectional Transformers", "ImageNet Classification with Deep CNNs"],
      correctIndex: 1,
      explanation: "'Attention Is All You Need' (2017) introduced self-attention, multi-head attention, positional encoding, and the encoder-decoder transformer. It replaced RNNs for sequence modeling.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What does the Chinchilla paper demonstrate?",
      options: ["Bigger models are always better", "Optimal training balances model size and data size — many large models were undertrained relative to their parameter count", "Small models beat large ones", "Data quality doesn't matter"],
      correctIndex: 1,
      explanation: "Chinchilla showed compute-optimal scaling: for a fixed budget, doubling params AND data beats doubling only params. Many pre-Chinchilla models wasted compute by being too large for their data.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why is evaluation of LLMs considered an open problem?",
      options: ["Benchmarks are too easy", "We lack comprehensive metrics that capture reasoning, creativity, safety, and real-world utility — existing benchmarks are narrow and gameable", "Evaluation is fully solved", "Only humans can evaluate models"],
      correctIndex: 1,
      explanation: "Perplexity measures prediction, not understanding. Benchmarks saturate and get gamed. Human eval is expensive and inconsistent. We genuinely don't know how to measure 'intelligence' in LLMs.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What is Ring Attention?",
      options: ["A circular neural network", "Distributing attention computation across multiple devices in a ring topology to train on longer sequences than fit in one GPU's memory", "Attention with periodic boundary conditions", "A regularization technique"],
      correctIndex: 1,
      explanation: "Ring Attention partitions the sequence across devices. Each device computes partial attention and passes KV blocks to the next device. Enables training with millions of tokens.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why should you rebuild this curriculum's implementations in PyTorch?",
      options: ["PyTorch is required for jobs", "Translating NumPy implementations to a framework bridges theory and practice, reveals gaps in understanding, and gives you production-ready skills", "NumPy code doesn't work", "PyTorch is faster to write"],
      correctIndex: 1,
      explanation: "You understand the math from NumPy. PyTorch adds autograd, GPU support, and ecosystem tools. The translation exercise deepens understanding and makes your knowledge practically applicable.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What makes long context (>128K tokens) challenging beyond just O(n²) compute?",
      options: ["GPUs can't store that many tokens", "Quality degrades: models struggle to retrieve and reason over information in the middle of very long contexts ('lost in the middle' problem)", "Tokenizers break", "Loss becomes NaN"],
      correctIndex: 1,
      explanation: "Even with efficient attention, models show U-shaped retrieval performance — good at beginning and end, poor in the middle. Positional encoding extrapolation also fails beyond trained lengths.",
      randomize: true
    },
    {
      id: "q16",
      type: "fill-blank",
      prompt: "The three main categories of efficient attention are: ___ attention (attend to subset), ___ attention (kernel trick replaces softmax), and ___ models (recurrent-like linear-time processing).",
      options: ["sparse, linear, state space", "dense, sparse, convolutional", "local, global, hybrid", "fast, slow, medium"],
      correctIndex: 0,
      explanation: "Sparse attention (Longformer), linear attention (kernel trick avoids n² matrix), and state space models (Mamba/S4) each attack the quadratic bottleneck differently.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is DPO (Direct Preference Optimization)?",
      options: ["A new optimizer like Adam", "An alignment method that directly optimizes on human preference pairs without training a separate reward model, simplifying RLHF", "A data preprocessing technique", "A dropout variant"],
      correctIndex: 1,
      explanation: "DPO skips the reward model + PPO pipeline of RLHF. It directly optimizes the policy on preferred vs rejected completions. Simpler, more stable, increasingly popular for alignment.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Why is reasoning still considered brittle in LLMs compared to humans?",
      options: ["LLMs can't do any reasoning", "LLMs pattern-match rather than truly reason — they fail on novel problems requiring systematic logic, even when similar problems were seen in training", "Reasoning requires GPUs", "Humans have special reasoning hardware"],
      correctIndex: 1,
      explanation: "LLMs excel at familiar reasoning patterns from training data but struggle with compositional novelty. Chain-of-thought helps but doesn't solve fundamental brittleness on out-of-distribution logic.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What's the value of understanding neural networks from first principles (as this curriculum teaches)?",
      options: ["It's slower than using libraries", "Frameworks change and architectures evolve, but mathematical fundamentals are permanent — first-principles understanding lets you adapt to any new development", "First principles aren't needed for applied ML", "It replaces the need for frameworks"],
      correctIndex: 1,
      explanation: "When a new architecture drops, someone who understands gradients, attention, and optimization from scratch can read the paper and implement it. Someone who only knows API calls must wait for library support.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which combination correctly matches the frontier area with its challenge?",
      options: [
        "Multimodal → O(n²) attention",
        "Agents → reliable planning and self-correction in multi-step tasks",
        "Efficient architectures → hallucination",
        "Alignment → KV cache compression"
      ],
      correctIndex: 1,
      explanation: "Agent reliability (planning, error recovery, tool selection) is the core open challenge. Multimodal faces alignment across modalities. Efficient arch faces compute/quality tradeoffs. Alignment faces value specification.",
      randomize: true
    }
  ]}
/>
