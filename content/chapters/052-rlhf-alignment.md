---
title: "RLHF & Alignment"
slug: "052-rlhf-alignment"
description: "How LLMs learn human preferences. Understand reward modeling, PPO, DPO, and why alignment is necessary beyond supervised fine-tuning."
track: "nn-advanced"
order: 7
read_time: 25
code_time: 15
execution_timeout: 10
prerequisites: ["051-pretraining-finetuning"]
---

# RLHF & Alignment

SFT teaches format. ==Alignment teaches preference.== A model can produce fluent, well-formatted responses that are still unhelpful, harmful, or dishonest. RLHF optimizes for human judgment.

## The Alignment Problem

After SFT, models can:
- Hallucinate confidently
- Follow harmful instructions
- Be sycophantic (agree with wrong premises)
- Produce verbose but unhelpful answers

These aren't knowledge gaps — they're ==preference gaps==. The model needs to learn what humans actually want.

## RLHF Pipeline

```
1. SFT Model → generates candidate responses
2. Human raters rank responses → preference dataset
3. Train Reward Model on rankings
4. Optimize policy (SFT model) against reward model via PPO
```

### Step 1: Reward Model

Trained to predict human preference scores:

<BlockMath latex="\mathcal{L}_{RM} = -\log\sigma(r(x, y_w) - r(x, y_l))" />

where <InlineMath latex="y_w" /> is the preferred response and <InlineMath latex="y_l" /> is the rejected one.

```python
def reward_model_loss(reward_fn, prompt, chosen, rejected):
    """Bradley-Terry pairwise loss."""
    r_chosen = reward_fn(prompt, chosen)
    r_rejected = reward_fn(prompt, rejected)
    return -np.log(sigmoid(r_chosen - r_rejected))
```

<PyRunner
  cellId="052-cell-1"
  defaultCode={`import numpy as np

def sigmoid(x): return 1/(1+np.exp(-x))

# Simulate reward model training
np.random.seed(42)
pairs = [
    ("Explain quantum physics", "Detailed accurate explanation", "Quantum physics is magic lol"),
    ("Write a poem", "Beautiful sonnet about stars", "Roses are red violets are blue"),
    ("Debug this code", "Found bug on line 5, here's fix", "Looks fine to me"),
]

print("Reward Model Training Data:")
print(f"{'Prompt':<25} | {'Chosen':<30} | {'Rejected':<30}")
print("─" * 90)
for p, c, r in pairs:
    print(f"{p:<25} | {c:<30} | {r:<30}")

# Simulate learned rewards
r_chosen = np.array([2.1, 1.8, 2.5])
r_rejected = np.array([-0.5, 0.2, -1.0])
loss = -np.mean(np.log(sigmoid(r_chosen - r_rejected)))

print(f"\nPairwise loss: {loss:.4f}")
print(f"✅ RM learns to score responses like human raters")
`}
/>

### Step 2: PPO Optimization

Optimize the policy to maximize reward while staying close to the SFT model:

<BlockMath latex="\mathcal{L}_{PPO} = \mathbb{E}\left[R(x,y) - \beta \cdot KL(\pi_\theta || \pi_{SFT})\right]" />

The KL penalty prevents the model from drifting too far from the SFT baseline (reward hacking).

> [!IMPORTANT] Why KL Penalty?
> Without it, the model exploits reward model flaws: overly long responses, repetitive praise, gaming specific phrases. KL keeps outputs natural.

## DPO: Direct Preference Optimization

Skip the reward model entirely. Optimize directly on preference pairs:

<BlockMath latex="\mathcal{L}_{DPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)" />

Simpler, more stable, no separate reward model needed. Increasingly preferred over PPO.

## Alignment Taxonomy

| Method | Complexity | Stability | Quality |
|--------|-----------|-----------|---------|
| SFT only | Low | High | Baseline |
| RLHF (PPO) | High | Medium | Best (with good RM) |
| DPO | Medium | High | Near-RLHF |
| RLAIF | Medium | High | Good (AI-generated preferences) |

<Quiz
  chapterSlug="052-rlhf-alignment"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is SFT insufficient for alignment?",
      options: ["SFT models are too small", "SFT teaches format and knowledge but not nuanced human preferences like helpfulness, honesty, and harmlessness", "SFT doesn't use enough data", "SFT can't handle long sequences"],
      correctIndex: 1,
      explanation: "SFT shows the model WHAT to generate but not WHICH generation is better when multiple valid responses exist. Preferences require comparative signal that SFT's single-target training cannot provide.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What problem does the KL penalty in RLHF prevent?",
      options: ["Overfitting", "Reward hacking: the model exploits reward model artifacts instead of genuinely improving quality", "Vanishing gradients", "Slow convergence"],
      correctIndex: 1,
      explanation: "Reward models are imperfect proxies. Without KL constraint, the policy finds adversarial inputs that score high on the RM but produce terrible actual outputs. KL anchors the policy near the SFT model.",
      randomize: false,
    }
  ]}
/>
