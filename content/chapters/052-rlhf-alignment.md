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
      options: [
        "SFT models are too small",
        "SFT teaches format and knowledge but not nuanced human preferences like helpfulness, honesty, and harmlessness",
        "SFT does not use enough data",
        "SFT cannot handle long sequences"
      ],
      correctIndex: 1,
      explanation: "SFT shows the model WHAT to generate but not WHICH generation is better when multiple valid responses exist. Preferences require comparative signal that single-target training cannot provide.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What problem does the KL penalty in RLHF prevent?",
      options: [
        "Overfitting to training data",
        "Reward hacking: the model exploits reward model artifacts instead of genuinely improving quality",
        "Vanishing gradients",
        "Slow convergence"
      ],
      correctIndex: 1,
      explanation: "Reward models are imperfect proxies. Without KL constraint, the policy finds adversarial inputs that score high on the RM but produce terrible actual outputs. KL anchors the policy near the SFT model.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What are the four steps of the RLHF pipeline?",
      options: [
        "Pretrain, fine-tune, evaluate, deploy",
        "SFT model generates candidates, humans rank them, train reward model on rankings, optimize policy via PPO against reward model",
        "Collect data, train model, test, iterate",
        "Encode, decode, attend, generate"
      ],
      correctIndex: 1,
      explanation: "RLHF is a four-stage process: start with SFT model, collect human preference data, learn a reward function from preferences, then use RL (PPO) to optimize the policy against that reward function.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "How does the reward model learn from human preferences?",
      options: [
        "Humans assign absolute scores to each response",
        "Humans compare pairs of responses (chosen vs rejected). The reward model learns via Bradley-Terry pairwise loss to predict which response humans prefer.",
        "The reward model is pretrained on Wikipedia",
        "Rewards are computed automatically from metrics"
      ],
      correctIndex: 1,
      explanation: "Pairwise comparison is easier and more reliable than absolute scoring. Loss = -log(sigmoid(r_chosen - r_rejected)). The RM learns to assign higher scores to preferred responses.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "If r_chosen=2.0 and r_rejected=0.5, what is the Bradley-Terry loss?",
      code: "import numpy as np\ndef sigmoid(x): return 1/(1+np.exp(-x))\nr_c, r_r = 2.0, 0.5\nloss = -np.log(sigmoid(r_c - r_r))\nprint(f'{loss:.4f}')",
      options: ["0.2014", "0.6931", "1.5000", "0.0000"],
      correctIndex: 0,
      explanation: "sigmoid(2.0 - 0.5) = sigmoid(1.5) = 0.8176. -log(0.8176) = 0.2014. Lower loss means the RM correctly separates chosen from rejected. As the gap increases, loss approaches 0.",
      randomize: true
    },
    {
      id: "q6",
      type: "fill-blank",
      prompt: "DPO eliminates the need for a separate ___ model by optimizing directly on ___ pairs.",
      options: ["reward, preference", "language, training", "policy, evaluation", "tokenization, input"],
      correctIndex: 0,
      explanation: "DPO derives a closed-form solution that optimizes the policy directly from preference pairs without training an intermediate reward model. Simpler pipeline, fewer failure modes, comparable quality.",
      randomize: false
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is reward hacking and why is it dangerous?",
      options: [
        "When the reward model is hacked by attackers",
        "The policy exploits imperfections in the reward model to achieve high scores without actually producing better outputs. Examples: overly verbose responses, repetitive praise, gaming specific phrases.",
        "When rewards become negative",
        "When the model refuses to generate any output"
      ],
      correctIndex: 1,
      explanation: "Reward models are learned approximations of human preference. They have blind spots the policy can exploit. This is why the KL penalty exists: it constrains the policy to stay near the SFT baseline where the RM is more reliable.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "Why is DPO increasingly preferred over PPO-based RLHF?",
      options: [
        "DPO produces better results",
        "DPO is simpler (no separate reward model, no RL training loop), more stable (no PPO hyperparameter sensitivity), and achieves near-RLHF quality with less engineering complexity.",
        "DPO uses less data",
        "PPO is deprecated"
      ],
      correctIndex: 1,
      explanation: "PPO requires careful tuning of clipping range, KL coefficient, value function, and rollout parameters. DPO reduces alignment to a simple classification-like loss on preference pairs. Easier to implement and debug.",
      randomize: true
    },
    {
      id: "q9",
      type: "ordering",
      prompt: "Order the complete LLM training pipeline from raw text to aligned assistant:",
      items: ["Pretraining on massive unlabeled text", "Supervised fine-tuning on instruction data", "Preference data collection (human rankings)", "Alignment (RLHF or DPO)"],
      correctOrder: [0, 1, 2, 3],
      explanation: "Knowledge (pretrain) then behavior (SFT) then preferences (collect) then alignment (optimize). Each stage builds on the previous. Skipping any stage produces inferior results.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What types of problems persist after SFT that alignment addresses?",
      options: [
        "Grammar errors and typos",
        "Hallucination, sycophancy, harmful compliance, verbosity without substance. These are preference issues, not knowledge or format issues.",
        "Slow inference speed",
        "Small vocabulary"
      ],
      correctIndex: 1,
      explanation: "SFT teaches the model to respond in the right format. Alignment teaches it WHICH responses humans actually prefer. A fluent, well-formatted hallucination is still wrong; alignment reduces these failure modes.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "In the PPO objective, what does the beta coefficient control?",
      options: [
        "Learning rate",
        "Strength of the KL penalty: higher beta keeps the policy closer to SFT (safer but less aligned), lower beta allows more deviation (better alignment but riskier reward hacking).",
        "Batch size",
        "Number of PPO epochs"
      ],
      correctIndex: 1,
      explanation: "L_PPO = E[R(x,y) - beta * KL(policy || SFT)]. Beta trades off reward maximization against staying close to the SFT baseline. Tuning beta is critical: too low enables hacking, too high prevents improvement.",
      randomize: true
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What does sigmoid(0) equal in the Bradley-Terry model?",
      code: "import numpy as np\nprint(f'{1/(1+np.exp(0)):.4f}')",
      options: ["0.5000", "1.0000", "0.0000", "0.7311"],
      correctIndex: 0,
      explanation: "sigmoid(0) = 0.5. When r_chosen = r_rejected, the model is maximally uncertain which is preferred. Loss = -log(0.5) = 0.693, the maximum pairwise loss.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What is RLAIF and how does it differ from RLHF?",
      options: [
        "They are identical",
        "RLAIF uses AI-generated preferences instead of human ratings. A strong model judges response pairs, reducing cost and scaling preference data collection.",
        "RLAIF uses reinforcement learning only",
        "RLAIF replaces SFT entirely"
      ],
      correctIndex: 1,
      explanation: "Human rating is expensive and slow. RLAIF leverages capable models (e.g., GPT-4) as preference judges. Quality depends on the judge model's alignment. Useful for scaling beyond human annotation capacity.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Your aligned model refuses to answer legitimate technical questions about security. What went wrong?",
      options: [
        "The model is working correctly",
        "Over-alignment: the safety training was too aggressive, causing excessive refusals. The model conflates legitimate technical discussion with harmful intent.",
        "The reward model was perfect",
        "Security topics should always be refused"
      ],
      correctIndex: 1,
      explanation: "Alignment must balance helpfulness and harmlessness. Over-indexing on safety creates an unhelpful model. Solutions: more nuanced preference data, better reward model calibration, adjusting the safety-helpfulness tradeoff.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "The KL divergence in RLHF measures how far the ___ has drifted from the ___ model. Keeping this small prevents reward hacking.",
      options: ["policy, SFT", "reward, base", "encoder, decoder", "model, data"],
      correctIndex: 0,
      explanation: "KL(pi_theta || pi_SFT) penalizes the current policy for deviating from the SFT baseline. The SFT model represents safe, reasonable outputs. Staying nearby ensures the reward model's assessments remain reliable.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why do preference datasets use pairwise comparisons instead of absolute scores?",
      options: [
        "Absolute scores are impossible to collect",
        "Comparing two responses is cognitively easier and more consistent for humans than assigning absolute quality scores on a scale. Relative judgments have lower variance.",
        "Pairwise data is smaller",
        "Algorithms require pairwise format"
      ],
      correctIndex: 1,
      explanation: "Is this response a 7/10 or 8/10? Hard to say consistently. Is response A better than B? Much easier and more reliable. Bradley-Terry model converts these relative judgments into a scalar reward function.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What happens if you skip alignment entirely and deploy an SFT model?",
      options: [
        "Nothing, SFT is sufficient",
        "The model may follow harmful instructions, hallucinate confidently, be sycophantic, or produce verbose unhelpful responses. It knows language but lacks preference calibration.",
        "The model refuses all requests",
        "Performance improves without alignment overhead"
      ],
      correctIndex: 1,
      explanation: "SFT models are knowledgeable and fluent but lack the preference signal that makes them helpful, honest, and harmless. Base ChatGPT before RLHF was notorious for these issues despite strong SFT.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "How does DPO's loss function relate to the reward model conceptually?",
      options: [
        "DPO uses a completely different objective",
        "DPO implicitly defines a reward as log(pi/policy_ref), eliminating the explicit RM. The policy directly learns to increase probability of chosen responses relative to rejected ones.",
        "DPO trains a better reward model",
        "DPO and RLHF have unrelated losses"
      ],
      correctIndex: 1,
      explanation: "DPO derives from the insight that the optimal policy under a reward is proportional to exp(reward). By inverting this relationship, DPO optimizes the policy directly from preferences without ever materializing the reward function.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What is sycophancy in LLMs and why does alignment help reduce it?",
      options: [
        "Generating too many tokens",
        "Agreeing with incorrect user premises to appear helpful. Alignment trains the model to prioritize truthfulness over agreeableness when they conflict.",
        "Refusing all requests",
        "Generating code instead of prose"
      ],
      correctIndex: 1,
      explanation: "SFT models learn to please the user, sometimes at the expense of accuracy. Preference data where honest corrections are ranked above sycophantic agreement teaches the model that truthfulness is preferred over flattery.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You have limited budget for alignment. Should you invest in more preference data or better PPO tuning?",
      options: [
        "Better PPO tuning",
        "More preference data. Reward model quality is the bottleneck; a good RM with simple PPO outperforms a bad RM with perfect PPO tuning.",
        "They are equally important",
        "Neither matters, just use SFT"
      ],
      correctIndex: 1,
      explanation: "The reward model caps alignment quality. No amount of PPO tuning can compensate for a reward model that misrepresents human preferences. Invest first in diverse, high-quality preference annotations.",
      randomize: true
    }
  ]}
/>
