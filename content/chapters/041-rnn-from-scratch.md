---
title: "Recurrent Neural Networks from Scratch"
slug: "041-rnn-from-scratch"
description: "Process sequences by maintaining hidden state across timesteps. Implement vanilla RNN forward and backward (BPTT) in NumPy, understand vanishing gradients."
track: "nn-intermediate"
order: 6
read_time: 28
code_time: 22
execution_timeout: 15
prerequisites: ["030-backpropagation", "028-activation-functions"]
---

# Recurrent Neural Networks from Scratch

CNNs process spatial structure. ==RNNs process temporal structure.== They maintain a hidden state that summarizes everything seen so far, updating it at each timestep.

## The RNN Cell

At each timestep <InlineMath latex="t" />:

<BlockMath latex="h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)" />

Same weights <InlineMath latex="W_{xh}" />, <InlineMath latex="W_{hh}" /> applied at every timestep. This is weight sharing across time, analogous to CNN weight sharing across space.

```python
import numpy as np

def rnn_forward(X, h0, Wxh, Whh, bh):
    """X: (T, N, D), h0: (N, H)
    Returns: H (T, N, H), final_h (N, H)
    """
    T, N, D = X.shape
    H = Whh.shape[1]
    
    states = np.zeros((T + 1, N, H))
    states[0] = h0
    
    for t in range(T):
        states[t+1] = np.tanh(
            X[t] @ Wxh + states[t] @ Whh + bh
        )
    
    return states[1:], states[-1]
```

<PyRunner
  cellId="041-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

T, N, D, H = 5, 2, 4, 8
X = np.random.randn(T, N, D) * 0.5
h0 = np.zeros((N, H))
Wxh = np.random.randn(D, H) * np.sqrt(2.0/(D+H))
Whh = np.random.randn(H, H) * np.sqrt(2.0/(D+H))
bh = np.zeros(H)

states = [h0.copy()]
h = h0
for t in range(T):
    h = np.tanh(X[t] @ Wxh + h @ Whh + bh)
    states.append(h.copy())

print("RNN Hidden State Evolution:")
print(f"{'t':>3} | {'mean':>7} | {'std':>7} | {'min':>7} | {'max':>7}")
print("─" * 40)
for t, s in enumerate(states):
    print(f"{t:3d} | {s.mean():+7.4f} | {s.std():7.4f} | {s.min():+7.4f} | {s.max():+7.4f}")

print(f"\n💡 Hidden state evolves over time, accumulating information")
print(f"   tanh keeps values in [-1, 1] but causes vanishing gradients")
`}
/>

## Backpropagation Through Time (BPTT)

The tricky part. Gradients flow backward through EVERY timestep:

<BlockMath latex="\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}" />

But <InlineMath latex="\frac{\partial h_t}{\partial h_{t-1}}" /> involves the tanh derivative AND <InlineMath latex="W_{hh}" />, multiplied T times. This product either explodes or vanishes.

```python
def rnn_backward(dstates_next, X, states, Wxh, Whh):
    """BPTT for vanilla RNN.
    dstates_next: (T, N, H) gradient from output layer
    states: (T+1, N, H) cached forward states
    """
    T, N, D = X.shape
    H = Whh.shape[1]
    
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros(H)
    dX = np.zeros_like(X)
    dh_next = np.zeros((N, H))
    
    for t in reversed(range(T)):
        # Total gradient at this timestep
        dh = dstates_next[t] + dh_next
        
        # Through tanh
        dtanh = (1 - states[t+1] ** 2) * dh
        
        # Parameter gradients
        dWxh += X[t].T @ dtanh
        dWhh += states[t].T @ dtanh
        dbh += np.sum(dtanh, axis=0)
        
        # Propagate to previous timestep
        dX[t] = dtanh @ Wxh.T
        dh_next = dtanh @ Whh.T
    
    return dX, dWxh, dWhh, dbh
```

<PyRunner
  cellId="041-cell-2"
  defaultCode={`import numpy as np

# Demonstrate vanishing gradient problem
np.random.seed(42)
H = 64
Whh = np.random.randn(H, H) * 0.9 / np.sqrt(H)

# Track gradient magnitude through time
grad_norms = []
dh = np.ones((1, H))  # unit gradient at final timestep

for t in range(20):
    # tanh derivative ≈ (1 - h²), assume h≈0.5 → deriv≈0.75
    dtanh_factor = 0.75
    dh = dtanh_factor * (dh @ Whh.T)
    grad_norms.append(np.linalg.norm(dh))

print("Gradient norm through time (vanishing):")
print(f"{'t back':>6} | {'||grad||':>10} | Visual")
print("─" * 40)
for t, g in enumerate(grad_norms):
    bar = "█" * int(g / grad_norms[0] * 30)
    if t % 3 == 0:
        print(f"{t+1:6d} | {g:10.6f} | {bar}")

print(f"\n⚠️ Gradient decays exponentially!")
print(f"   After 20 steps: {grad_norms[-1]/grad_norms[0]:.2e} of original")
print(f"   This is why vanilla RNNs can't learn long-range dependencies")
`}
/>

> [!IMPORTANT] The Vanishing Gradient Problem
> With tanh derivative ≤ 1 and <InlineMath latex="\|W_{hh}\| < 1" />, the gradient product shrinks exponentially. After ~10-20 timesteps, the signal is gone. This is why LSTM and GRU exist.

## Simple Character-Level Language Model

Despite limitations, vanilla RNNs can learn short-range patterns:

<PyRunner
  cellId="041-cell-3"
  defaultCode={`import numpy as np

# Tiny char-level model demo
np.random.seed(42)
text = "hello world hello world hello"
vocab = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
V = len(vocab)
H = 32
seq_len = 10

print(f"Vocabulary: {vocab}")
print(f"Vocab size: {V}, Hidden: {H}")
print(f"Training on '{text}' with seq_len={seq_len}")

# Initialize
Wxh = np.random.randn(V, H) * 0.1
Whh = np.random.randn(H, H) * 0.1
Why = np.random.randn(H, V) * 0.1
bh, by = np.zeros(H), np.zeros(V)

# Train for a few iterations
lr = 0.1
for epoch in range(50):
    h = np.zeros((1, H))
    loss = 0
    
    for i in range(len(text) - 1):
        x_idx = char_to_idx[text[i]]
        y_idx = char_to_idx[text[i+1]]
        
        # One-hot input
        x = np.zeros((1, V)); x[0, x_idx] = 1
        
        # Forward
        h = np.tanh(x @ Wxh + h @ Whh + bh)
        logits = h @ Why + by
        probs = np.exp(logits) / np.exp(logits).sum()
        
        loss -= np.log(probs[0, y_idx] + 1e-8)
        
        # Backward (simplified single-step)
        dy = probs.copy(); dy[0, y_idx] -= 1
        dWhy = h.T @ dy; dby = dy.sum(axis=0)
        dh = dy @ Why.T * (1 - h**2)
        dWxh = x.T @ dh; dWhh_prev = h @ dh  # simplified
        dbh = dh.sum(axis=0)
        
        # Clip gradients
        for d in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(d, -5, 5, out=d)
        
        Wxh -= lr * dWxh; Whh -= lr * dWhh
        Why -= lr * dWhy; bh -= lr * dbh; by -= lr * dby
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: loss={loss/(len(text)-1):.3f}")

# Generate
h = np.zeros((1, H))
x_idx = char_to_idx['h']
generated = 'h'
for _ in range(25):
    x = np.zeros((1, V)); x[0, x_idx] = 1
    h = np.tanh(x @ Wxh + h @ Whh + bh)
    logits = h @ Why + by
    probs = np.exp(logits) / np.exp(logits).sum()
    x_idx = np.random.choice(V, p=probs[0])
    generated += idx_to_char[x_idx]

print(f"\nGenerated: '{generated}'")
print(f"💡 Vanilla RNN learns short patterns but struggles with longer text")
`}
/>

<Quiz
  chapterSlug="041-rnn-from-scratch"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does BPTT cause vanishing gradients in vanilla RNNs?",
      options: [
        "The learning rate is too small",
        "Gradients are multiplied by tanh derivatives (≤1) and Whh at each timestep, causing exponential decay over long sequences",
        "BPTT uses too much memory",
        "The loss function is non-convex"
      ],
      correctIndex: 1,
      explanation: "Each backward step multiplies by ∂tanh/∂z × Whh. Since |tanh'| ≤ 1 and typically ||Whh|| < 1, the product decays exponentially. After ~20 steps, the gradient signal is essentially zero.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is weight sharing across time in RNNs analogous to?",
      options: [
        "Batch normalization across samples",
        "Weight sharing across spatial positions in CNNs — same parameters applied at every timestep, enabling generalization across sequence positions",
        "Dropout across layers",
        "Residual connections between layers"
      ],
      correctIndex: 1,
      explanation: "Just as CNN kernels detect features anywhere in the image, RNN weights (Wxh, Whh) process information identically at every timestep. This temporal weight sharing is what makes RNNs handle variable-length sequences.",
      randomize: false
    },
    {
      id: "q3",
      type: "shape-prediction",
      prompt: "If X has shape (T=10, N=32, D=64) and Wxh has shape (64, 128), what shape is the hidden state h_t at each timestep?",
      options: ["(32, 128)", "(10, 32, 128)", "(32, 64)", "(128,)"],
      correctIndex: 0,
      explanation: "h_t = tanh(X[t] @ Wxh + h_{t-1} @ Whh + bh). X[t] is (N, D)=(32, 64), Wxh is (D, H)=(64, 128). Result: (32, 128). Each sample in the batch gets its own H-dimensional hidden state.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "In rnn_forward, why store states as (T+1, N, H) with states[0] = h0?",
      options: [
        "To save memory by reusing the initial state",
        "states[t] holds h_{t-1}, so states[t+1] = h_t. This indexing makes the backward pass clean: dh_next uses states[t] directly without off-by-one errors.",
        "NumPy requires contiguous arrays",
        "To enable parallel computation"
      ],
      correctIndex: 1,
      explanation: "The (T+1) array stores h0 through hT. During BPTT, computing dWhh at timestep t needs h_{t-1} = states[t]. Without this offset, you'd constantly adjust indices, inviting bugs.",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What does the tanh derivative evaluate to when h = 0?",
      code: "import numpy as np\nh = 0.0\ndtanh = 1 - h**2\nprint(dtanh)",
      options: ["1.0", "0.0", "0.5", "-1.0"],
      correctIndex: 0,
      explanation: "d/dh tanh(h) = 1 - tanh²(h). When h=0, tanh(0)=0, so derivative = 1-0 = 1. This is the maximum tanh derivative. As |h|→1, derivative→0, which is where vanishing gradients originate.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "In rnn_backward, why is `dh = dstates_next[t] + dh_next` a sum rather than just dh_next?",
      options: [
        "It's a bug — should be subtraction",
        "Gradient flows from TWO sources: the output layer at timestep t (dstates_next[t]) AND the future timestep t+1 via Whh (dh_next). Chain rule requires summing all incoming gradient paths.",
        "Numerical stability requires averaging",
        "Only dh_next matters; dstates_next is ignored"
      ],
      correctIndex: 1,
      explanation: "h_t affects both the output at time t AND h_{t+1} (which affects all future outputs). BPTT accumulates gradients from both paths. At the final timestep, dh_next=0 since there's no future.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "The RNN update equation h_t = tanh(___ · x_t + ___ · h_{t-1} + b_h) uses two weight matrices that are ___ across all timesteps.",
      options: ["Wxh, Whh, shared", "Wxh, Why, different", "Whh, Why, shared", "Wxh, Whh, independent"],
      correctIndex: 0,
      explanation: "Wxh maps input to hidden, Whh maps previous hidden to current hidden. Both are reused at every timestep — this is temporal weight sharing, the defining characteristic of RNNs.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "Why does gradient clipping help with RNN training?",
      options: [
        "It prevents vanishing gradients",
        "It prevents exploding gradients by capping gradient magnitudes, allowing training to continue even when occasional large gradients occur",
        "It speeds up convergence",
        "It replaces the need for careful initialization"
      ],
      correctIndex: 1,
      explanation: "While vanishing gradients are the more common RNN problem, exploding gradients can also occur when ||Whh|| > 1. Gradient clipping (e.g., clip to [-5, 5]) caps extreme values, preventing NaN losses and training divergence.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "What is the computational complexity of BPTT for a sequence of length T?",
      options: ["O(T) per parameter update", "O(1) regardless of sequence length", "O(T²) due to pairwise interactions", "O(log T) with tree-based methods"],
      correctIndex: 0,
      explanation: "BPTT iterates backward through all T timesteps once, computing gradients at each step. Total work is proportional to T. Memory is also O(T) since all intermediate states must be cached.",
      randomize: true
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "After 10 backward steps with tanh derivative ≈ 0.75 and ||Whh|| ≈ 0.9, approximately what fraction of the original gradient remains?",
      code: "factor = 0.75 * 0.9\nremaining = factor ** 10\nprint(f'{remaining:.4f}')",
      options: ["0.0563", "0.7500", "0.0001", "0.5625"],
      correctIndex: 0,
      explanation: "Each step multiplies by ~0.75×0.9 = 0.675. After 10 steps: 0.675^10 ≈ 0.056. Only ~5.6% of the gradient survives. After 20 steps: ~0.003. This exponential decay is the vanishing gradient problem.",
      randomize: true
    },
    {
      id: "q11",
      type: "ordering",
      prompt: "Order the operations in one BPTT backward step at timestep t:",
      items: ["Sum gradients: dh = dstates_next[t] + dh_next", "Apply tanh derivative: dtanh = (1 - h²) × dh", "Accumulate parameter gradients: dWxh += X[t].T @ dtanh", "Propagate to previous timestep: dh_next = dtanh @ Whh.T"],
      correctOrder: [0, 1, 2, 3],
      explanation: "First combine gradient sources, then apply activation derivative, then compute parameter gradients, finally propagate backward. The order ensures dh_next is ready for the next (earlier) timestep.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why can't vanilla RNNs learn that 'the cat sat on the ___' should be 'mat' when the sentence is 50 words long?",
      options: [
        "The vocabulary doesn't contain 'mat'",
        "The gradient signal from the loss at position 50 has vanished by the time it reaches position 1 where 'cat' was encoded. The hidden state retains some information but the learning signal cannot flow back.",
        "RNNs can only process 10 timesteps",
        "tanh saturates after 5 characters"
      ],
      correctIndex: 1,
      explanation: "This is the long-range dependency problem. The forward pass preserves some information in h_t, but BPTT's gradient decays exponentially. The network can't learn to associate distant events because the training signal disappears.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "In the character-level language model, why use one-hot encoding for input characters?",
      options: [
        "One-hot is more memory efficient",
        "Characters are categorical with no ordinal relationship. One-hot provides an unbiased representation where each character is equidistant from all others in embedding space.",
        "RNNs require binary inputs",
        "It prevents gradient explosion"
      ],
      correctIndex: 1,
      explanation: "Character 'a' isn't numerically closer to 'b' than to 'z'. Integer encoding would imply false ordinal structure. One-hot treats all characters as equally distinct, letting the network learn meaningful representations via Wxh.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What happens if you initialize Whh with large values (e.g., std=1.0 instead of 0.01)?",
      options: [
        "Training converges faster",
        "Hidden states quickly saturate tanh to ±1, making derivatives near zero everywhere. Gradients vanish immediately and the network becomes untrainable.",
        "The model memorizes the training data",
        "Nothing changes — tanh handles any scale"
      ],
      correctIndex: 1,
      explanation: "Large Whh → large pre-activations → tanh saturates at ±1 → derivative ≈ 0 → no gradient flow. Small initialization (std ≈ 0.01-0.1) keeps pre-activations in tanh's linear region where gradients are healthy.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "During text generation, the RNN samples the next character from the ___ distribution produced by softmax over logits.",
      options: ["uniform", "probability", "Gaussian", "deterministic"],
      correctIndex: 1,
      explanation: "The output layer produces logits → softmax → probability distribution over vocabulary. Sampling (not argmax) introduces diversity in generated text. Argmax would always pick the most likely character, producing repetitive output.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why does the backward loop iterate `for t in reversed(range(T))` instead of forward?",
      options: [
        "Python requires reverse iteration for numerical stability",
        "BPTT follows the chain rule backward: gradient at t depends on gradient at t+1 (dh_next). You must start from the last timestep where dh_next=0 and propagate backward.",
        "Forward iteration causes memory errors",
        "It's arbitrary — either direction works"
      ],
      correctIndex: 1,
      explanation: "The chain rule flows from output to input. dh_next at timestep t comes from timestep t+1's computation. Starting at t=T-1 with dh_next=0, each step computes and passes dh_next to the previous timestep.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "How does truncated BPTT address the vanishing gradient problem?",
      options: [
        "It uses a different activation function",
        "It limits backward propagation to K timesteps (e.g., K=20) instead of the full sequence, trading long-range learning for tractable gradients and lower memory",
        "It increases the learning rate",
        "It adds skip connections"
      ],
      correctIndex: 1,
      explanation: "Truncated BPTT stops gradient flow after K steps, preventing exponential decay over very long sequences. The network still processes the full sequence forward but only learns from recent context. This is a practical compromise.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What shape does dWxh have after accumulating gradients across all T timesteps?",
      code: "D, H = 64, 128\ndWxh_shape = (D, H)\nprint(dWxh_shape)",
      options: ["(64, 128)", "(10, 64, 128)", "(128, 64)", "(64,)"],
      correctIndex: 0,
      explanation: "dWxh has the same shape as Wxh: (D, H). Gradients from all T timesteps are summed (+=) into this single matrix, reflecting weight sharing — one set of parameters used at every timestep.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What fundamental architectural change do LSTM/GRU make to solve the vanishing gradient problem?",
      options: [
        "They use ReLU instead of tanh",
        "They introduce gated pathways that allow gradients to flow through additive connections (cell state / carry path) without being repeatedly multiplied by activation derivatives",
        "They remove recurrence entirely",
        "They use larger hidden states"
      ],
      correctIndex: 1,
      explanation: "LSTM's cell state acts as an information highway with additive updates (c_t = f·c_{t-1} + i·g). The forget gate f controls retention, but the additive structure means gradients can flow unchanged through many timesteps, avoiding multiplicative decay.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You train a char-level RNN and loss plateaus at 2.5 (random chance for V=12 would be log(12)≈2.48). What does this indicate?",
      options: [
        "The model is perfectly trained",
        "The model learned nothing — it's predicting near-uniform probabilities. Likely causes: learning rate too high/low, poor initialization, or vanishing gradients preventing any learning.",
        "The vocabulary is too large",
        "This is expected behavior for RNNs"
      ],
      correctIndex: 1,
      explanation: "Loss ≈ log(V) means the model outputs roughly uniform probabilities — equivalent to random guessing. The network failed to learn any patterns. Check initialization scale, learning rate, gradient norms, and consider gradient clipping.",
      randomize: true
    }
  ]}
/>
