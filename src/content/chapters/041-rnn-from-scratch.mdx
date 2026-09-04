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
      explanation: "Each backward step multiplies by ∂tanh/∂z × Whh. Since |tanh'| ≤ 1 and typically ||Whh|| < 1, the product decays exponentially. After ~20 steps, the gradient is essentially zero.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is weight sharing across time in RNNs analogous to?",
      options: [
        "Batch normalization across samples",
        "Weight sharing across spatial positions in CNNs — same parameters applied at every position/timestep",
        "Dropout across layers",
        "Residual connections"
      ],
      correctIndex: 1,
      explanation: "Just as CNN kernels detect features anywhere in the image, RNN weights process information the same way at every timestep. This enables generalization across sequence positions.",
      randomize: false,
    }
  ]}
/>
