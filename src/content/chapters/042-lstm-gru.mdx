---
title: "LSTM & GRU"
slug: "042-lstm-gru"
description: "Solve the vanishing gradient problem with gating mechanisms. Implement LSTM and GRU cells from scratch in NumPy with full forward and backward passes."
track: "nn-intermediate"
order: 7
read_time: 30
code_time: 25
execution_timeout: 15
prerequisites: ["041-rnn-from-scratch"]
---

# LSTM & GRU

Vanilla RNNs forget everything after ~20 steps. ==LSTMs and GRUs use gates to control information flow==, enabling learning over hundreds of timesteps.

## LSTM Cell

Three gates control a separate memory cell:

- **Forget gate** <InlineMath latex="f_t" />: What to discard from memory
- **Input gate** <InlineMath latex="i_t" />: What new info to store
- **Output gate** <InlineMath latex="o_t" />: What to output from memory

<BlockMath latex="f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)" />
<BlockMath latex="i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)" />
<BlockMath latex="\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)" />
<BlockMath latex="C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t" />
<BlockMath latex="o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)" />
<BlockMath latex="h_t = o_t \odot \tanh(C_t)" />

```python
import numpy as np

def lstm_forward(X, h0, c0, params):
    """X: (T, N, D), returns H (T, N, H)"""
    T, N, D = X.shape
    H = params["Wf"].shape[1]
    
    Wf, Wi, Wc, Wo = params["Wf"], params["Wi"], params["Wc"], params["Wo"]
    bf, bi, bc, bo = params["bf"], params["bi"], params["bc"], params["bo"]
    
    H_states = np.zeros((T, N, H))
    C_states = np.zeros((T + 1, N, H))
    C_states[0] = c0
    gates_cache = []
    
    h_prev = h0
    for t in range(T):
        hx = np.concatenate([h_prev, X[t]], axis=1)  # (N, H+D)
        
        f = sigmoid(hx @ Wf + bf)
        i = sigmoid(hx @ Wi + bi)
        c_tilde = np.tanh(hx @ Wc + bc)
        o = sigmoid(hx @ Wo + bo)
        
        C_states[t+1] = f * C_states[t] + i * c_tilde
        H_states[t] = o * np.tanh(C_states[t+1])
        
        gates_cache.append((hx, f, i, c_tilde, o, C_states[t+1]))
        h_prev = H_states[t]
    
    return H_states, C_states, gates_cache
```

<PyRunner
  cellId="042-cell-1"
  defaultCode={`import numpy as np
np.random.seed(42)

def sigmoid(z): return 1/(1+np.exp(-z))

T, N, D, H = 10, 2, 4, 8
X = np.random.randn(T, N, D) * 0.5
h0, c0 = np.zeros((N,H)), np.zeros((N,H))

# Concatenated weight matrices
HD = H + D
Wf = np.random.randn(HD, H)*0.3; bf = np.ones(H)*2  # forget bias=2 → remember by default
Wi = np.random.randn(HD, H)*0.3; bi = np.zeros(H)
Wc = np.random.randn(HD, H)*0.3; bc = np.zeros(H)
Wo = np.random.randn(HD, H)*0.3; bo = np.zeros(H)

h, c = h0.copy(), c0.copy()
print("LSTM State Evolution:")
print(f"{'t':>3} | {'h_mean':>7} | {'h_std':>7} | {'c_mean':>7} | {'c_std':>7}")
print("─" * 42)
for t in range(T):
    hx = np.concatenate([h, X[t]], axis=1)
    f = sigmoid(hx@Wf+bf); i = sigmoid(hx@Wi+bi)
    ct = np.tanh(hx@Wc+bc); o = sigmoid(hx@Wo+bo)
    c = f*c + i*ct
    h = o*np.tanh(c)
    print(f"{t:3d} | {h.mean():+7.4f} | {h.std():7.4f} | {c.mean():+7.4f} | {c.std():7.4f}")

print(f"\n💡 Cell state (c) can maintain information across many steps")
print(f"   Forget gate bias=2 means 'remember by default'")
print(f"   This is why LSTMs solve the vanishing gradient problem!")
`}
/>

### Why LSTM Solves Vanishing Gradients

The key: <InlineMath latex="C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t" />

When <InlineMath latex="f_t \approx 1" />, the gradient flows through <InlineMath latex="C" /> unchanged — like a highway. No repeated multiplication by small derivatives. The forget gate LEARNS when to preserve gradients.

> [!IMPORTANT] Forget Gate Initialization
> Initialize forget gate bias to 1 or 2. This makes the LSTM start by remembering, not forgetting. Without this, early training discards information before learning what matters.

## GRU: Simpler Alternative

Two gates instead of three, no separate cell state:

<BlockMath latex="r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)" />
<BlockMath latex="z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)" />
<BlockMath latex="\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)" />
<BlockMath latex="h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t" />

```python
def gru_cell(x, h_prev, params):
    hx = np.concatenate([h_prev, x], axis=1)
    r = sigmoid(hx @ params["Wr"] + params["br"])
    z = sigmoid(hx @ params["Wz"] + params["bz"])
    
    rh = np.concatenate([r * h_prev, x], axis=1)
    h_tilde = np.tanh(rh @ params["Wh"] + params["bh"])
    
    h_new = (1 - z) * h_prev + z * h_tilde
    return h_new
```

<PyRunner
  cellId="042-cell-2"
  defaultCode={`import numpy as np
np.random.seed(42)

def sigmoid(z): return 1/(1+np.exp(-z))

H, D = 8, 4
HD = H + D

# GRU parameters
Wr = np.random.randn(HD, H)*0.3; br = np.zeros(H)
Wz = np.random.randn(HD, H)*0.3; bz = np.ones(H)  # update bias=1 → keep old state
Wh = np.random.randn(HD, H)*0.3; bh = np.zeros(H)

h = np.zeros((1, H))
x_seq = np.random.randn(10, 1, D) * 0.5

print("GRU Update Gate (z) values over time:")
print("z≈0 → keep old state, z≈1 → use new candidate")
print(f"{'t':>3} | {'z_mean':>7} | {'r_mean':>7} | {'h_std':>7}")
print("─" * 32)
for t in range(10):
    hx = np.concatenate([h, x_seq[t]], axis=1)
    r = sigmoid(hx@Wr+br)
    z = sigmoid(hx@Wz+bz)
    rh = np.concatenate([r*h, x_seq[t]], axis=1)
    ht = np.tanh(rh@Wh+bh)
    h = (1-z)*h + z*ht
    print(f"{t:3d} | {z.mean():7.4f} | {r.mean():7.4f} | {h.std():7.4f}")

print(f"\n💡 GRU has ~25% fewer params than LSTM")
print(f"   Performance is often comparable")
print(f"   Choose GRU for efficiency, LSTM for complex sequences")
`}
/>

## LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Separate <InlineMath latex="C_t" /> | None (uses <InlineMath latex="h_t" /> directly) |
| Parameters | ~4× more than vanilla RNN | ~3× more than vanilla RNN |
| Long-range memory | Excellent | Very good |
| Training speed | Slower | Faster |
| When to use | Complex sequences, language modeling | Default choice, smaller datasets |

> [!NOTE] Practical Advice
> Start with GRU. Switch to LSTM if you need better long-range dependencies or your task involves very long sequences (>200 timesteps). Both massively outperform vanilla RNNs.

<Quiz
  chapterSlug="042-lstm-gru"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "How does the LSTM cell state solve the vanishing gradient problem?",
      options: [
        "It uses ReLU instead of tanh",
        "When forget gate ≈ 1, gradients flow through C unchanged via addition (not multiplication), creating a gradient highway",
        "It normalizes gradients at each step",
        "It uses skip connections"
      ],
      correctIndex: 1,
      explanation: "C_t = f⊙C_{t-1} + i⊙C̃. When f≈1, ∂C_t/∂C_{t-1} ≈ 1. The gradient passes through addition, not multiplication by small derivatives. This preserves gradient magnitude over hundreds of timesteps.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why initialize the LSTM forget gate bias to a positive value (1-2)?",
      options: [
        "To make training faster",
        "So the LSTM starts by remembering (f≈sigmoid(2)≈0.88) rather than forgetting, preserving information until it learns what to discard",
        "To prevent overflow",
        "It doesn't matter"
      ],
      correctIndex: 1,
      explanation: "With bias=0, f=sigmoid(0)=0.5 — the LSTM forgets half its memory every step initially. With bias=2, f≈0.88, so it retains most information and gradually learns selective forgetting.",
      randomize: false,
    }
  ]}
/>
