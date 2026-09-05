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
        "When forget gate ≈ 1, gradients flow through C unchanged via addition (not multiplication), creating a gradient highway across many timesteps",
        "It normalizes gradients at each step",
        "It uses residual skip connections"
      ],
      correctIndex: 1,
      explanation: "C_t = f⊙C_{t-1} + i⊙C̃. When f≈1, ∂C_t/∂C_{t-1} ≈ 1. The gradient passes through addition, not multiplication by small derivatives. This preserves gradient magnitude over hundreds of timesteps.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why initialize the LSTM forget gate bias to a positive value (1-2)?",
      options: [
        "To make training converge faster",
        "So the LSTM starts by remembering (f≈sigmoid(2)≈0.88) rather than forgetting, preserving information until it learns what to selectively discard",
        "To prevent numerical overflow",
        "It doesn't matter — any initialization works"
      ],
      correctIndex: 1,
      explanation: "With bias=0, f=sigmoid(0)=0.5 — the LSTM forgets half its memory every step initially. With bias=2, f≈0.88, so it retains most information and gradually learns selective forgetting.",
      randomize: false
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What are the three gates in an LSTM cell?",
      options: [
        "Input, output, reset",
        "Forget (what to discard from memory), input (what new info to store), output (what to expose from memory)",
        "Update, reset, candidate",
        "Encode, decode, attention"
      ],
      correctIndex: 1,
      explanation: "Forget gate controls retention of old memory, input gate controls writing of new information, output gate controls what the hidden state reveals. Each is a sigmoid producing values in [0,1].",
      randomize: true
    },
    {
      id: "q4",
      type: "shape-prediction",
      prompt: "If h_prev has shape (N, H=128) and X[t] has shape (N, D=64), what is the shape of hx = concat([h_prev, X[t]])?",
      options: ["(N, 192)", "(N, 128)", "(2N, 128)", "(N, 64)"],
      correctIndex: 0,
      explanation: "Concatenating along axis=1: (N, 128) + (N, 64) → (N, 192). All four gates use this concatenated vector as input, so each gate weight matrix has shape (H+D, H) = (192, 128).",
      randomize: true
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What does sigmoid(2) evaluate to approximately?",
      code: "import numpy as np\nprint(f'{1/(1+np.exp(-2)):.4f}')",
      options: ["0.8808", "0.5000", "0.7311", "1.0000"],
      correctIndex: 0,
      explanation: "sigmoid(2) = 1/(1+e^(-2)) ≈ 0.8808. This is why forget gate bias=2 makes the LSTM 'remember by default' — the forget gate outputs ~0.88, retaining most of the previous cell state.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "In the LSTM update C_t = f ⊙ C_{t-1} + i ⊙ C̃, what operation does ⊙ represent?",
      options: [
        "Matrix multiplication",
        "Element-wise (Hadamard) product — each dimension of the gate independently scales the corresponding dimension of the cell state",
        "Dot product",
        "Outer product"
      ],
      correctIndex: 1,
      explanation: "Gates produce per-dimension values in [0,1]. Element-wise multiplication means each memory dimension is independently gated — some dimensions can be fully retained while others are fully overwritten.",
      randomize: true
    },
    {
      id: "q7",
      type: "fill-blank",
      prompt: "The GRU combines LSTM's forget and input gates into a single ___ gate, and has no separate ___ state.",
      options: ["update, cell", "reset, hidden", "output, memory", "forget, gate"],
      correctIndex: 0,
      explanation: "GRU's update gate z controls both forgetting old info and accepting new info simultaneously: h_t = (1-z)⊙h_{t-1} + z⊙h̃. No separate cell state — the hidden state serves double duty.",
      randomize: false
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "What role does the GRU reset gate r play?",
      options: [
        "It resets the entire hidden state to zero",
        "It controls how much past hidden state contributes to the candidate h̃ computation. When r≈0, the candidate ignores history; when r≈1, it fully incorporates it.",
        "It replaces the forget gate entirely",
        "It controls the learning rate"
      ],
      correctIndex: 1,
      explanation: "h̃ = tanh(W[r⊙h_{t-1}, x] + b). The reset gate modulates h_{t-1} before computing the candidate. This lets the GRU 'forget' irrelevant past context specifically for the new candidate computation.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Approximately how many more parameters does LSTM have compared to vanilla RNN?",
      options: ["Same number", "~2× more", "~4× more (4 gates × (H+D)×H weights + biases)", "~10× more"],
      correctIndex: 2,
      explanation: "Vanilla RNN: Wxh(D×H) + Whh(H×H) + bh(H). LSTM: 4 sets of these (Wf, Wi, Wc, Wo + biases). Total ≈ 4×(H+D)×H + 4×H. For typical sizes, this is roughly 4× the vanilla RNN parameter count.",
      randomize: true
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "In GRU, if z=0.9, what fraction of the NEW candidate h̃ contributes to h_t?",
      code: "z = 0.9\nnew_fraction = z\nold_fraction = 1 - z\nprint(f'new={new_fraction:.1f}, old={old_fraction:.1f}')",
      options: ["new=0.9, old=0.1", "new=0.1, old=0.9", "new=0.5, old=0.5", "new=1.0, old=0.0"],
      correctIndex: 0,
      explanation: "h_t = (1-z)⊙h_{t-1} + z⊙h̃. When z=0.9: 90% new candidate, 10% old state. High z means 'accept the update'. Low z means 'keep the old state'. The update gate interpolates between old and new.",
      randomize: true
    },
    {
      id: "q11",
      type: "ordering",
      prompt: "Order the LSTM forward pass computations:",
      items: ["Concatenate [h_{t-1}, x_t]", "Compute gates (f, i, o) and candidate C̃", "Update cell state: C_t = f⊙C_{t-1} + i⊙C̃", "Compute hidden state: h_t = o⊙tanh(C_t)"],
      correctOrder: [0, 1, 2, 3],
      explanation: "First prepare input, then compute all gates in parallel (they share the same input), then update cell state using forget and input gates, finally produce hidden state using output gate.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "Why must gates_cache store (hx, f, i, c_tilde, o, C_states[t+1]) during LSTM forward?",
      options: [
        "For visualization only",
        "BPTT needs these cached values: gate outputs for their sigmoid/tanh derivatives, hx for weight gradients, and cell states for the recurrent gradient path through C",
        "To enable dropout between gates",
        "NumPy requires pre-allocation"
      ],
      correctIndex: 1,
      explanation: "Each gate's backward pass needs its forward output (sigmoid derivative = g(1-g), tanh derivative = 1-g²). The cell state is needed for the gradient highway. hx is needed for dW computations.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "When should you prefer LSTM over GRU?",
      options: [
        "Always — LSTM is strictly better",
        "When sequences are very long (>200 timesteps) or the task requires complex long-range dependencies where the extra gating capacity provides measurable benefit",
        "When training data is small",
        "When computational speed matters most"
      ],
      correctIndex: 1,
      explanation: "LSTM's separate cell state and output gate provide finer control over memory access. For short sequences or limited data, GRU's simplicity often matches LSTM. The extra complexity pays off mainly on long, complex sequences.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What happens if all LSTM gates output 0.5 at every timestep?",
      options: [
        "The network works normally",
        "Cell state decays by half each step (f=0.5), new info is halved (i=0.5), and output is halved (o=0.5). Information degrades exponentially — similar to vanilla RNN vanishing.",
        "Gradients explode",
        "The cell state grows without bound"
      ],
      correctIndex: 1,
      explanation: "C_t = 0.5·C_{t-1} + 0.5·C̃. After T steps, original C_0 contribution is multiplied by 0.5^T → vanishes. Gates must learn to approach 1 (for retention) or 0 (for selective forgetting) to function properly.",
      randomize: true
    },
    {
      id: "q15",
      type: "fill-blank",
      prompt: "In h_t = o_t ⊙ tanh(C_t), the output gate controls ___ of the cell state is exposed as the hidden state.",
      options: ["how much", "which dimensions", "the sign", "the magnitude only"],
      correctIndex: 1,
      explanation: "o_t is per-dimension in [0,1]. Some dimensions of tanh(C_t) can be fully passed through (o≈1) while others are suppressed (o≈0). The output gate selectively exposes relevant memory contents.",
      randomize: false
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "How does GRU's parameter count compare to LSTM?",
      options: ["Same", "~25% fewer (2 gates vs 3, no separate cell state weights)", "~50% fewer", "More parameters"],
      correctIndex: 1,
      explanation: "LSTM: 4 gates × (H+D)×H params. GRU: 3 weight matrices × (H+D)×H (reset, update, candidate). Roughly 25% fewer parameters, leading to faster training with often comparable performance.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why does the additive structure C_t = f⊙C_{t-1} + i⊙C̃ prevent vanishing gradients but multiplicative h_t = tanh(W·h_{t-1}) doesn't?",
      options: [
        "Addition is computationally cheaper",
        "∂C_t/∂C_{t-1} = f (element-wise). When f≈1, gradient passes through unchanged. In vanilla RNN, ∂h_t/∂h_{t-1} = tanh'·W which multiplies small values repeatedly.",
        "tanh prevents addition from working",
        "There's no mathematical difference"
      ],
      correctIndex: 1,
      explanation: "The partial derivative of an additive update w.r.t. the previous state is just the gate value (≈1 when remembering). Multiplicative updates chain derivatives that shrink exponentially. Addition creates a gradient highway; multiplication creates a gradient bottleneck.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "How many total weight parameters does an LSTM have with H=128, D=64?",
      code: "H, D = 128, 64\nparams_per_gate = (H + D) * H\ntotal = 4 * params_per_gate\nprint(total)",
      options: ["98304", "24576", "49152", "196608"],
      correctIndex: 0,
      explanation: "Each gate: (H+D)×H = 192×128 = 24,576 weights. Four gates: 4×24,576 = 98,304 weight parameters. Plus 4×128 = 512 bias parameters. Total: 98,816. Compare to vanilla RNN: (D+H)×H = 24,576.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "You're building a sentiment analyzer for tweets (avg 15 words). Should you use LSTM or GRU?",
      options: [
        "LSTM — always better for NLP",
        "GRU — sequences are short, fewer parameters reduce overfitting risk on limited data, and performance will likely match LSTM",
        "Vanilla RNN — 15 steps is within its range",
        "Neither — use a CNN instead"
      ],
      correctIndex: 1,
      explanation: "15-word sequences are well within both architectures' capabilities. GRU's smaller parameter count reduces overfitting risk on typical sentiment datasets. Start with GRU; switch to LSTM only if validation metrics show clear improvement.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "During LSTM backprop, why does the gradient through the cell state NOT vanish even over 100+ timesteps?",
      options: [
        "Because tanh is applied to the cell state",
        "The cell state update is additive (C_t = f⊙C_{t-1} + ...). The gradient ∂L/∂C_{t-1} = ∂L/∂C_t ⊙ f_t. When f_t ≈ 1, this is essentially identity — no repeated multiplication by small derivatives.",
        "Gradient clipping prevents vanishing",
        "The output gate compensates for gradient loss"
      ],
      correctIndex: 1,
      explanation: "This is the core LSTM insight. Unlike vanilla RNN where gradients multiply through tanh'·W at each step, the cell state gradient only multiplies by f_t (≈1 when remembering). The additive connection creates an unimpeded gradient highway.",
      randomize: true
    }
  ]}
/>
