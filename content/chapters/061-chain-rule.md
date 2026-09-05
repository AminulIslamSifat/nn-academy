---
title: "The Chain Rule"
slug: "061-chain-rule"
description: "The mathematical foundation of backpropagation. Understand how derivatives compose through nested functions and why this enables efficient gradient computation."
track: "calculus-autodiff"
order: 2
read_time: 22
code_time: 18
execution_timeout: 10
prerequisites: ["060-derivatives-and-gradients"]
---

# The Chain Rule

Backpropagation IS the chain rule applied systematically. ==If you understand the chain rule, you understand backprop.==

## Single-Variable Chain Rule

For composed functions <InlineMath latex="y = f(g(x))" />:

<BlockMath latex="\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}" />

<PyRunner
  cellId="061-cell-1"
  defaultCode={`import numpy as np

# y = sigmoid(x² + 1)
# dy/dx = sigmoid'(x²+1) · 2x

def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_deriv(out): return out * (1 - out)

x = 2.0

# Forward
g = x**2 + 1          # g(x) = x² + 1 = 5
y = sigmoid(g)         # y = σ(5)

# Backward (chain rule)
dy_dg = sigmoid_deriv(y)   # dσ/dg at g=5
dg_dx = 2 * x              # dg/dx = 2x = 4
dy_dx = dy_dg * dg_dx      # chain rule!

# Verify numerically
h = 1e-7
numerical = (sigmoid((x+h)**2+1) - sigmoid((x-h)**2+1)) / (2*h)

print(f"y = sigmoid(x² + 1) at x = {x}")
print(f"  Forward:  g = {g}, y = {y:.6f}")
print(f"  Chain rule: dy/dx = {dy_dg:.6f} × {dg_dx:.1f} = {dy_dx:.6f}")
print(f"  Numerical:  dy/dx = {numerical:.6f}")
print(f"  Match: {np.isclose(dy_dx, numerical)}")
`}
/>

## Multi-Layer Chain Rule

For a deep network <InlineMath latex="y = f_n(f_{n-1}(...f_1(x)...))" />:

<BlockMath latex="\frac{dy}{dx} = \frac{\partial f_n}{\partial f_{n-1}} \cdot \frac{\partial f_{n-1}}{\partial f_{n-2}} \cdots \frac{\partial f_1}{\partial x}" />

This product of matrices is exactly what backpropagation computes — but in reverse order, reusing intermediate results.

<PyRunner
  cellId="061-cell-2"
  defaultCode={`import numpy as np

# 3-layer network: y = σ(W3 · σ(W2 · σ(W1 · x)))
np.random.seed(42)
x = np.random.randn(3)
W1 = np.random.randn(3, 4) * 0.5
W2 = np.random.randn(4, 4) * 0.5
W3 = np.random.randn(4, 1) * 0.5

def sigmoid(z): return 1/(1+np.exp(-z))

# Forward pass (cache everything)
z1 = W1.T @ x;  a1 = sigmoid(z1)
z2 = W2.T @ a1; a2 = sigmoid(z2)
z3 = W3.T @ a2; y = sigmoid(z3)

# Backward pass (chain rule in reverse)
dy_dz3 = y * (1 - y)
dz3_da2 = W3
da2_dz2 = a2 * (1 - a2)
dz2_da1 = W2
da1_dz1 = a1 * (1 - a1)
dz1_dx = W1

# Full gradient dy/dx
dy_dx = dz1_dx @ (da1_dz1 * (dz2_da1 @ (da2_dz2 * (dz3_da2 @ dy_dz3))))

# Verify numerically
h = 1e-5
num_grad = np.zeros(3)
for i in range(3):
    xp = x.copy(); xp[i] += h
    xm = x.copy(); xm[i] -= h
    yp = sigmoid(W3.T @ sigmoid(W2.T @ sigmoid(W1.T @ xp)))
    ym = sigmoid(W3.T @ sigmoid(W2.T @ sigmoid(W1.T @ xm)))
    num_grad[i] = (yp - ym) / (2*h)

print(f"Chain rule through 3 layers:")
print(f"  Analytical gradient: {dy_dx.flatten().round(6)}")
print(f"  Numerical gradient:  {num_grad.round(6)}")
print(f"  Match: {np.allclose(dy_dx.flatten(), num_grad, atol=1e-4)}")
print(f"\n✅ This IS backpropagation — just the chain rule applied layer by layer")
`}
/>

## Why Reverse Mode?

Forward mode computes gradients one input at a time. Reverse mode (backprop) computes gradients for ALL parameters in one pass. For a network with millions of parameters and one scalar loss, reverse mode is millions of times more efficient.

> [!IMPORTANT] The Fundamental Insight
> Forward pass: compute and cache intermediates. Backward pass: multiply local gradients in reverse, reusing cached values. Total cost: ~2× forward pass. This is why deep learning is computationally feasible.

<Quiz
  chapterSlug="061-chain-rule"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the chain rule for y = f(g(x))?",
      options: ["dy/dx = f'(x) + g'(x)", "dy/dx = (dy/dg) × (dg/df) — the derivative of the composition is the product of individual derivatives", "dy/dx = f(g'(x))", "dy/dx = f'(g(x))"],
      correctIndex: 1,
      explanation: "Chain rule: dy/dx = f'(g(x)) · g'(x). Derivatives compose multiplicatively through nested functions. This is the mathematical foundation of backpropagation.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "For y = sigmoid(x²+1) at x=2, what is dg/dx?",
      code: "import numpy as np\nx = 2.0\ng = x**2 + 1\ndg_dx = 2 * x\nprint(f'g={g}, dg/dx={dg_dx}')",
      options: ["4.0", "5.0", "2.0", "8.0"],
      correctIndex: 0,
      explanation: "g(x) = x²+1, so dg/dx = 2x. At x=2: dg/dx = 4. This is the inner derivative in the chain rule.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In the chain rule dy/dx = (dy/dg)(dg/dx), what is dy/dg for y = sigmoid(g)?",
      options: ["sigmoid(g)", "sigmoid(g) × (1 - sigmoid(g)) — the sigmoid derivative expressed in terms of its output", "1/(1+exp(-g))", "exp(-g)"],
      correctIndex: 1,
      explanation: "σ'(z) = σ(z)(1-σ(z)). Conveniently expressed using the output value itself, which is already computed during the forward pass.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why is reverse-mode differentiation (backprop) preferred over forward-mode for neural networks?",
      options: ["Easier to implement", "Networks have millions of parameters but one scalar loss. Reverse mode computes ALL parameter gradients in one backward pass; forward mode needs one pass per parameter.", "Forward mode doesn't work with ReLU", "Reverse mode uses less memory"],
      correctIndex: 1,
      explanation: "Forward mode: O(params) passes. Reverse mode: O(1) passes. For 100M parameters, that's 100M× speedup. This efficiency is why deep learning is computationally feasible.",
      randomize: true
    },
    {
      id: "q5",
      type: "fill-blank",
      prompt: "For a deep network y = fₙ(fₙ₋₁(...f₁(x)...)), the full derivative is the ___ of all layer-wise partial derivatives, computed in ___ order during backprop.",
      options: ["product, reverse", "sum, forward", "product, forward", "sum, reverse"],
      correctIndex: 0,
      explanation: "Chain rule gives a product of Jacobians. Backprop computes this product from output to input (reverse order), reusing intermediate results.",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What does backpropagation cache during the forward pass?",
      options: ["Nothing", "Intermediate activations and pre-activation values needed to compute local gradients during the backward pass", "Only the final loss", "The input data only"],
      correctIndex: 1,
      explanation: "sigmoid'(z) = σ(z)(1-σ(z)) needs σ(z) from forward pass. ReLU'(z) needs z to check sign. Every layer caches its forward outputs for backward use.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is the computational cost of backpropagation relative to the forward pass?",
      options: ["10× more expensive", "~2× the forward pass cost — one forward + one backward pass computes gradients for ALL parameters", "Same as forward", "O(n²) times forward"],
      correctIndex: 1,
      explanation: "Backward pass has similar FLOPs to forward pass. Total ≈ 2× forward. Compare to numerical gradient: O(n) forward passes. This 2× vs O(n)× is why backprop enabled deep learning.",
      randomize: true
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "In the 3-layer example, what operation combines local gradients?",
      code: "# Simplified chain through 3 layers\ndy_dz3 = 0.1   # output gradient\nda2_dz2 = 0.2  # layer 2 local grad\nda1_dz1 = 0.3  # layer 1 local grad\n# Chain rule: multiply local gradients\nfull_grad = dy_dz3 * da2_dz2 * da1_dz1\nprint(f'{full_grad:.6f}')",
      options: ["0.006000", "0.600000", "0.060000", "0.000600"],
      correctIndex: 0,
      explanation: "0.1 × 0.2 × 0.3 = 0.006. Chain rule multiplies local derivatives. Note how gradients shrink through layers — this is the vanishing gradient problem.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why do we need to verify analytical gradients against numerical gradients?",
      options: ["Numerical is always more accurate", "Chain rule derivations for deep networks are error-prone. Numerical gradients serve as ground truth to catch implementation bugs.", "Analytical gradients don't exist", "It's required by NumPy"],
      correctIndex: 1,
      explanation: "A single sign error or missing term in chain rule derivation silently produces wrong gradients. Gradient checking catches these bugs before they corrupt training.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "In the backward pass, why are gradients multiplied (not added)?",
      options: ["Convention", "The chain rule produces products of derivatives. Each layer's contribution scales the gradient flowing backward through the network.", "Addition would cause overflow", "Multiplication is faster"],
      correctIndex: 1,
      explanation: "dy/dx = (dy/da)(da/db)(db/dx). Composition → multiplication. This is fundamental calculus, not a design choice.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What causes the vanishing gradient problem in terms of the chain rule?",
      options: ["Too many parameters", "When each local derivative is < 1 (e.g., sigmoid' ≤ 0.25), their product shrinks exponentially with depth: 0.25^L → 0 for large L", "Learning rate too high", "Batch size too small"],
      correctIndex: 1,
      explanation: "Chain rule multiplies local gradients. If each is < 1, the product decays exponentially. 10 layers of sigmoid: 0.25¹⁰ ≈ 10⁻⁶. Early layers get near-zero gradients.",
      randomize: true
    },
    {
      id: "q12",
      type: "fill-blank",
      prompt: "Forward pass computes ___ and caches intermediates. Backward pass multiplies ___ gradients in reverse order using cached values.",
      options: ["activations, local", "gradients, global", "losses, partial", "weights, total"],
      correctIndex: 0,
      explanation: "Forward: x → z₁ → a₁ → z₂ → a₂ → ... → y (cache all zᵢ, aᵢ). Backward: ∂L/∂y → ∂L/∂zₙ → ∂L/∂aₙ₋₁ → ... → ∂L/∂x using cached values.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "How does ReLU help with vanishing gradients compared to sigmoid?",
      options: ["ReLU has larger derivatives", "ReLU'(z) = 1 for z > 0, so active neurons pass gradients unchanged (no shrinkage). Only inactive neurons (z ≤ 0) block gradients.", "ReLU doesn't use chain rule", "ReLU has no derivative"],
      correctIndex: 1,
      explanation: "Sigmoid' ≤ 0.25 everywhere → guaranteed shrinkage. ReLU' = 1 for positive inputs → gradients flow without attenuation through active paths. This enables deeper networks.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "In matrix form, what does the chain rule look like for a linear layer y = Wx followed by activation a = σ(y)?",
      options: ["∂L/∂x = ∂L/∂a × σ'(y)", "∂L/∂x = Wᵀ × (σ'(y) ⊙ ∂L/∂a) — transpose of weights times element-wise product of activation derivative and upstream gradient", "∂L/∂x = W × ∂L/∂a", "∂L/∂x = ∂L/∂a × W"],
      correctIndex: 1,
      explanation: "Linear layer Jacobian is W. By chain rule: ∂L/∂x = (∂a/∂y)(∂y/∂x)ᵀ × ∂L/∂a = diag(σ') × W × ∂L/∂a = Wᵀ(σ' ⊙ ∂L/∂a). Transpose appears because we go backward.",
      randomize: true
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What does np.isclose compare?",
      code: "import numpy as np\nanalytical = 0.006001\nnumerical = 0.006000\nprint(np.isclose(analytical, numerical))",
      options: ["True (within default tolerance)", "False (not exactly equal)", "Error", "None"],
      correctIndex: 0,
      explanation: "isclose checks |a-b| ≤ atol + rtol×|b|. Default rtol=1e-5. Difference of 1e-6 is well within tolerance. Use this for gradient checking.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What is automatic differentiation (autograd)?",
      options: ["Numerical differentiation with smaller h", "Systematic application of the chain rule through a computation graph — recording operations during forward pass and applying chain rule automatically during backward", "Symbolic differentiation", "Finite difference method"],
      correctIndex: 1,
      explanation: "Autograd records every operation (computational graph). During backward, it applies chain rule mechanically. No manual derivation needed. PyTorch/JAX/TensorFlow all use this.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why does the backward pass go in REVERSE order?",
      options: ["Convention", "Each layer's gradient depends on the gradient from the layer ABOVE it. You must start from the loss and propagate backward to accumulate the full chain.", "Forward order doesn't work mathematically", "Memory access patterns"],
      correctIndex: 1,
      explanation: "∂L/∂w₁ requires ∂L/∂a₁ which requires ∂L/∂a₂ which requires ∂L/∂y. Dependencies flow output→input. Reverse order satisfies these dependencies naturally.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "If a network has 10 sigmoid layers, approximately what fraction of the original gradient reaches layer 1?",
      options: ["≈ 0.25", "≈ 0.25¹⁰ ≈ 10⁻⁶ — essentially zero, explaining why deep sigmoid networks don't train", "≈ 0.5", "≈ 1.0"],
      correctIndex: 1,
      explanation: "Max sigmoid' = 0.25. Through 10 layers: 0.25¹⁰ ≈ 9.5×10⁻⁷. Layer 1 receives virtually no gradient signal. This motivated ReLU and residual connections.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What is the relationship between backpropagation and the chain rule?",
      options: ["They're unrelated", "Backpropagation IS the chain rule applied systematically to a computation graph — it's not a separate algorithm, just organized chain rule", "Backprop replaces the chain rule", "Chain rule only works for single-variable functions"],
      correctIndex: 1,
      explanation: "Every step in backprop is a direct application of the chain rule. The 'innovation' is the systematic organization (computation graph + caching + reverse traversal), not new mathematics.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which statement best captures why understanding the chain rule matters for deep learning?",
      options: ["It's only needed for exams", "The chain rule explains HOW gradients flow, WHY they vanish/explode, and WHAT backprop actually computes — it's the mathematical engine of all learning", "Modern frameworks eliminate the need to understand it", "Only researchers need it"],
      correctIndex: 1,
      explanation: "Vanishing gradients, exploding gradients, skip connections, normalization — all make sense through the lens of chain rule multiplication. Understanding it means understanding deep learning.",
      randomize: true
    }
  ]}
/>
