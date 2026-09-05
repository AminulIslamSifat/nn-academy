---
title: "Derivatives & Gradients"
slug: "060-derivatives-and-gradients"
description: "From single-variable derivatives to multivariate gradients. Build intuition for what gradients mean geometrically and why they point uphill."
track: "calculus-autodiff"
order: 1
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["003-vectorization"]
---

# Derivatives & Gradients

A derivative measures how a function changes when its input changes slightly. A gradient generalizes this to multiple inputs. ==Gradients tell neural networks which direction reduces loss.==

## Single-Variable Derivative

<BlockMath latex="f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}" />

Numerical approximation with small h:

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

<PyRunner
  cellId="060-cell-1"
  defaultCode={`import numpy as np

def numerical_deriv(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

# Test on known functions
tests = [
    (lambda x: x**2, 3.0, 6.0, "x² at x=3"),
    (lambda x: np.sin(x), 0.0, 1.0, "sin(x) at x=0"),
    (lambda x: np.exp(x), 1.0, np.e, "exp(x) at x=1"),
    (lambda x: 1/(1+np.exp(-x)), 0.0, 0.25, "sigmoid at x=0"),
]

print(f"{'Function':<20} | {'Numerical':>10} | {'Analytical':>10} | {'Error':>10}")
print("─" * 58)
for f, x, expected, name in tests:
    num = numerical_deriv(f, x)
    print(f"{name:<20} | {num:10.6f} | {expected:10.6f} | {abs(num-expected):10.2e}")
`}
/>

## Multivariate Gradient

For <InlineMath latex="f(x_1, x_2, ..., x_n)" />, the gradient is the vector of partial derivatives:

<BlockMath latex="\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)" />

<PyRunner
  cellId="060-cell-2"
  defaultCode={`import numpy as np

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

# f(x,y) = x² + 3xy + y²
f = lambda v: v[0]**2 + 3*v[0]*v[1] + v[1]**2
x = np.array([2.0, 3.0])

grad = numerical_gradient(f, x)
analytical = np.array([2*x[0] + 3*x[1], 3*x[0] + 2*x[1]])

print(f"f(x,y) = x² + 3xy + y² at ({x[0]}, {x[1]})")
print(f"  Numerical gradient: {grad}")
print(f"  Analytical gradient: {analytical}")
print(f"  Match: {np.allclose(grad, analytical)}")
print(f"\n💡 Gradient points in the direction of steepest ascent")
print(f"   Negative gradient = steepest descent = what we use for training!")
`}
/>

## The Jacobian

When the output is also a vector, the gradient becomes a matrix (Jacobian):

<BlockMath latex="J_{ij} = \frac{\partial f_i}{\partial x_j}" />

This is what backpropagation computes layer by layer.

> [!IMPORTANT] Key Insight
> The gradient of the loss w.r.t. every parameter tells you exactly how to adjust each weight to reduce the loss. Gradient descent follows this signal. Everything in deep learning optimization flows from this one idea.

<Quiz
  chapterSlug="060-derivatives-and-gradients"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does a derivative measure?",
      options: ["The value of a function", "How a function's output changes when its input changes slightly — the instantaneous rate of change", "The area under a curve", "The maximum of a function"],
      correctIndex: 1,
      explanation: "f'(x) = lim_{h→0} [f(x+h)-f(x)]/h. It quantifies sensitivity: how much f changes per unit change in x at a specific point.",
      randomize: true
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the numerical derivative of x² at x=3?",
      code: "import numpy as np\ndef numerical_deriv(f, x, h=1e-5):\n    return (f(x+h) - f(x-h)) / (2*h)\nresult = numerical_deriv(lambda x: x**2, 3.0)\nprint(f'{result:.4f}')",
      options: ["≈ 6.0000", "≈ 9.0000", "≈ 3.0000", "≈ 0.0000"],
      correctIndex: 0,
      explanation: "d/dx(x²) = 2x. At x=3: 2×3 = 6. The central difference formula accurately approximates this.",
      randomize: true
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why use central difference (f(x+h)-f(x-h))/(2h) instead of forward difference (f(x+h)-f(x))/h?",
      options: ["It's faster", "Central difference has O(h²) error vs O(h) for forward difference — much more accurate for the same step size", "Forward difference doesn't work for neural nets", "Fewer function evaluations"],
      correctIndex: 1,
      explanation: "Taylor expansion: forward error ∝ h, central error ∝ h². For h=1e-5: forward gives ~1e-5 error, central gives ~1e-10. Two extra evals for 5 orders of magnitude better accuracy.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What does the gradient vector ∇f point toward?",
      options: ["The minimum of f", "The direction of steepest ascent (maximum increase of f)", "A random direction", "The origin"],
      correctIndex: 1,
      explanation: "Gradient points uphill. Gradient DESCENT uses the NEGATIVE gradient (-∇f) to move downhill toward minima. This is the foundation of all neural network training.",
      randomize: true
    },
    {
      id: "q5",
      type: "fill-blank",
      prompt: "For f(x,y) = x² + 3xy + y², the partial derivative ∂f/∂x = ___ and ∂f/∂y = ___.",
      options: ["2x+3y, 3x+2y", "2x, 2y", "x²+3y, 3x+y²", "3x+2y, 2x+3y"],
      correctIndex: 0,
      explanation: "∂/∂x treats y as constant: d/dx(x²+3xy+y²) = 2x+3y. Similarly ∂/∂y = 3x+2y. The gradient is the vector (2x+3y, 3x+2y).",
      randomize: true
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "In the numerical_gradient function, why copy x before modifying it?",
      code: "x_plus = x.copy(); x_plus[i] += h",
      options: ["copy() is faster", "To avoid mutating the original x — each partial derivative needs x modified in only ONE dimension while others stay fixed", "NumPy requires it", "copy() adds randomness"],
      correctIndex: 1,
      explanation: "Partial derivative w.r.t. xᵢ requires changing ONLY xᵢ. Without copy, modifying x_plus would corrupt x for subsequent dimensions.",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is the Jacobian matrix?",
      options: ["A scalar derivative", "When f maps vectors to vectors, the Jacobian Jᵢⱼ = ∂fᵢ/∂xⱼ is the matrix of all partial derivatives — generalizing the gradient to vector-valued functions", "A type of loss function", "The Hessian matrix"],
      correctIndex: 1,
      explanation: "Gradient: scalar output, vector input → vector of partials. Jacobian: vector output, vector input → matrix of partials. Backprop chains Jacobians layer by layer.",
      randomize: true
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the numerical derivative of sigmoid at x=0?",
      code: "import numpy as np\ndef numerical_deriv(f, x, h=1e-5):\n    return (f(x+h) - f(x-h)) / (2*h)\nsigmoid = lambda x: 1/(1+np.exp(-x))\nprint(f'{numerical_deriv(sigmoid, 0.0):.4f}')",
      options: ["≈ 0.2500", "≈ 0.5000", "≈ 1.0000", "≈ 0.0000"],
      correctIndex: 0,
      explanation: "sigmoid'(x) = sigmoid(x)(1-sigmoid(x)). At x=0: sigmoid(0)=0.5, so 0.5×0.5=0.25. This is the maximum slope of sigmoid.",
      randomize: true
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why is the choice of h critical in numerical differentiation?",
      options: ["Any h works equally well", "Too large → truncation error (approximation is coarse). Too small → floating-point cancellation error. Optimal h ≈ 1e-5 to 1e-7 balances both.", "h must always be 1.0", "h doesn't affect accuracy"],
      correctIndex: 1,
      explanation: "Large h: f(x+h) differs significantly from linear approximation. Tiny h: f(x+h)≈f(x-h) and subtraction loses precision. Sweet spot depends on function and precision.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "In gradient descent, why do we subtract the gradient instead of adding it?",
      options: ["Convention", "Gradient points toward steepest ASCENT. To minimize loss, we move in the opposite direction: θ ← θ - lr × ∇L(θ).", "Addition causes overflow", "Subtraction is faster"],
      correctIndex: 1,
      explanation: "∇L points uphill (increasing loss). We want to go downhill (decreasing loss). Negative gradient = steepest descent direction.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What is the computational cost of numerical gradient for n parameters?",
      options: ["O(1)", "O(n) — need 2 function evaluations per parameter (one +h, one -h), totaling 2n evaluations", "O(n²)", "O(log n)"],
      correctIndex: 1,
      explanation: "Each partial derivative requires perturbing one parameter and evaluating f twice. For millions of parameters, this is prohibitively slow — motivating backpropagation.",
      randomize: true
    },
    {
      id: "q12",
      type: "fill-blank",
      prompt: "The gradient of f(x₁,x₂,...,xₙ) is a ___ containing ___ partial derivatives.",
      options: ["vector, n", "scalar, n", "matrix, n²", "tensor, 2n"],
      correctIndex: 0,
      explanation: "Gradient is an n-dimensional vector where each component is ∂f/∂xᵢ. Same dimensionality as the input.",
      randomize: true
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "At a local minimum, what is the gradient?",
      options: ["Maximum", "Zero (or very close to zero) — no direction leads to lower loss", "Infinity", "Undefined"],
      correctIndex: 1,
      explanation: "At a minimum, all partial derivatives are zero: the function is flat. Gradient descent converges when ‖∇L‖ ≈ 0. This is the stopping criterion.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why is numerical gradient checking important during backprop implementation?",
      options: ["It replaces backprop entirely", "It verifies your analytical gradients are correct by comparing them against numerical gradients — catching bugs in chain rule derivations", "It's faster than backprop", "It's required by NumPy"],
      correctIndex: 1,
      explanation: "Backprop derivations are error-prone. Compare analytical gradient vs numerical gradient: if relative error < 1e-5, your implementation is likely correct. Debug with this first.",
      randomize: true
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What does np.allclose check?",
      code: "import numpy as np\na = np.array([1.0000001, 2.0000002])\nb = np.array([1.0, 2.0])\nprint(np.allclose(a, b))",
      options: ["True (values are close within default tolerance)", "False (they're not exactly equal)", "Error", "None"],
      correctIndex: 0,
      explanation: "allclose checks |a-b| ≤ atol + rtol×|b|. Default rtol=1e-5, atol=1e-8. Essential for comparing numerical vs analytical gradients where exact equality is impossible.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "If ∇L(θ) = [0.1, -0.5, 0.3] and learning rate = 0.01, what is the update?",
      options: ["θ += [0.001, -0.005, 0.003]", "θ -= [0.001, -0.005, 0.003] → θ += [-0.001, 0.005, -0.003]", "θ = [0.1, -0.5, 0.3]", "No update needed"],
      correctIndex: 1,
      explanation: "θ ← θ - lr × ∇L = θ - 0.01 × [0.1, -0.5, 0.3] = θ + [-0.001, 0.005, -0.003]. Each parameter moves opposite to its gradient component.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is the relationship between derivatives and backpropagation?",
      options: ["No relationship", "Backpropagation efficiently computes gradients using the chain rule — composing layer-by-layer Jacobians instead of expensive numerical differentiation", "Backprop replaces derivatives", "Derivatives are only for single-variable functions"],
      correctIndex: 1,
      explanation: "Backprop applies chain rule systematically: ∂L/∂w = ∂L/∂y × ∂y/∂w. This computes ALL gradients in O(n) time instead of O(n²) numerical differentiation.",
      randomize: true
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "For f(x) = sin(x), what is f'(π/2)?",
      options: ["1", "0 — cos(π/2) = 0, meaning sin is at its peak and momentarily flat", "-1", "π/2"],
      correctIndex: 1,
      explanation: "d/dx(sin x) = cos x. cos(π/2) = 0. At the peak of sine, the tangent is horizontal — zero rate of change.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Why can't we always use analytical derivatives instead of numerical ones?",
      options: ["Analytical derivatives are slower", "For complex compositions (deep networks), deriving analytical gradients by hand is error-prone. Autograd frameworks automate this, but understanding numerical gradients helps verify correctness.", "Numerical is always preferred", "Analytical derivatives don't exist for neural nets"],
      correctIndex: 1,
      explanation: "Analytical gradients ARE used in production (via autograd). But during development/debugging, numerical gradients serve as ground truth to validate implementations.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which best summarizes the role of gradients in deep learning?",
      options: ["Gradients are optional", "Gradients tell every parameter exactly how to change to reduce loss — they are the signal that drives all learning in neural networks", "Only used for initialization", "Gradients measure model accuracy"],
      correctIndex: 1,
      explanation: "Every weight update in every training step follows the gradient. Without gradients, there is no learning. Understanding gradients is understanding how neural networks learn.",
      randomize: true
    }
  ]}
/>
