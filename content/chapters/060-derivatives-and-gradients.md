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
      prompt: "Why do we use the central difference (f(x+h)-f(x-h))/(2h) instead of forward difference (f(x+h)-f(x))/h?",
      options: ["It's faster", "Central difference has O(h²) error vs O(h) for forward difference — much more accurate for the same step size", "Forward difference doesn't work for neural networks", "It uses fewer function evaluations"],
      correctIndex: 1,
      explanation: "Taylor expansion shows forward difference error is proportional to h, while central difference error is proportional to h². For h=1e-5, that's 1e-5 vs 1e-10 error.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does the gradient vector point toward?",
      options: ["The minimum of the function", "The direction of steepest ascent (maximum increase)", "A random direction", "The origin"],
      correctIndex: 1,
      explanation: "The gradient points uphill — toward the direction of maximum increase. Gradient DESCENT moves in the NEGATIVE gradient direction, which is steepest descent toward the minimum.",
      randomize: false,
    }
  ]}
/>
