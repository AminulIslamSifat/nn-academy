---
title: "Computational Graphs & Autograd"
slug: "062-computational-graphs"
description: "Represent computations as directed acyclic graphs. Build a minimal autograd engine from scratch that automatically computes gradients via reverse-mode differentiation."
track: "calculus-autodiff"
order: 3
read_time: 28
code_time: 25
execution_timeout: 15
prerequisites: ["061-chain-rule"]
---

# Computational Graphs & Autograd

Every neural network computation can be represented as a graph of operations. ==Autograd walks this graph backward to compute gradients automatically.== This is what PyTorch and TensorFlow do under the hood.

## The Graph

```
x ──→ [×W1] ──→ [+] ──→ [σ] ──→ [×W2] ──→ [+] ──→ [σ] ──→ loss
         ↑        ↑              ↑        ↑
        W1       b1             W2       b2
```

Each node stores its value and knows how to compute its local gradient.

## Building a Minimal Autograd Engine

```python
import numpy as np

class Value:
    """Scalar value with gradient tracking."""
    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1 / (1 + np.exp(-self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Topological sort
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)
        build(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

<PyRunner
  cellId="062-cell-1"
  defaultCode={`import numpy as np

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._children = set(children)
        self._op = op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1/(1+np.exp(-self.data)), (self,), 'σ')
        def _backward():
            self.grad += out.data*(1-out.data)*out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for n in reversed(topo): n._backward()
    
    def __repr__(self): return f"Value({self.data:.4f}, grad={self.grad:.4f})"

# Test: y = sigmoid(a*b + c)
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
y = (a * b + c).sigmoid()
y.backward()

print(f"y = sigmoid(a*b + c) = sigmoid({a.data}*{b.data}+{c.data})")
print(f"  y = {y}")
print(f"  a.grad = {a.grad:.6f}")
print(f"  b.grad = {b.grad:.6f}")
print(f"  c.grad = {c.grad:.6f}")

# Verify numerically
h = 1e-5
for name, val in [('a', a), ('b', b), ('c', c)]:
    orig = val.data
    val.data = orig + h
    yp = (Value(orig+h)*b+c).sigmoid().data if name=='a' else (a*Value(orig+h)+c).sigmoid().data if name=='b' else (a*b+Value(orig+h)).sigmoid().data
    val.data = orig - h
    ym = (Value(orig-h)*b+c).sigmoid().data if name=='a' else (a*Value(orig-h)+c).sigmoid().data if name=='b' else (a*b+Value(orig-h)).sigmoid().data
    val.data = orig
    num = (yp-ym)/(2*h)
    print(f"  {name} numerical grad: {num:.6f} (match: {abs(num - val.grad) < 1e-4})")
`}
/>

> [!IMPORTANT] This Is What PyTorch Does
> PyTorch's `Tensor` is essentially this `Value` class but optimized for n-dimensional arrays, GPU acceleration, and hundreds of operations. The principle is identical: build a graph during forward, walk it backward to accumulate gradients.

## Topological Sort

The backward pass must visit nodes in reverse topological order — children before parents. This ensures that when we compute a node's gradient, all downstream gradients are already computed.

<Quiz
  chapterSlug="062-computational-graphs"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why does the backward pass use topological sort?",
      options: ["For speed", "To ensure each node's gradient is fully accumulated from all downstream paths before its own backward function runs", "To reduce memory usage", "It's optional"],
      correctIndex: 1,
      explanation: "A node can feed into multiple downstream operations. All of their gradient contributions must be summed before the node's own backward pass distributes the total gradient to its inputs.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "In the autograd engine, why does each operation's _backward use += instead of = for gradients?",
      options: ["Convention", "A variable can be used in multiple operations; += accumulates gradient contributions from all paths (multivariate chain rule)", "To prevent overwriting", "For numerical stability"],
      correctIndex: 1,
      explanation: "If x is used in both (x+y) and (x*z), the total gradient dx/dL has contributions from both paths. Using += accumulates them. Using = would lose one contribution.",
      randomize: false,
    }
  ]}
/>
