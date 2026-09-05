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
      prompt: "What is a computational graph?",
      options: ["A visualization tool", "A directed acyclic graph where nodes represent operations/values and edges represent data flow — it records the computation for automatic gradient computation", "A neural network architecture", "A type of loss function"],
      correctIndex: 1,
      explanation: "Computational graphs capture every operation during the forward pass. During backward, the graph is traversed in reverse to apply the chain rule automatically.",
      randomize: true
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does the Value class store besides the scalar data?",
      options: ["Only the value", "The data, accumulated gradient (.grad), children that produced it, and a _backward function encoding the local derivative", "Only the gradient", "The entire network architecture"],
      correctIndex: 1,
      explanation: "Value stores: data (forward result), grad (accumulated backward signal), _children (graph structure), _backward (local gradient function). This is all autograd needs.",
      randomize: true
    },
    {
      id: "q3",
      type: "code-output",
      prompt: "For y = a * b where a=2, b=-3, what is the local gradient da/dy?",
      code: "# In __mul__._backward:\n# self.grad += other.data * out.grad\n# If out.grad = 1.0 (starting gradient):\na_data, b_data = 2.0, -3.0\nout_grad = 1.0\nda_dy = b_data * out_grad\ndb_da = a_data * out_grad\nprint(f'da/dy = {da_dy}, db/dy = {db_da}')",
      options: ["da/dy = -3.0, db/dy = 2.0", "da/dy = 2.0, db/dy = -3.0", "da/dy = -6.0, db/dy = -6.0", "da/dy = 1.0, db/dy = 1.0"],
      correctIndex: 0,
      explanation: "d(a×b)/da = b = -3. d(a×b)/db = a = 2. The local gradient for multiplication is simply the OTHER operand.",
      randomize: true
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why does each operation's _backward use += instead of = for gradients?",
      options: ["Convention", "A variable can be used in multiple operations. += accumulates gradient contributions from ALL paths (multivariate chain rule).", "To prevent overwriting", "Numerical stability"],
      correctIndex: 1,
      explanation: "If x appears in both (x+y) and (x*z), total ∂L/∂x = ∂L/∂(x+y)·∂(x+y)/∂x + ∂L/∂(x*z)·∂(x*z)/∂x. Each path adds its contribution via +=.",
      randomize: true
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Why does the backward pass use topological sort?",
      options: ["For speed", "To ensure each node's gradient is fully accumulated from all downstream paths BEFORE its own _backward runs", "To reduce memory", "It's optional"],
      correctIndex: 1,
      explanation: "A node may feed multiple downstream ops. All their gradient contributions must be summed (via +=) before this node propagates its total gradient to its own inputs.",
      randomize: true
    },
    {
      id: "q6",
      type: "fill-blank",
      prompt: "The backward pass visits nodes in ___ topological order. The root node's gradient is initialized to ___.",
      options: ["reverse, 1.0", "forward, 0.0", "reverse, 0.0", "forward, 1.0"],
      correctIndex: 0,
      explanation: "Reverse topo ensures children are processed before parents. Root gradient = 1.0 because dL/dL = 1 (the loss w.r.t. itself).",
      randomize: true
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is the local gradient for addition (a + b)?",
      options: ["da/d(a+b) = b, db/d(a+b) = a", "da/d(a+b) = 1, db/d(a+b) = 1 — gradient flows unchanged through addition", "da/d(a+b) = a, db/d(a+b) = b", "da/d(a+b) = 0, db/d(a+b) = 0"],
      correctIndex: 1,
      explanation: "d(a+b)/da = 1, d(a+b)/db = 1. Addition is a gradient pass-through: upstream gradient is copied equally to both inputs.",
      randomize: true
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "How does PyTorch's Tensor relate to the Value class shown here?",
      options: ["Completely different", "PyTorch's Tensor is essentially Value but for n-dimensional arrays with GPU support, hundreds of operations, and optimized C++ backend — same principle", "Tensor doesn't track gradients", "Value is more powerful than Tensor"],
      correctIndex: 1,
      explanation: "Same concept: record operations → build graph → traverse backward. PyTorch scales it to tensors, GPUs, distributed training, and complex operations.",
      randomize: true
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is sigmoid'(z) expressed in terms of sigmoid(z)?",
      code: "import numpy as np\nz = 1.0\nsig = 1/(1+np.exp(-z))\nsig_deriv = sig * (1 - sig)\nprint(f'σ({z})={sig:.4f}, σ\'= {sig_deriv:.4f}')",
      options: ["σ(z)(1-σ(z)) — uses the output value itself", "exp(-z)/(1+exp(-z))²", "1/(1+exp(-z))", "σ(z)²"],
      correctIndex: 0,
      explanation: "σ'(z) = σ(z)(1-σ(z)). This elegant form means we only need the forward output to compute the backward gradient — no need to recompute exp.",
      randomize: true
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What happens if you forget to call backward() on the loss?",
      options: ["Training continues normally", "All .grad values remain 0.0 — no gradients computed, no learning occurs", "Error is raised", "Gradients are computed numerically instead"],
      correctIndex: 1,
      explanation: "backward() triggers the reverse traversal. Without it, the graph exists but gradients are never propagated. Every .grad stays at its initial value of 0.",
      randomize: true
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why is the computational graph described as a DAG (Directed Acyclic Graph)?",
      options: ["Neural networks are always linear", "Operations have a clear direction (inputs→outputs) and no cycles — you can't have a value that depends on itself", "DAGs are faster to traverse", "Cycles cause infinite loops in forward pass"],
      correctIndex: 1,
      explanation: "Each operation produces new values from previous ones. No circular dependencies. This guarantees topological sort exists and backward pass terminates.",
      randomize: true
    },
    {
      id: "q12",
      type: "multiple-choice",
      prompt: "In the topological sort, why do we recurse into children BEFORE appending the current node?",
      options: ["Arbitrary choice", "Post-order traversal ensures all descendants appear before their ancestors in the list. Reversing gives ancestors-first (output-to-input) order for backward.", "Pre-order is incorrect", "BFS would be better"],
      correctIndex: 1,
      explanation: "build(child) first, then append(self). Result: leaves first, root last. reversed(): root first, leaves last. This is exactly the backward pass order.",
      randomize: true
    },
    {
      id: "q13",
      type: "fill-blank",
      prompt: "During forward pass, each operation creates a new Value and stores a ___ function. During backward pass, these functions are called in ___ order to accumulate gradients.",
      options: ["_backward, reverse topological", "_forward, topological", "_backward, random", "_gradient, forward"],
      correctIndex: 0,
      explanation: "Forward builds the graph with _backward closures. Backward traverses in reverse topo order, calling each _backward to propagate gradients to children.",
      randomize: true
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What is the key advantage of autograd over manual gradient derivation?",
      options: ["Autograd is faster", "Autograd eliminates error-prone manual chain rule derivations — you define the forward computation and gradients are computed automatically and correctly", "Manual derivation is impossible", "Autograd uses less memory"],
      correctIndex: 1,
      explanation: "Manual backprop for complex architectures is extremely error-prone. Autograd mechanically applies chain rule through the recorded graph. Correctness is guaranteed by construction.",
      randomize: true
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "If a Value is used in three different operations, how many times will its .grad be updated during backward?",
      options: ["Once", "Three times — once from each downstream path, each adding its contribution via +=", "Zero times", "Depends on depth"],
      correctIndex: 1,
      explanation: "Each downstream operation's _backward adds its gradient contribution. Three uses → three += updates. Final .grad is the sum of all three path contributions.",
      randomize: true
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "Why initialize self.grad = 0.0 in the Value constructor?",
      options: ["Convention", "Gradients accumulate via +=. Starting at 0 ensures clean accumulation. Non-zero initial values would corrupt gradient computation.", "NumPy requires it", "0.0 is the default float"],
      correctIndex: 1,
      explanation: "Every _backward does self.grad += contribution. Starting at 0 means final .grad = sum of all contributions. Starting elsewhere adds bias.",
      randomize: true
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What distinguishes reverse-mode autodiff from forward-mode?",
      options: ["Speed of forward pass", "Reverse mode computes gradients for ALL inputs in one backward pass from one output. Forward mode computes gradient of one output w.r.t. ALL inputs in one forward pass.", "Memory usage only", "They produce different results"],
      correctIndex: 1,
      explanation: "Reverse: efficient when outputs << inputs (typical in NN: 1 loss, millions of params). Forward: efficient when inputs << outputs. NN training always uses reverse mode.",
      randomize: true
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "For y = sigmoid(a*b+c) with a=2,b=-3,c=10, what is the inner value g=a*b+c?",
      code: "a, b, c = 2.0, -3.0, 10.0\ng = a*b + c\nprint(f'g = {a}*{b}+{c} = {g}')",
      options: ["4.0", "-6.0", "16.0", "7.0"],
      correctIndex: 0,
      explanation: "2×(-3)+10 = -6+10 = 4. Then y = sigmoid(4) ≈ 0.982. The chain rule flows backward through this computation.",
      randomize: true
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "What would happen if the topological sort visited nodes in FORWARD order instead of reverse?",
      options: ["Same result", "Gradients would be wrong — leaf nodes would propagate before their downstream gradients are accumulated, producing incomplete/partial gradients", "Faster execution", "No effect on correctness"],
      correctIndex: 1,
      explanation: "Forward order processes inputs before outputs. But inputs need the TOTAL gradient from all downstream paths, which hasn't been computed yet. Reverse order guarantees completeness.",
      randomize: true
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Which best describes what building an autograd engine teaches you?",
      options: ["How to use PyTorch", "The exact mechanism behind every deep learning framework — recording operations, building graphs, and mechanically applying chain rule in reverse", "How to optimize GPU code", "How to design neural network architectures"],
      correctIndex: 1,
      explanation: "Building autograd demystifies frameworks. You see that 'magic' gradient computation is just organized chain rule. This understanding transfers to debugging, custom ops, and research.",
      randomize: true
    }
  ]}
/>
