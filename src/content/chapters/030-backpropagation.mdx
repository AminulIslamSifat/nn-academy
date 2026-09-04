---
title: "Backpropagation"
slug: "030-backpropagation"
description: "The algorithm that makes deep learning possible. Derive and implement backprop from scratch for a 2-layer network, understanding every gradient."
track: "nn-beginner"
order: 3
read_time: 30
code_time: 25
execution_timeout: 15
prerequisites: ["028-activation-functions", "029-loss-functions"]
---

# Backpropagation

Forward pass: data flows through the network, producing predictions.
Backward pass: ==error flows backward, computing how much each weight contributed to the mistake.==

Backpropagation is just the chain rule applied systematically. That's it. No magic. Just calculus.

## The Network We'll Build

A 2-layer network for MNIST:

```
Input (784) → Linear → ReLU → Linear → Softmax → Loss
     X          W1       h      W2       ŷ        L
              b1                b2
```

Every symbol matters. Let's track them all.

## Forward Pass (Review)

```python
import numpy as np

def forward(X, W1, b1, W2, b2):
    # Layer 1
    Z1 = X @ W1 + b1          # pre-activation
    H = np.maximum(0, Z1)     # ReLU activation
    
    # Layer 2
    Z2 = H @ W2 + b2          # logits
    
    # Softmax
    shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    return Z1, H, Z2, probs
```

## Backward Pass: Step by Step

We need <InlineMath latex="\frac{\partial L}{\partial W_1}" />, <InlineMath latex="\frac{\partial L}{\partial b_1}" />, <InlineMath latex="\frac{\partial L}{\partial W_2}" />, <InlineMath latex="\frac{\partial L}{\partial b_2}" />.

### Step 1: Gradient at Output (Softmax + CCE)

From chapter 029, we know this simplifies beautifully:

<BlockMath latex="\frac{\partial L}{\partial Z_2} = \frac{\hat{y} - y}{N}" />

```python
dZ2 = (probs - y_onehot) / batch_size  # (N, 10)
```

### Step 2: Gradients for W2 and b2

Since <InlineMath latex="Z_2 = H W_2 + b_2" />:

<BlockMath latex="\frac{\partial L}{\partial W_2} = H^T \cdot \frac{\partial L}{\partial Z_2}, \quad \frac{\partial L}{\partial b_2} = \sum_{batch} \frac{\partial L}{\partial Z_2}" />

```python
dW2 = H.T @ dZ2           # (hidden, 10)
db2 = np.sum(dZ2, axis=0)  # (10,)
```

### Step 3: Gradient Through ReLU

Propagate error back to layer 1:

<BlockMath latex="\frac{\partial L}{\partial H} = \frac{\partial L}{\partial Z_2} \cdot W_2^T" />

Then apply ReLU derivative (pass gradient only where <InlineMath latex="Z_1 > 0" />):

<BlockMath latex="\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial H} \odot \mathbb{1}[Z_1 > 0]" />

```python
dH = dZ2 @ W2.T                    # (N, hidden)
dZ1 = dH * (Z1 > 0).astype(float)  # ReLU derivative
```

### Step 4: Gradients for W1 and b1

Same pattern as step 2:

<BlockMath latex="\frac{\partial L}{\partial W_1} = X^T \cdot \frac{\partial L}{\partial Z_1}, \quad \frac{\partial L}{\partial b_1} = \sum_{batch} \frac{\partial L}{\partial Z_1}" />

```python
dW1 = X.T @ dZ1            # (784, hidden)
db1 = np.sum(dZ1, axis=0)   # (hidden,)
```

## Complete Implementation

<PyRunner
  cellId="030-cell-1"
  defaultCode={`import numpy as np

def softmax(Z):
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    H = np.maximum(0, Z1)
    Z2 = H @ W2 + b2
    probs = softmax(Z2)
    return Z1, H, Z2, probs

def backward(X, y_oh, Z1, H, probs, W2):
    N = X.shape[0]
    
    # Output layer gradient
    dZ2 = (probs - y_oh) / N
    dW2 = H.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    
    # Hidden layer gradient
    dH = dZ2 @ W2.T
    dZ1 = dH * (Z1 > 0).astype(float)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# Test with random data
np.random.seed(42)
N, D, H_dim, C = 4, 784, 128, 10
X = np.random.randn(N, D) * 0.01
W1 = np.random.randn(D, H_dim) * np.sqrt(2.0 / D)
b1 = np.zeros(H_dim)
W2 = np.random.randn(H_dim, C) * np.sqrt(2.0 / H_dim)
b2 = np.zeros(C)

y = np.array([3, 7, 1, 9])
y_oh = np.zeros((N, C))
y_oh[np.arange(N), y] = 1

Z1, H, Z2, probs = forward(X, W1, b1, W2, b2)
grads = backward(X, y_oh, Z1, H, probs, W2)

print("Gradient shapes:")
for name, g in grads.items():
    print(f"  {name}: {g.shape}, mean={g.mean():.6f}, std={g.std():.6f}")

print(f"\n✅ All gradients computed without any framework!")
`}
/>

## Training Loop

Now put it all together:

```python
def train(X_train, y_train, X_test, y_test, 
          hidden=128, lr=0.1, epochs=10, batch_size=64):
    N, D = X_train.shape
    C = 10
    
    # Initialize weights (He initialization)
    W1 = np.random.randn(D, hidden) * np.sqrt(2.0 / D)
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, C) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(C)
    
    for epoch in range(epochs):
        # Shuffle data
        idx = np.random.permutation(N)
        total_loss = 0
        correct = 0
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = idx[start:end]
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx]
            
            # One-hot encode
            y_oh = np.zeros((len(y_b), C))
            y_oh[np.arange(len(y_b)), y_b] = 1
            
            # Forward
            Z1, H, Z2, probs = forward(X_b, W1, b1, W2, b2)
            
            # Loss
            loss = -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-7, 1)), axis=1))
            total_loss += loss
            correct += np.sum(np.argmax(probs, axis=1) == y_b)
            
            # Backward
            grads = backward(X_b, y_oh, Z1, H, probs, W2)
            
            # Update weights
            W1 -= lr * grads["dW1"]
            b1 -= lr * grads["db1"]
            W2 -= lr * grads["dW2"]
            b2 -= lr * grads["db2"]
        
        n_batches = N // batch_size
        acc = correct / N * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/n_batches:.4f} | Acc: {acc:.1f}%")
    
    return W1, b1, W2, b2
```

<PyRunner
  cellId="030-cell-2"
  defaultCode={`import numpy as np

def softmax(Z):
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    H = np.maximum(0, Z1)
    Z2 = H @ W2 + b2
    probs = softmax(Z2)
    return Z1, H, Z2, probs

def backward(X, y_oh, Z1, H, probs, W2):
    N = X.shape[0]
    dZ2 = (probs - y_oh) / N
    dW2 = H.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dH = dZ2 @ W2.T
    dZ1 = dH * (Z1 > 0).astype(float)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# Generate synthetic MNIST-like data for demo
np.random.seed(42)
N_train, N_test = 1000, 200
D, C = 784, 10

# Create separable clusters for each digit
X_train = np.random.randn(N_train, D) * 0.3
y_train = np.random.randint(0, C, N_train)
for i in range(N_train):
    X_train[i, y_train[i]*78:(y_train[i]+1)*78] += 1.0

X_test = np.random.randn(N_test, D) * 0.3
y_test = np.random.randint(0, C, N_test)
for i in range(N_test):
    X_test[i, y_test[i]*78:(y_test[i]+1)*78] += 1.0

# Train!
hidden, lr, epochs, batch_size = 128, 0.5, 10, 64
W1 = np.random.randn(D, hidden) * np.sqrt(2.0 / D)
b1 = np.zeros(hidden)
W2 = np.random.randn(hidden, C) * np.sqrt(2.0 / hidden)
b2 = np.zeros(C)

for epoch in range(epochs):
    idx = np.random.permutation(N_train)
    total_loss, correct = 0, 0
    
    for start in range(0, N_train, batch_size):
        end = min(start + batch_size, N_train)
        bi = idx[start:end]
        X_b, y_b = X_train[bi], y_train[bi]
        y_oh = np.zeros((len(y_b), C))
        y_oh[np.arange(len(y_b)), y_b] = 1
        
        Z1, H, Z2, probs = forward(X_b, W1, b1, W2, b2)
        loss = -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-7, 1)), axis=1))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y_b)
        
        grads = backward(X_b, y_oh, Z1, H, probs, W2)
        W1 -= lr * grads["dW1"]
        b1 -= lr * grads["db1"]
        W2 -= lr * grads["dW2"]
        b2 -= lr * grads["db2"]
    
    nb = N_train // batch_size
    print(f"Epoch {epoch+1:2d}/10 | Loss: {total_loss/nb:.4f} | Train Acc: {correct/N_train*100:.1f}%")

# Test accuracy
test_probs = softmax(forward(X_test, W1, b1, W2, b2)[-1])
test_acc = np.sum(np.argmax(test_probs, axis=1) == y_test) / N_test * 100
print(f"\n🎯 Test Accuracy: {test_acc:.1f}%")
`}
/>

## The Chain Rule Visualization

Here's the full computational graph and gradient flow:

```
FORWARD:                          BACKWARD:
X ──→ [W1,b1] ──→ Z1 ──→ ReLU ──→ H ──→ [W2,b2] ──→ Z2 ──→ softmax ──→ ŷ ──→ Loss
                                  ↑                              ↓
                            dL/dZ1 ←── dL/dH ←── dL/dZ2 ←── dL/dŷ
                            (= dH·𝟙[Z1>0])     (= dŷ·W2ᵀ)   (= ŷ-y)/N
```

> [!IMPORTANT] Key Insight
> Every backward arrow is a matrix multiplication or element-wise operation. The shapes must match. If your gradients have wrong shapes, you've made a chain rule error. ==Always verify shapes.==

## Numerical Gradient Checking

How do you know your backprop is correct? Compare against finite differences:

<BlockMath latex="\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}" />

<PyRunner
  cellId="030-cell-3"
  defaultCode={`import numpy as np

def softmax(Z):
    shifted = Z - np.max(Z, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=-1, keepdims=True)

def compute_loss(X, y_oh, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    H = np.maximum(0, Z1)
    Z2 = H @ W2 + b2
    probs = softmax(Z2)
    return -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-7, 1)), axis=1))

np.random.seed(42)
X = np.random.randn(4, 10)
y_oh = np.zeros((4, 5)); y_oh[np.arange(4), [0,2,1,3]] = 1
W1 = np.random.randn(10, 8) * 0.5; b1 = np.zeros(8)
W2 = np.random.randn(8, 5) * 0.5; b2 = np.zeros(5)

# Analytical gradient
def backward_full(X, y_oh, W1, b1, W2, b2):
    Z1 = X @ W1 + b1; H = np.maximum(0, Z1)
    Z2 = H @ W2 + b2; probs = softmax(Z2)
    N = X.shape[0]
    dZ2 = (probs - y_oh) / N
    dW2 = H.T @ dZ2; db2 = np.sum(dZ2, axis=0)
    dH = dZ2 @ W2.T; dZ1 = dH * (Z1 > 0).astype(float)
    dW1 = X.T @ dZ1; db1 = np.sum(dZ1, axis=0)
    return dW1, db1, dW2, db2

a_dW1, a_db1, a_dW2, a_db2 = backward_full(X, y_oh, W1, b1, W2, b2)

# Numerical gradient check for W2[0,0]
eps = 1e-5
orig = W2[0, 0]
W2[0, 0] = orig + eps; lp = compute_loss(X, y_oh, W1, b1, W2, b2)
W2[0, 0] = orig - eps; lm = compute_loss(X, y_oh, W1, b1, W2, b2)
W2[0, 0] = orig
num_grad = (lp - lm) / (2 * eps)

print(f"W2[0,0]:")
print(f"  Analytical: {a_dW2[0,0]:+.8f}")
print(f"  Numerical:  {num_grad:+.8f}")
print(f"  Difference: {abs(a_dW2[0,0] - num_grad):.2e}")
print(f"  {'✅ PASS' if abs(a_dW2[0,0] - num_grad) < 1e-5 else '❌ FAIL'}")
`}
/>

<Callout type="warning" title="Debug With Gradient Checking">
Always numerically check gradients when implementing backprop for new architectures. Run it on a tiny batch first. If relative error > 1e-5, you have a bug. Turn it off during actual training — it's O(n) forward passes per parameter.
</Callout>

<Quiz
  chapterSlug="030-backpropagation"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "In backprop, why does the ReLU derivative use (Z1 > 0) instead of (H > 0)?",
      options: [
        "They're identical since H = max(0, Z1)",
        "Z1 preserves the sign information needed for the gradient gate",
        "H is always non-negative so (H > 0) would always be True",
        "It doesn't matter which one you use"
      ],
      correctIndex: 1,
      explanation: "While H = max(0, Z1) means H > 0 and Z1 > 0 give the same boolean result, conceptually the ReLU derivative gates on the PRE-ACTIVATION value Z1. Using Z1 makes the intent clear: we're checking whether the neuron was active before the nonlinearity.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What shape should dW1 have if X is (64, 784) and hidden layer has 128 neurons?",
      options: ["(64, 128)", "(784, 128)", "(128, 784)", "(128, 128)"],
      correctIndex: 1,
      explanation: "dW1 = X.T @ dZ1. X.T is (784, 64) and dZ1 is (64, 128), so dW1 is (784, 128) — matching W1's shape. Always verify gradient shapes match parameter shapes.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why do we divide dZ2 by N (batch size) in the backward pass?",
      options: [
        "To prevent gradient explosion",
        "Because the loss uses mean (not sum) over the batch, so the gradient inherits the 1/N factor",
        "It's required by the softmax derivative",
        "To normalize across different batch sizes"
      ],
      correctIndex: 1,
      explanation: "CCE loss = -mean(sum(...)). The mean introduces a 1/N factor. Since dL/dZ2 is the derivative of the MEAN loss, it includes 1/N. If you used SUM loss instead, you'd divide by N in the learning rate instead.",
      randomize: true,
    }
  ]}
/>
