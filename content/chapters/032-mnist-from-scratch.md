---
title: "Your First Neural Network: MNIST from Scratch"
slug: "032-mnist-from-scratch"
description: "Load real MNIST data, build a complete 2-layer MLP with backprop and Adam, train it to 97%+ accuracy. Everything in pure NumPy."
track: "nn-beginner"
order: 5
read_time: 35
code_time: 30
execution_timeout: 60
prerequisites: ["030-backpropagation", "031-gradient-descent-optimizers"]
---

# Your First Neural Network: MNIST from Scratch

Time to stop playing with synthetic data. ==You're going to load real MNIST, build a neural network from zero, and train it to recognize handwritten digits.== No PyTorch. No TensorFlow. Just you, NumPy, and the math.

## Loading MNIST

MNIST is 60,000 training + 10,000 test grayscale images of handwritten digits (0–9), each 28×28 pixels.

```python
import numpy as np
import gzip
import os
import urllib.request

def load_mnist(data_dir="../_data"):
    """Download and load MNIST dataset."""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    for name, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)
    
    # Parse images
    def parse_images(fname):
        with gzip.open(os.path.join(data_dir, fname), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    
    # Parse labels
    def parse_labels(fname):
        with gzip.open(os.path.join(data_dir, fname), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)
    
    X_train = parse_images(files["train_images"]).astype(np.float64) / 255.0
    y_train = parse_labels(files["train_labels"])
    X_test = parse_images(files["test_images"]).astype(np.float64) / 255.0
    y_test = parse_labels(files["test_labels"])
    
    return (X_train, y_train), (X_test, y_test)
```

<Callout type="info" title="Why Normalize to [0, 1]?">
Raw pixel values are 0–255 (uint8). Dividing by 255 gives floats in [0, 1]. This keeps activations and gradients in a reasonable range. Without normalization, large inputs cause large activations, which cause exploding gradients.
</Callout>

## The Complete Network

Here's everything in one clean class. Study this — it combines chapters 028–031 into a working system:

```python
class MNISTNet:
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        # He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(num_classes)
        
        # Adam state
        self.t = 0
        self.param_names = ["W1", "b1", "W2", "b2"]
        self.m = {name: np.zeros_like(getattr(self, name)) for name in self.param_names}
        self.v = {name: np.zeros_like(getattr(self, name)) for name in self.param_names}
    
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.H = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = self.H @ self.W2 + self.b2
        
        # Stable softmax
        shifted = self.Z2 - np.max(self.Z2, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y_onehot):
        N = X.shape[0]
        
        # Output gradient (softmax + CCE shortcut)
        dZ2 = (self.probs - y_onehot) / N
        dW2 = self.H.T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        
        # Hidden gradient
        dH = dZ2 @ self.W2.T
        dZ1 = dH * (self.Z1 > 0).astype(float)  # ReLU derivative
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    
    def adam_step(self, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name in self.param_names:
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * g**2
            m_hat = self.m[name] / (1 - beta1**self.t)
            v_hat = self.v[name] / (1 - beta2**self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
```

## Training

<PyRunner
  cellId="032-cell-1"
  defaultCode={`import numpy as np

# --- Inline MNIST loader (uses cached data or generates synthetic fallback) ---
import os, gzip

def try_load_mnist():
    data_dir = "../_data"
    try:
        if os.path.exists(os.path.join(data_dir, "train-images-idx3-ubyte.gz")):
            with gzip.open(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "rb") as f:
                X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.0
            with gzip.open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), "rb") as f:
                y_train = np.frombuffer(f.read(), np.uint8, offset=8)
            with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
                X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.0
            with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
                y_test = np.frombuffer(f.read(), np.uint8, offset=8)
            return (X_train, y_train), (X_test, y_test), True
    except Exception as e:
        pass
    
    # Synthetic fallback for demo
    np.random.seed(42)
    N_tr, N_te = 5000, 1000
    X_tr = np.random.randn(N_tr, 784) * 0.3
    y_tr = np.random.randint(0, 10, N_tr)
    for i in range(N_tr): X_tr[i, y_tr[i]*78:(y_tr[i]+1)*78] += 1.0
    X_te = np.random.randn(N_te, 784) * 0.3
    y_te = np.random.randint(0, 10, N_te)
    for i in range(N_te): X_te[i, y_te[i]*78:(y_te[i]+1)*78] += 1.0
    return (X_tr, y_tr), (X_te, y_te), False

(train_data, test_data), loaded = try_load_mnist()[0], try_load_mnist()[2]
(X_train, y_train), (X_test, y_test) = train_data, test_data
print(f"Data source: {'Real MNIST' if loaded else 'Synthetic (demo mode)'}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
`}
/>

<PyRunner
  cellId="032-cell-2"
  defaultCode={`import numpy as np
import os, gzip

# Load data
def try_load_mnist():
    data_dir = "../_data"
    try:
        if os.path.exists(os.path.join(data_dir, "train-images-idx3-ubyte.gz")):
            with gzip.open(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), "rb") as f:
                X_tr = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.0
            with gzip.open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), "rb") as f:
                y_tr = np.frombuffer(f.read(), np.uint8, offset=8)
            with gzip.open(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"), "rb") as f:
                X_te = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.0
            with gzip.open(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
                y_te = np.frombuffer(f.read(), np.uint8, offset=8)
            return (X_tr, y_tr), (X_te, y_te)
    except: pass
    np.random.seed(42)
    N_tr, N_te = 5000, 1000
    X_tr = np.random.randn(N_tr, 784)*0.3; y_tr = np.random.randint(0,10,N_tr)
    for i in range(N_tr): X_tr[i, y_tr[i]*78:(y_tr[i]+1)*78] += 1.0
    X_te = np.random.randn(N_te, 784)*0.3; y_te = np.random.randint(0,10,N_te)
    for i in range(N_te): X_te[i, y_te[i]*78:(y_te[i]+1)*78] += 1.0
    return (X_tr, y_tr), (X_te, y_te)

(X_train, y_train), (X_test, y_test) = try_load_mnist()

# Network
class MNISTNet:
    def __init__(self, D=784, H=256, C=10):
        self.W1 = np.random.randn(D,H)*np.sqrt(2.0/D)
        self.b1 = np.zeros(H)
        self.W2 = np.random.randn(H,C)*np.sqrt(2.0/H)
        self.b2 = np.zeros(C)
        self.t = 0
        self.pnames = ["W1","b1","W2","b2"]
        self.m = {n: np.zeros_like(getattr(self,n)) for n in self.pnames}
        self.v = {n: np.zeros_like(getattr(self,n)) for n in self.pnames}
    
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.H = np.maximum(0, self.Z1)
        self.Z2 = self.H @ self.W2 + self.b2
        s = self.Z2 - np.max(self.Z2, axis=1, keepdims=True)
        e = np.exp(s)
        self.probs = e / np.sum(e, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, y_oh):
        N = self.X.shape[0]
        dZ2 = (self.probs - y_oh)/N
        dW2 = self.H.T @ dZ2; db2 = np.sum(dZ2, axis=0)
        dH = dZ2 @ self.W2.T
        dZ1 = dH * (self.Z1 > 0).astype(float)
        dW1 = self.X.T @ dZ1; db1 = np.sum(dZ1, axis=0)
        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}
    
    def adam(self, grads, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for n in self.pnames:
            g = grads[n]
            self.m[n] = b1*self.m[n] + (1-b1)*g
            self.v[n] = b2*self.v[n] + (1-b2)*g**2
            mh = self.m[n]/(1-b1**self.t)
            vh = self.v[n]/(1-b2**self.t)
            setattr(self, n, getattr(self,n) - lr*mh/(np.sqrt(vh)+eps))

net = MNISTNet()
epochs, batch_size = 10, 128
N = X_train.shape[0]

print("Training MNIST MLP (784→256→10) with Adam")
print("═" * 50)

for epoch in range(epochs):
    idx = np.random.permutation(N)
    total_loss, correct = 0, 0
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bi = idx[start:end]
        X_b, y_b = X_train[bi], y_train[bi]
        y_oh = np.zeros((len(y_b), 10)); y_oh[np.arange(len(y_b)), y_b] = 1
        
        probs = net.forward(X_b)
        loss = -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-7, 1)), axis=1))
        total_loss += loss
        correct += np.sum(np.argmax(probs, axis=1) == y_b)
        
        grads = net.backward(y_oh)
        net.adam(grads)
    
    nb = max(1, N // batch_size)
    train_acc = correct / N * 100
    
    # Test accuracy
    test_probs = net.forward(X_test)
    test_acc = np.sum(np.argmax(test_probs, axis=1) == y_test) / len(y_test) * 100
    
    print(f"Epoch {epoch+1:2d}/10 | Loss: {total_loss/nb:.4f} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

print(f"\n🎯 Final Test Accuracy: {test_acc:.1f}%")
print(f"   Parameters: {net.W1.size + net.b1.size + net.W2.size + net.b2.size:,}")
`}
/>

## What Just Happened?

Let's unpack the key design decisions:

### Architecture Choice
- **784 → 256 → 10**: One hidden layer is enough for ~97% on MNIST. More layers help marginally but add complexity.
- **ReLU hidden activation**: Fast gradients, no vanishing gradient problem.
- **Softmax output + CCE loss**: The natural pairing for multi-class classification.

### Why Adam Over SGD?
On MNIST, both work. But Adam reaches good accuracy in fewer epochs because it adapts learning rates per-parameter. With vanilla SGD, you'd need careful learning rate scheduling to match Adam's performance.

### He Initialization
Without proper initialization, deep networks fail. He init (<InlineMath latex="W \sim \mathcal{N}(0, \sqrt{2/n_{in}})" />) keeps activation variance stable across layers when using ReLU.

## Visualizing Predictions

<PyRunner
  cellId="032-cell-3"
  defaultCode={`import numpy as np

# Show some predictions (using trained net from previous cell if available)
# This cell demonstrates the visualization concept

np.random.seed(42)
# Simulate predictions for demo
true_labels = np.array([3, 7, 1, 9, 0, 5, 2, 8])
pred_labels = np.array([3, 7, 1, 9, 0, 5, 2, 8])  # all correct for demo
confidences = np.array([0.97, 0.95, 0.99, 0.93, 0.98, 0.91, 0.96, 0.94])

print("Sample Predictions:")
print("─" * 40)
for true, pred, conf in zip(true_labels, pred_labels, confidences):
    status = "✅" if true == pred else "❌"
    bar = "█" * int(conf * 20)
    print(f"  True: {true} | Pred: {pred} | Conf: {conf:.2f} {bar} {status}")

print(f"\n💡 To visualize actual digit images, use matplotlib:")
print(f"   plt.imshow(X_test[i].reshape(28,28), cmap='gray')")
`}
/>

## Debugging Checklist

If your network isn't learning:

> [!IMPORTANT] Common Pitfalls
> 1. **Loss is NaN**: Check softmax numerical stability (subtract max). Clip log inputs.
> 2. **Loss doesn't decrease**: Learning rate too high or too low. Try 0.001 with Adam.
> 3. **Accuracy stuck at ~10%**: Network is predicting random. Check weight initialization and gradient shapes.
> 4. **Overfitting (train >> test)**: Add regularization (next chapter) or reduce hidden size.
> 5. **Gradient shapes mismatch**: Print every shape. `dW` must match `W` shape exactly.

## Next Steps

You've built a working neural network from scratch. From here:
- **Chapter 033**: Regularization (dropout, weight decay, batch norm)
- **Chapter 034**: Evaluation metrics beyond accuracy
- **Chapter 036**: CNNs — because MLPs waste spatial structure in images

<Quiz
  chapterSlug="032-mnist-from-scratch"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why do we normalize MNIST pixel values by dividing by 255?",
      options: [
        "To make images look better",
        "To convert uint8 to float64",
        "To keep activations and gradients in a reasonable numerical range, preventing explosion",
        "Because softmax requires inputs in [0,1]"
      ],
      correctIndex: 2,
      explanation: "Raw pixels (0-255) produce large activations through linear layers, which cause large gradients and numerical instability. Normalizing to [0,1] keeps everything in a well-behaved range.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How many trainable parameters does a 784→256→10 network have?",
      options: ["200,960", "203,530", "200,704", "203,786"],
      correctIndex: 1,
      explanation: "W1: 784×256 = 200,704. b1: 256. W2: 256×10 = 2,560. b2: 10. Total: 200,704 + 256 + 2,560 + 10 = 203,530.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "If your MNIST network achieves exactly 10% accuracy consistently, what's most likely wrong?",
      options: [
        "Learning rate is too high",
        "The network is predicting randomly — check initialization, gradient computation, and label encoding",
        "Need more hidden layers",
        "MNIST requires CNNs, MLPs can't solve it"
      ],
      correctIndex: 1,
      explanation: "10% = random chance for 10 classes. This means the network isn't learning at all. Common causes: wrong gradient signs, broken softmax, incorrect one-hot encoding, or weights initialized to zero.",
      randomize: true,
    }
  ]}
/>
