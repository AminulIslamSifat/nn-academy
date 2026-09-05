---
title: "Aggregation & Calculus ufuncs"
slug: "018-ufunc-aggregation-and-calculus"
description: "Sum, product, cumulative operations, discrete derivatives, and numerical gradient checking — the building blocks for loss computation and backpropagation verification."
track: "ufuncs"
order: 3
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["017-ufunc-math-operations"]
---

# Aggregation & Calculus ufuncs

Aggregation reduces arrays to scalars or smaller arrays. Discrete calculus (`diff`, `gradient`) approximates derivatives numerically. Together they form the backbone of loss computation and gradient checking in neural networks.

## Sum Along Axes

Understanding `axis` is fundamental. Every loss function uses it.

<PyRunner cellId="018-cell-1" defaultCode={`import numpy as np

m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Total sum:    {m.sum()}")           # scalar: 45
print(f"axis=0 (cols): {m.sum(axis=0)}")     # [12, 15, 18]
print(f"axis=1 (rows): {m.sum(axis=1)}")     # [6, 15, 24]

# keepdims preserves dimensionality — critical for broadcasting in gradients
print(f"\nWithout keepdims: {m.sum(axis=1).shape}")       # (3,)
print(f"With keepdims:    {m.sum(axis=1, keepdims=True).shape}")  # (3, 1)`}/>

<Callout type="tip" title="keepdims Matters">
When computing batch losses, `keepdims=True` preserves shape for correct broadcasting with weights/biases during backprop. Without it, you'll get silent shape mismatches.
</Callout>

## Handling NaN Values

Real data has missing values. Standard aggregations propagate NaN; `nan*` variants ignore them.

<PyRunner cellId="018-cell-2" defaultCode={`import numpy as np

data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

print(f"sum:    {np.sum(data)}")      # nan (propagates!)
print(f"nansum: {np.nansum(data)}")   # 9.0 (ignores NaN)
print(f"mean:   {np.mean(data)}")     # nan
print(f"nanmean:{np.nanmean(data)}")  # 3.0

# Count non-NaN elements
print(f"\nValid count: {np.count_nonzero(~np.isnan(data))}")
print(f"isnan mask:  {np.isnan(data)}")`}/>

## Cumulative Operations

Running sums and products — useful for sequence models and probability chains.

<PyRunner cellId="018-cell-3" defaultCode={`import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(f"cumsum:  {np.cumsum(a)}")    # [1, 3, 6, 10, 15]
print(f"cumprod: {np.cumprod(a)}")   # [1, 2, 6, 24, 120] = factorials!

# 2D cumulative along axis
m = np.array([[1, 2], [3, 4], [5, 6]])
print(f"\ncumsum axis=0:\n{np.cumsum(m, axis=0)}")
print(f"cumsum axis=1:\n{np.cumsum(m, axis=1)}")`}/>

### Practical: Batch MSE Loss

<PyRunner cellId="018-cell-4" defaultCode={`import numpy as np

np.random.seed(42)
pred = np.random.randn(32, 10)
target = np.random.randn(32, 10)

# Per-sample MSE: sum squared errors across features (axis=1)
per_sample_mse = np.sum((pred - target)**2, axis=1)
batch_mse = per_sample_mse.mean()

print(f"Per-sample MSE (first 5): {per_sample_mse[:5].round(4)}")
print(f"Batch mean MSE:           {batch_mse:.4f}")
print(f"Shape check: {per_sample_mse.shape} → scalar {batch_mse}")`}/>

---

## Product Operations

Products grow fast. Use log-space for large products.

<PyRunner cellId="018-cell-5" defaultCode={`import numpy as np

print(f"prod([1,2,3,4,5]): {np.prod([1,2,3,4,5])}")  # 120

# NaN handling
data = np.array([2.0, np.nan, 4.0])
print(f"prod:    {np.prod(data)}")      # nan
print(f"nanprod: {np.nanprod(data)}")   # 8.0

# Chain probabilities (common in HMMs, seq models)
probs = np.array([0.9, 0.8, 0.7, 0.6])
print(f"\nChain probs: {np.cumprod(probs).round(4)}")
print(f"Final joint: {np.prod(probs):.4f}")`}/>

<Callout type="warning" title="Overflow Prevention">
Products of many small numbers underflow; products of large numbers overflow. Solution: work in log-space.

`prod(a) = exp(sum(log(a)))`

This converts multiplication to addition, avoiding both overflow and underflow.
</Callout>

### Log-Space Products

<PyRunner cellId="018-cell-6" defaultCode={`import numpy as np

# Many small probabilities → underflow in direct product
probs = np.full(1000, 0.99)
print(f"Direct prod: {np.prod(probs)}")  # 0.0 (underflow!)

# Log-space: stable
log_prod = np.exp(np.sum(np.log(probs)))
print(f"Log-space:   {log_prod:.6e}")

# Even better: stay in log-space as long as possible
log_probs_sum = np.sum(np.log(probs))
print(f"Log-prod (stay in log): {log_probs_sum:.4f}")
print(f"exp of that: {np.exp(log_probs_sum):.6e}")`}/>

---

## Discrete Derivatives

### np.diff: Consecutive Differences

<PyRunner cellId="018-cell-7" defaultCode={`import numpy as np

a = np.array([1, 3, 6, 10, 15])
print(f"diff:     {np.diff(a)}")        # [2, 3, 4, 5]
print(f"diff n=2: {np.diff(a, n=2)}")   # [1, 1, 1] (second derivative)

# Output length is always N-1
print(f"Input len: {len(a)}, diff len: {len(np.diff(a))}")`}/>

### np.gradient: Central Differences

More accurate than `diff` — uses central differences for interior points.

<PyRunner cellId="018-cell-8" defaultCode={`import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Numerical derivative via gradient
dy_num = np.gradient(y, x)
dy_exact = np.cos(x)

max_error = np.abs(dy_num - dy_exact).max()
print(f"d/dx sin(x) ≈ cos(x)")
print(f"Max error: {max_error:.8f}")
print(f"First 5 numerical: {dy_num[:5].round(4)}")
print(f"First 5 exact:     {dy_exact[:5].round(4)}")`}/>

### Numerical Gradient Checking

==This is how you verify backprop implementations.== Compare analytical gradients against finite differences.

<PyRunner cellId="018-cell-9" defaultCode={`import numpy as np

def f(x):
    return x**3 - 2*x**2 + x

def num_grad(f, x, eps=1e-7):
    """Central difference approximation."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Analytical: f'(x) = 3x² - 4x + 1
x_vals = np.array([0.0, 1.0, 2.0, 3.0])
for x in x_vals:
    analytical = 3*x**2 - 4*x + 1
    numerical = num_grad(f, x)
    rel_error = abs(analytical - numerical) / (abs(analytical) + 1e-8)
    status = "✅" if rel_error < 1e-5 else "❌"
    print(f"x={x:.1f}: analytic={analytical:+.6f}, numeric={numerical:+.6f}, err={rel_error:.2e} {status}")`}/>

<Callout type="info" title="Gradient Check Protocol">
1. Use a tiny batch (N=4) and small network
2. Check every parameter matrix/vector
3. Relative error < 1e-5 = pass, > 1e-3 = bug
4. Turn OFF during actual training (O(n) forward passes per parameter)
</Callout>

---

## LCM & GCD

Less common in DL but useful for grid/tile calculations and scheduling.

<PyRunner cellId="018-cell-10" defaultCode={`import numpy as np

a = np.array([12, 18, 24])
b = np.array([8, 12, 36])
print(f"GCD: {np.gcd(a, b)}")   # [4, 6, 12]
print(f"LCM: {np.lcm(a, b)}")   # [24, 36, 72]

# reduce: fold over entire array
nums = np.array([12, 18, 24, 36])
print(f"\nGCD of all: {np.gcd.reduce(nums)}")  # gcd(gcd(gcd(12,18),24),36) = 6
print(f"LCM of all: {np.lcm.reduce(nums)}")  # 72`}/>

<Quiz
  chapterSlug="018-ufunc-aggregation-and-calculus"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nm = np.array([[1,2],[3,4]])\nprint(m.sum(axis=0))",
      options: ["[4 6]", "[3 7]", "10", "[[1 2]]"],
      correctIndex: 0,
      explanation: "axis=0 collapses rows: [1+3, 2+4] = [4, 6]. Column-wise sum.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nm = np.array([[1,2],[3,4]])\nprint(m.sum(axis=1))",
      options: ["[3 7]", "[4 6]", "10", "[[3] [7]]"],
      correctIndex: 0,
      explanation: "axis=1 collapses columns: [1+2, 3+4] = [3, 7]. Row-wise sum.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does keepdims=True do in aggregation functions?",
      options: ["Preserves the reduced axis as size 1 instead of removing it", "Keeps the original array unchanged", "Prevents NaN propagation", "Makes the operation faster"],
      correctIndex: 0,
      explanation: "keepdims=True keeps the collapsed dimension as size 1 (e.g., (3,) → (3,1)). Critical for broadcasting in backprop.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "np.sum([1, np.nan, 3]) returns:",
      options: ["nan", "4.0", "0", "Error"],
      correctIndex: 0,
      explanation: "Standard aggregations propagate NaN. Use np.nansum() to skip NaN values.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\ndata = np.array([1.0, np.nan, 3.0, np.nan, 5.0])\nprint(np.nanmean(data))",
      options: ["3.0", "nan", "5.0", "9.0"],
      correctIndex: 0,
      explanation: "nanmean ignores NaN values: (1+3+5)/3 = 3.0.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.cumsum([1,2,3,4]))",
      options: ["[ 1  3  6 10]", "[1 2 3 4]", "10", "[1 3 6]"],
      correctIndex: 0,
      explanation: "Running sum: 1, 1+2=3, 3+3=6, 6+4=10.",
      randomize: true,
    },
    {
      id: "q7",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.cumprod([1,2,3,4]))",
      options: ["[ 1  2  6 24]", "[1 2 3 4]", "24", "[ 1  3  6 10]"],
      correctIndex: 0,
      explanation: "Running product: 1, 1×2=2, 2×3=6, 6×4=24. These are factorials!",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "How do you compute large products without overflow/underflow?",
      options: ["Work in log-space: exp(sum(log(a)))", "Use float128", "Split into smaller chunks", "It's impossible"],
      correctIndex: 0,
      explanation: "log converts multiplication to addition: prod(a) = exp(Σlog(a)). Stay in log-space as long as possible.",
      randomize: true,
    },
    {
      id: "q9",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.diff([1,4,9,16]))",
      options: ["[3 5 7]", "[1 4 9 16]", "[4 9 16]", "[1 2 3 4]"],
      correctIndex: 0,
      explanation: "Consecutive differences: 4−1=3, 9−4=5, 16−9=7. Output length is always N−1.",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "What does np.diff(a, n=2) compute?",
      options: ["Second-order differences (diff of diff)", "Differences with step size 2", "Two separate diff arrays", "Cumulative differences"],
      correctIndex: 0,
      explanation: "n=2 applies diff twice: first differences, then differences of those. Approximates the second derivative.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "How does np.gradient differ from np.diff?",
      options: ["gradient uses central differences for interior points; diff uses forward differences only", "They are identical", "gradient is slower but less accurate", "diff handles 2D arrays better"],
      correctIndex: 0,
      explanation: "gradient uses (f(x+h)−f(x−h))/2h at interior points (more accurate), forward/backward at edges. diff only uses forward differences.",
      randomize: true,
    },
    {
      id: "q12",
      type: "shape-prediction",
      prompt: "m has shape (5, 8). What is m.sum(axis=1, keepdims=True).shape?",
      options: ["(5, 1)", "(5,)", "(1, 8)", "(8,)"],
      correctIndex: 0,
      explanation: "axis=1 collapses columns. keepdims=True preserves it as size 1: (5, 8) → (5, 1).",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "In batch MSE loss, why do you sum over axis=1 then mean over axis=0?",
      options: ["Sum over features per sample (axis=1), then average across samples (axis=0)", "Sum over samples, then average features", "Both axes give the same result", "You should sum over both axes at once"],
      correctIndex: 0,
      explanation: "Each sample's error is summed across its features (axis=1). Then the batch mean averages across all samples.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "Why does np.prod of many small probabilities often return 0.0?",
      options: ["Floating-point underflow — the product becomes smaller than the smallest representable float", "NumPy caps products at 1.0", "Integer overflow", "It's a bug"],
      correctIndex: 0,
      explanation: "Multiplying many values <1 produces exponentially smaller numbers that eventually underflow to 0.0 in float64.",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What is numerical gradient checking used for?",
      options: ["Verifying that analytical backprop gradients are correct by comparing to finite differences", "Computing gradients when backprop isn't available", "Speeding up training", "Regularizing gradients"],
      correctIndex: 0,
      explanation: "Compare analytical gradient vs (f(x+ε)−f(x−ε))/2ε. Relative error < 1e-5 means your backprop is correct.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What relative error threshold indicates a correct gradient implementation?",
      options: ["< 1e-5", "< 1.0", "< 0.1", "Exactly 0"],
      correctIndex: 0,
      explanation: "Relative error < 1e-5 = pass. Between 1e-5 and 1e-3 = suspicious. > 1e-3 = likely a bug in backprop.",
      randomize: true,
    },
    {
      id: "q17",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nnums = np.array([12, 18, 24, 36])\nprint(np.gcd.reduce(nums))",
      options: ["6", "12", "36", "72"],
      correctIndex: 0,
      explanation: "GCD folded across array: gcd(12,18)=6, gcd(6,24)=6, gcd(6,36)=6.",
      randomize: true,
    },
    {
      id: "q18",
      type: "shape-prediction",
      prompt: "a has shape (10,). What is np.diff(a).shape?",
      options: ["(9,)", "(10,)", "(8,)", "(11,)"],
      correctIndex: 0,
      explanation: "diff computes consecutive differences, producing N−1 elements from N inputs.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "Should you leave gradient checking on during actual training?",
      options: ["No — it requires O(n) forward passes per parameter and is extremely slow", "Yes — always verify gradients", "Only every 100 epochs", "Only for large networks"],
      correctIndex: 0,
      explanation: "Gradient checking is a debugging tool. It's far too expensive for training. Use it once to verify, then turn off.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "When computing chain probabilities in sequence models, why use cumprod?",
      options: ["To track the running joint probability at each step without recomputing from scratch", "cumprod is faster than prod", "To normalize probabilities", "To convert to log-space"],
      correctIndex: 0,
      explanation: "cumprod gives P(x₁), P(x₁,x₂), P(x₁,x₂,x₃)… incrementally. Useful for monitoring probability decay in HMMs and language models.",
      randomize: true,
    }
  ]}
/>
