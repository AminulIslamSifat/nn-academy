---
title: "Math ufuncs: Arithmetic, Rounding & Logarithms"
slug: "017-ufunc-math-operations"
description: "Element-wise arithmetic, five rounding modes, and logarithmic/exponential functions — including numerical stability patterns essential for deep learning."
track: "ufuncs"
order: 2
read_time: 20
code_time: 15
execution_timeout: 10
prerequisites: ["016-ufunc-intro-and-creation"]
---

# Math ufuncs: Arithmetic, Rounding & Logarithms

These are the workhorse ufuncs you'll use in every neural network implementation. Understanding division variants, rounding edge cases, and log-space computation prevents subtle bugs in loss functions and training loops.

## Arithmetic Operations

Operators on arrays dispatch to ufuncs: `a + b` calls `np.add(a, b)`.

<PyRunner cellId="017-cell-1" defaultCode={`import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print(f"add:      {np.add(a, b)}")
print(f"subtract: {np.subtract(a, b)}")
print(f"multiply: {np.multiply(a, b)}")
print(f"divide:   {np.divide(a, b)}")
print(f"power:    {np.power(b, 2)}")
print(f"mod:      {np.mod(a, 3)}")
print(f"\na+b == np.add(a,b): {np.array_equal(a+b, np.add(a,b))}")`}/>

### Division Variants (Know the Difference!)

Python's `/` and NumPy's division have different behaviors with negatives.

<PyRunner cellId="017-cell-2" defaultCode={`import numpy as np

a = np.array([7, -7, 8, -8])
b = np.array([3, 3, 3, 3])

print("Division variants:")
print(f"  true_divide:  {np.true_divide(a, b)}")   # Python /
print(f"  floor_divide: {np.floor_divide(a, b)}")  # rounds toward -inf
print(f"  remainder:    {np.remainder(a, b)}")       # a % b
print(f"  divmod:       {np.divmod(a, b)}")          # (quotient, remainder)

print(f"\n⚠️  floor_divide(-7, 3) = {np.floor_divide(-7, 3)}")
print(f"   NOT -2 (which is truncation toward zero)")
print(f"   It's -3 (rounds toward NEGATIVE infinity)")`}/>

<Callout type="warning" title="Floor vs Truncation">
`floor_divide` rounds toward −∞. Python's `int()` truncates toward zero. For negative numbers these give different results. This matters in index calculations and grid operations.
</Callout>

---

## Five Rounding Modes

Each rounding mode handles negative numbers differently. Know which one your algorithm needs.

<PyRunner cellId="017-cell-3" defaultCode={`import numpy as np

vals = np.array([2.7, 2.3, -2.7, -2.3, 2.5, -2.5, 3.5])

print(f"{'Value':>6} {'around':>7} {'floor':>6} {'ceil':>6} {'trunc':>6} {'fix':>6}")
print("─" * 45)
for v in vals:
    print(f"{v:6.1f} {np.around(v):7.0f} {np.floor(v):6.0f} {np.ceil(v):6.0f} {np.trunc(v):6.0f} {np.fix(v):6.0f}")

print(f"\nKey differences:")
print(f"  around: round-half-to-even (banker's rounding)")
print(f"  floor:  always toward -∞")
print(f"  ceil:   always toward +∞")
print(f"  trunc:  always toward zero")
print(f"  fix:    same as trunc for scalars")`}/>

<Callout type="info" title="Banker's Rounding">
`np.around(2.5) = 2.0`, not 3.0. Round-half-to-even reduces statistical bias when aggregating many rounded values. This is the IEEE 754 standard.
</Callout>

### Rounding with Decimal Places

<PyRunner cellId="017-cell-4" defaultCode={`import numpy as np

a = np.array([3.14159, 2.71828, 1.41421])
print(f"Original: {a}")
print(f"dec=2:    {np.around(a, 2)}")
print(f"dec=1:    {np.around(a, 1)}")

# Negative decimals round to tens, hundreds, etc.
big = np.array([15, 25, 35, 45, 55])
print(f"\n{big} → dec=-1: {np.around(big, -1)}")`}/>

---

## Logarithmic & Exponential Functions

These are critical for softmax, cross-entropy loss, and probability computations.

### Basic Logs and Exp

<PyRunner cellId="017-cell-5" defaultCode={`import numpy as np

a = np.array([1, 2, 4, 8, 16, 100], dtype=float)
print(f"ln(a):    {np.log(a).round(4)}")
print(f"log2(a):  {np.log2(a).round(4)}")
print(f"log10(a): {np.log10(a).round(4)}")
print(f"exp(ln(a)) == a: {np.allclose(np.exp(np.log(a)), a)}")

print(f"\n⚠️  np.log(0) = {np.log(0)}")
print(f"   np.log(-1) = {np.log(-1)}")`}/>

### Numerical Stability: log1p and expm1

When working with tiny values near zero, standard `log(1+x)` and `exp(x)-1` lose precision due to floating-point cancellation.

<PyRunner cellId="017-cell-6" defaultCode={`import numpy as np

tiny = 1e-15

print("Precision loss with small values:")
print(f"  log(1+tiny):   {np.log(1 + tiny):.20e}")    # loses precision!
print(f"  log1p(tiny):   {np.log1p(tiny):.20e}")     # accurate
print(f"  exp(tiny)-1:   {np.exp(tiny) - 1:.20e}")    # loses precision!
print(f"  expm1(tiny):   {np.expm1(tiny):.20e}")     # accurate

print(f"\nWhy? When tiny ≈ 1e-15:")
print(f"  1 + tiny rounds to 1.0 in float64")
print(f"  log(1.0) = 0.0 ← wrong!")
print(f"  log1p computes log(1+x) without forming 1+x first")`}/>

### Stable Softmax (Critical Pattern)

The naive softmax overflows for large logits. The shifted version is mathematically identical but numerically safe.

<PyRunner cellId="017-cell-7" defaultCode={`import numpy as np

logits = np.array([1000, 1001, 1002])

# NAIVE: exp(1000) overflows to inf
naive = np.exp(logits) / np.exp(logits).sum()
print(f"Naive softmax:  {naive}")  # nan or inf!

# STABLE: subtract max first
shifted = logits - logits.max()
e = np.exp(shifted)
stable = e / e.sum()
print(f"Stable softmax: {stable}")

# Why does this work?
# exp(x - c) / sum(exp(x - c)) = exp(x) / sum(exp(x))
# Same ratios, no overflow
print(f"\nProof: exp(x-max)/sum = exp(x)/sum (same result, no overflow)")`}/>

<Callout type="danger" title="Cross-Entropy Safety">
`log(0) = -inf` crashes training. Always clip probabilities before taking log: `np.clip(probs, 1e-12, 1.0)`. Or better: use the log-sum-exp trick to compute log-softmax directly without ever forming raw probabilities.
</Callout>

### Log-Sum-Exp Trick

<PyRunner cellId="017-cell-8" defaultCode={`import numpy as np

def stable_log_softmax(logits):
    """Compute log(softmax(x)) without overflow or underflow."""
    c = logits.max(axis=-1, keepdims=True)
    return logits - c - np.log(np.sum(np.exp(logits - c), axis=-1, keepdims=True))

logits = np.array([[1000, 1001, 1002], [-1000, -999, -998]])
log_probs = stable_log_softmax(logits)
probs = np.exp(log_probs)

print(f"Log-probs: {log_probs.round(4)}")
print(f"Probs:     {probs.round(6)}")
print(f"Sum to 1:  {probs.sum(axis=1)}")
print(f"No inf/nan: {np.isfinite(log_probs).all()}")`}/>

<Quiz
  chapterSlug="017-ufunc-math-operations"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Is a + b equivalent to np.add(a, b) for NumPy arrays?",
      options: ["Yes — arithmetic operators dispatch to ufuncs internally", "No — they behave differently", "Only for integer arrays", "Only for 1D arrays"],
      correctIndex: 0,
      explanation: "Operators like +, -, *, / on ndarrays call the corresponding ufunc (add, subtract, multiply, divide).",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.floor_divide(-7, 3))",
      options: ["-3", "-2", "-1", "-4"],
      correctIndex: 0,
      explanation: "floor_divide rounds toward −∞. −7/3 = −2.33… → floor is −3. NOT −2 (that would be truncation toward zero).",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the difference between floor_divide and truncation toward zero?",
      options: ["floor_divide rounds toward −∞; truncation rounds toward 0. They differ for negative numbers", "They are identical", "floor_divide is slower", "truncation only works on integers"],
      correctIndex: 0,
      explanation: "For −7/3: floor_divide gives −3 (toward −∞), truncation gives −2 (toward 0). This matters in index calculations.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.around(2.5))",
      options: ["2.0", "3.0", "2", "Error"],
      correctIndex: 0,
      explanation: "NumPy uses banker's rounding (round-half-to-even). 2.5 rounds to the nearest even number: 2.0.",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.around(3.5))",
      options: ["4.0", "3.0", "4", "3"],
      correctIndex: 0,
      explanation: "Banker's rounding: 3.5 rounds to nearest even → 4.0. Both 2.5→2 and 3.5→4 follow round-half-to-even.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Which rounding mode always rounds toward negative infinity?",
      options: ["np.floor", "np.ceil", "np.trunc", "np.around"],
      correctIndex: 0,
      explanation: "floor always rounds toward −∞. ceil toward +∞. trunc toward 0. around uses round-half-to-even.",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What does np.trunc(-2.7) return?",
      options: ["-2.0", "-3.0", "-2", "2.0"],
      correctIndex: 0,
      explanation: "trunc removes the fractional part, rounding toward zero. −2.7 → −2.0 (not −3.0).",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.log(0))",
      options: ["-inf", "0", "nan", "Error"],
      correctIndex: 0,
      explanation: "log(0) = −∞. NumPy returns -inf with a RuntimeWarning. This is why you must clip probabilities before taking log.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Why use np.log1p(x) instead of np.log(1 + x)?",
      options: ["Maintains precision when x is very small (near zero)", "It's faster", "They produce identical results always", "log1p handles negative inputs better"],
      correctIndex: 0,
      explanation: "When x ≈ 1e-15, 1+x rounds to 1.0 in float64, so log(1+x) = 0 (wrong). log1p computes log(1+x) without forming the sum first.",
      randomize: true,
    },
    {
      id: "q10",
      type: "multiple-choice",
      prompt: "Why use np.expm1(x) instead of np.exp(x) - 1?",
      options: ["Maintains precision when x is very small", "It's faster", "They're identical", "expm1 avoids overflow"],
      correctIndex: 0,
      explanation: "For tiny x, exp(x) ≈ 1.0 due to floating-point, so exp(x)-1 = 0 (wrong). expm1 computes exp(x)-1 accurately.",
      randomize: true,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Why do we subtract max(logits) before exp in softmax?",
      options: ["Prevents exp overflow while preserving the probability ratios", "Makes computation faster", "Changes the output distribution", "It's optional optimization"],
      correctIndex: 0,
      explanation: "exp(x−c)/Σexp(x−c) = exp(x)/Σexp(x). Mathematically identical, but x−c ≤ 0 so exp never overflows.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What happens with naive softmax on large logits?",
      code: "import numpy as np\nlogits = np.array([1000, 1001, 1002])\nresult = np.exp(logits) / np.exp(logits).sum()\nprint(result)",
      options: ["[nan nan nan] or [inf inf inf]", "Correct probabilities", "[0.33 0.33 0.33]", "Error"],
      correctIndex: 0,
      explanation: "exp(1000) overflows to inf in float64. inf/inf = nan. Always use the stable version with max subtraction.",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What does np.remainder(-7, 3) return?",
      options: ["2", "-1", "1", "-2"],
      correctIndex: 0,
      explanation: "remainder follows floor_divide: −7 = 3×(−3) + 2. The remainder is always non-negative when divisor is positive.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "In cross-entropy loss, why must you clip probabilities before taking log?",
      options: ["Because log(0) = −inf which produces nan loss and crashes training", "To speed up computation", "To normalize probabilities", "It's not necessary"],
      correctIndex: 0,
      explanation: "If any predicted probability is exactly 0, log(0) = −inf, making the loss inf/nan. Clip to a small epsilon like 1e-12.",
      randomize: true,
    },
    {
      id: "q15",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nbig = np.array([15, 25, 35, 45, 55])\nprint(np.around(big, -1))",
      options: ["[20 20 40 40 60]", "[10 20 30 40 50]", "[15 25 35 45 55]", "Error"],
      correctIndex: 0,
      explanation: "Negative decimals round to tens. Banker's rounding: 15→20, 25→20 (even), 35→40, 45→40 (even), 55→60.",
      randomize: true,
    },
    {
      id: "q16",
      type: "multiple-choice",
      prompt: "What is the log-sum-exp trick used for?",
      options: ["Computing log(softmax(x)) without overflow or underflow", "Speeding up matrix multiplication", "Converting between log bases", "Normalizing arrays"],
      correctIndex: 0,
      explanation: "log_softmax = x − max(x) − log(Σexp(x − max(x))). Stays in log-space throughout, avoiding exp overflow and log(0).",
      randomize: true,
    },
    {
      id: "q17",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([1, 2, 4, 8], dtype=float)\nprint(np.allclose(np.exp(np.log(a)), a))",
      options: ["True", "False", "Error", "nan"],
      correctIndex: 0,
      explanation: "exp and log are inverses. exp(log(a)) recovers the original values (within floating-point tolerance).",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "Which function computes element-wise natural logarithm?",
      options: ["np.log", "np.ln", "np.natural_log", "np.log_e"],
      correctIndex: 0,
      explanation: "np.log is the natural logarithm (base e). np.log2 and np.log10 are base-2 and base-10 respectively.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "np.divmod(a, b) returns:",
      options: ["A tuple of (floor_divide(a,b), remainder(a,b))", "The quotient only", "The remainder only", "A single array with both values interleaved"],
      correctIndex: 0,
      explanation: "divmod returns both the floor quotient and remainder as a tuple, computed in one pass for efficiency.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "In a neural network, where are log/exp ufuncs most critical?",
      options: ["Softmax, cross-entropy loss, and probability computations", "Weight initialization", "Matrix multiplication", "Data loading"],
      correctIndex: 0,
      explanation: "Softmax uses exp, cross-entropy uses log, and both need numerical stability tricks. These are the most common sources of nan/inf bugs in training.",
      randomize: true,
    }
  ]}
/>
