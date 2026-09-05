---
title: "Trigonometric, Hyperbolic & Set Operations"
slug: "019-ufunc-trigonometry-and-sets"
description: "Sin/cos/tan and their inverses, hyperbolic functions as activations, degree conversion, and set operations for data preprocessing — all with NN-relevant context."
track: "ufuncs"
order: 4
read_time: 18
code_time: 12
execution_timeout: 10
prerequisites: ["017-ufunc-math-operations"]
---

# Trigonometric, Hyperbolic & Set Operations

Trig functions appear in positional encodings, signal processing, and rotation matrices. Hyperbolic functions (especially `tanh`) are classic activation functions. Set operations handle label filtering, vocabulary management, and data deduplication.

## Trigonometric Functions

==All NumPy trig functions expect radians.== Use `deg2rad` / `rad2deg` for conversion.

### Basic Trig

<PyRunner cellId="019-cell-1" defaultCode={`import numpy as np

angles_rad = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi])
angles_deg = np.rad2deg(angles_rad)

print(f"Degrees: {angles_deg.round(1)}")
print(f"sin:     {np.sin(angles_rad).round(4)}")
print(f"cos:     {np.cos(angles_rad).round(4)}")
print(f"tan:     {np.tan(angles_rad[:5]).round(4)}")  # skip pi/2 (undefined)

# Pythagorean identity
print(f"\nsin²+cos²=1: {np.allclose(np.sin(angles_rad)**2 + np.cos(angles_rad)**2, 1)}")`}/>

### Inverse Trig & arctan2

<PyRunner cellId="019-cell-2" defaultCode={`import numpy as np

vals = np.array([0, 0.5, 0.7071, 1.0])
print(f"arcsin → degrees: {np.rad2deg(np.arcsin(vals)).round(1)}°")
print(f"arccos → degrees: {np.rad2deg(np.arccos(vals)).round(1)}°")

# arctan2(y, x) gives correct quadrant — essential for angle computation
x = np.array([1, 0, -1, 0, 1, -1])
y = np.array([0, 1, 0, -1, 1, -1])
angles = np.rad2deg(np.arctan2(y, x))
print(f"\narctan2 angles: {angles}°")
print(f"(Note: arctan2 handles all 4 quadrants correctly)")`}/>

<Callout type="info" title="Why arctan2 Instead of arctan?">
`arctan(y/x)` can't distinguish between quadrants I/III or II/IV. `arctan2(y, x)` takes separate arguments and returns the correct angle in [-π, π]. Always use `arctan2` for direction/angle calculations.
</Callout>

### Floating-Point Gotcha

<PyRunner cellId="019-cell-3" defaultCode={`import numpy as np

# sin(pi) should be 0, but...
print(f"np.sin(np.pi) = {np.sin(np.pi)}")       # ~1.22e-16, NOT 0!
print(f"Is zero? {np.sin(np.pi) == 0}")           # False
print(f"Close to zero? {np.isclose(np.sin(np.pi), 0)}")  # True

# Always use np.isclose for float comparisons
# Never use == for floating point results`}/>

### Degree Conversion Utilities

<PyRunner cellId="019-cell-4" defaultCode={`import numpy as np

deg = np.array([0, 30, 45, 60, 90, 180, 360])
rad = np.deg2rad(deg)
back = np.rad2deg(rad)

print(f"Degrees: {deg}")
print(f"Radians: {rad.round(4)}")
print(f"Round-trip OK: {np.allclose(deg, back)}")`}/>

---

## Hyperbolic Functions

### sinh, cosh, tanh

<PyRunner cellId="019-cell-5" defaultCode={`import numpy as np

x = np.linspace(-3, 3, 7)
print(f"x:    {x.round(2)}")
print(f"sinh: {np.sinh(x).round(4)}")
print(f"cosh: {np.cosh(x).round(4)}")
print(f"tanh: {np.tanh(x).round(4)}")

# Fundamental identity
print(f"\ncosh²-sinh²=1: {np.allclose(np.cosh(x)**2 - np.sinh(x)**2, 1)}")`}/>

### tanh as Activation Function

`tanh` was the default activation before ReLU. Still used in LSTM gates and output layers for bounded outputs.

<PyRunner cellId="019-cell-6" defaultCode={`import numpy as np

x = np.linspace(-5, 5, 11)
print("tanh activation curve:")
for xi in x:
    val = np.tanh(xi)
    bar_len = int((val + 1) * 15)  # map [-1,1] → [0,30]
    bar = "█" * bar_len
    print(f"  {xi:+5.1f} → {val:+.4f} {bar}")

print(f"\nKey properties:")
print(f"  Range: (-1, 1)")
print(f"  Zero-centered: tanh(0) = {np.tanh(0)}")
print(f"  Saturates at ±1 for |x| > 3")
print(f"  Derivative: 1 - tanh(x)² (easy to compute from output!)")`}/>

<Callout type="tip" title="tanh Derivative Shortcut">
If you already computed `h = tanh(z)`, then `dh/dz = 1 - h²`. No need to recompute tanh during backprop. This is why tanh was popular before ReLU.
</Callout>

### Inverse Hyperbolic

<PyRunner cellId="019-cell-7" defaultCode={`import numpy as np

v = np.array([0, 0.5, 0.9, -0.5])
print(f"arcsinh: {np.arcsinh(v).round(4)}")
print(f"arctanh: {np.arctanh(v).round(4)}")

# Domain warning
print(f"\narctanh(1.0) = {np.arctanh(1.0)}")   # inf
print(f"arctanh(1.5) = {np.arctanh(1.5)}")   # nan (outside domain)`}/>

---

## Set Operations

Essential for label management, vocabulary building, and data preprocessing.

### unique

<PyRunner cellId="019-cell-8" defaultCode={`import numpy as np

a = np.array([3, 1, 2, 3, 1, 4, 2, 5, 3])

print(f"unique:      {np.unique(a)}")

# With counts — histogram in one call
vals, counts = np.unique(a, return_counts=True)
print(f"values:      {vals}")
print(f"counts:      {counts}")

# With inverse indices — reconstruct original from unique values
vals, inv = np.unique(a, return_inverse=True)
print(f"inverse idx: {inv}")
print(f"reconstruct: {vals[inv]}")
print(f"matches original: {np.array_equal(vals[inv], a)}")`}/>

### Set Algebra

<PyRunner cellId="019-cell-9" defaultCode={`import numpy as np

train_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_labels = np.array([0, 2, 4, 6, 8, 10, 12])

print(f"Intersection (shared): {np.intersect1d(train_labels, test_labels)}")
print(f"Union (all):           {np.union1d(train_labels, test_labels)}")
print(f"In train only:         {np.setdiff1d(train_labels, test_labels)}")
print(f"In test only:          {np.setdiff1d(test_labels, train_labels)}")
print(f"Symmetric diff:        {np.setxor1d(train_labels, test_labels)}")`}/>

### Membership Testing with isin

<PyRunner cellId="019-cell-10" defaultCode={`import numpy as np

# Filter dataset by valid labels
all_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
valid_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

mask = np.isin(all_data, valid_labels)
print(f"Data:    {all_data}")
print(f"Mask:    {mask}")
print(f"Valid:   {all_data[mask]}")
print(f"Invalid: {all_data[~mask]}")

# Practical: remove NaN rows
data = np.array([[1, 2], [np.nan, 4], [5, 6], [7, np.nan]])
clean_mask = ~np.any(np.isnan(data), axis=1)
print(f"\nClean rows:\n{data[clean_mask]}")`}/>

<Quiz
  chapterSlug="019-ufunc-trigonometry-and-sets"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "NumPy trig functions (sin, cos, tan) expect angles in:",
      options: ["Radians", "Degrees", "Gradians", "Auto-detected units"],
      correctIndex: 0,
      explanation: "All NumPy trig ufuncs use radians. Use np.deg2rad() and np.rad2deg() for conversion.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.sin(np.pi))",
      options: ["~1.2e-16 (not exactly 0)", "0.0 exactly", "1.0", "Error"],
      correctIndex: 0,
      explanation: "π isn't exactly representable in float64, so sin(π) ≈ 1.22e-16. Always use np.isclose() for float comparisons.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "How should you compare floating-point results like sin(π) to zero?",
      options: ["Use np.isclose(value, 0) instead of ==", "Use == directly", "Round to integers first", "Cast to int"],
      correctIndex: 0,
      explanation: "Float arithmetic introduces tiny errors. np.isclose uses a tolerance to check approximate equality.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why prefer np.arctan2(y, x) over np.arctan(y/x)?",
      options: ["arctan2 correctly determines the quadrant from separate x,y signs", "arctan2 is faster", "arctan handles complex numbers", "arctan returns degrees"],
      correctIndex: 0,
      explanation: "arctan(y/x) loses sign info — can't distinguish Q1/Q3 or Q2/Q4. arctan2(y,x) returns the correct angle in [−π, π].",
      randomize: true,
    },
    {
      id: "q5",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.allclose(np.sin(np.array([0, np.pi/6, np.pi/2]))**2 + np.cos(np.array([0, np.pi/6, np.pi/2]))**2, 1))",
      options: ["True", "False", "Error", "nan"],
      correctIndex: 0,
      explanation: "Pythagorean identity: sin²θ + cos²θ = 1 for all θ. allclose confirms within floating-point tolerance.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "What is the range of np.tanh(x)?",
      options: ["(−1, 1)", "(0, 1)", "(−∞, ∞)", "[0, 1]"],
      correctIndex: 0,
      explanation: "tanh maps all reals to (−1, 1). It's zero-centered, which made it preferred over sigmoid before ReLU.",
      randomize: true,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "What is tanh(0)?",
      options: ["0", "1", "-1", "0.5"],
      correctIndex: 0,
      explanation: "tanh is zero-centered: tanh(0) = 0. This is one reason it was preferred over sigmoid (which has sigmoid(0) = 0.5).",
      randomize: true,
    },
    {
      id: "q8",
      type: "multiple-choice",
      prompt: "If h = tanh(z), what is dh/dz?",
      options: ["1 − h²", "h × (1 − h)", "cosh(z)", "1 / cosh²(z) only"],
      correctIndex: 0,
      explanation: "The derivative of tanh is 1 − tanh²(z). Since you already have h = tanh(z), compute 1 − h² without recomputing tanh.",
      randomize: true,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "At what |x| value does tanh effectively saturate (≈ ±1)?",
      options: ["|x| > 3", "|x| > 1", "|x| > 10", "Never saturates"],
      correctIndex: 0,
      explanation: "tanh(3) ≈ 0.995. Beyond |x| > 3, the output is essentially ±1 and gradients vanish (vanishing gradient problem).",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.intersect1d([1,2,3], [2,3,4]))",
      options: ["[2 3]", "[1 2 3 4]", "[1 4]", "[2]"],
      correctIndex: 0,
      explanation: "intersect1d returns sorted elements present in BOTH arrays: 2 and 3.",
      randomize: true,
    },
    {
      id: "q11",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.union1d([1,2,3], [2,3,4]))",
      options: ["[1 2 3 4]", "[2 3]", "[1 4]", "[1 2 3 2 3 4]"],
      correctIndex: 0,
      explanation: "union1d returns sorted unique elements from either array: {1, 2, 3, 4}.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\nprint(np.setdiff1d([1,2,3,4,5], [2,4]))",
      options: ["[1 3 5]", "[2 4]", "[1 2 3 4 5]", "[1 3]"],
      correctIndex: 0,
      explanation: "setdiff1d returns elements in the first array that are NOT in the second: 1, 3, 5.",
      randomize: true,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "What does np.unique(a, return_counts=True) return?",
      options: ["A tuple of (unique_values, count_of_each)", "Just the unique values", "Just the counts", "A dictionary"],
      correctIndex: 0,
      explanation: "return_counts=True gives both the sorted unique values and how many times each appears. Useful for histograms.",
      randomize: true,
    },
    {
      id: "q14",
      type: "multiple-choice",
      prompt: "What does np.unique(a, return_inverse=True) provide?",
      options: ["Indices such that unique_vals[inverse] reconstructs the original array", "Count of each unique value", "First occurrence index of each unique value", "Sorted version of the array"],
      correctIndex: 0,
      explanation: "Inverse indices map back: unique_vals[inverse_indices] == original_array. Essential for label encoding.",
      randomize: true,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What does np.isin(a, b) return?",
      options: ["A boolean mask: True where elements of a are found in b", "The intersection of a and b", "Indices of matching elements", "A count of matches"],
      correctIndex: 0,
      explanation: "isin produces a boolean array the same shape as a. Use it as a mask: a[np.isin(a, valid_labels)].",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.array([3, 1, 2, 3, 1])\nvals, inv = np.unique(a, return_inverse=True)\nprint(vals[inv])",
      options: ["[3 1 2 3 1]", "[1 2 3]", "[1 1 2 3 3]", "Error"],
      correctIndex: 0,
      explanation: "vals = [1,2,3], inv = [2,0,1,2,0]. vals[inv] reconstructs the original: [3,1,2,3,1].",
      randomize: true,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Where do trig functions appear in neural networks?",
      options: ["Positional encodings in Transformers, signal processing, rotation matrices", "Weight initialization", "Loss functions", "Batch normalization"],
      correctIndex: 0,
      explanation: "Transformer positional encodings use sin/cos at different frequencies. Also used in signal processing layers and geometric deep learning.",
      randomize: true,
    },
    {
      id: "q18",
      type: "multiple-choice",
      prompt: "np.arctanh(1.0) returns:",
      options: ["inf", "1.0", "0.0", "nan"],
      correctIndex: 0,
      explanation: "arctanh is defined on (−1, 1). At the boundary x=1, it approaches +∞. Beyond |x|>1, it returns nan.",
      randomize: true,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "In data preprocessing, how would you filter a dataset to only include valid labels?",
      options: ["mask = np.isin(labels, valid_labels); data = data[mask]", "data = np.intersect1d(data, valid_labels)", "data = data[labels in valid_labels]", "np.where(labels == valid_labels)"],
      correctIndex: 0,
      explanation: "isin creates a boolean mask for membership testing. Index with it to filter rows efficiently.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "What does np.setxor1d(a, b) return?",
      options: ["Elements in either a or b but NOT in both (symmetric difference)", "Elements in both a and b", "All elements from both", "Elements only in a"],
      correctIndex: 0,
      explanation: "Symmetric difference: (a ∪ b) − (a ∩ b). Useful for finding mismatched labels between train/test sets.",
      randomize: true,
    }
  ]}
/>
