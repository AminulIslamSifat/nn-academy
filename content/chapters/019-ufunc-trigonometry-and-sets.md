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
      type: "code-output",
      prompt: "np.sin(np.pi) returns:",
      code: "import numpy as np\nprint(np.sin(np.pi))",
      options: ["~1.2e-16 (not exactly 0)", "0.0 exactly", "1.0", "Error"],
      correctIndex: 0,
      explanation: "Floating-point precision: π isn't exact in float64, so sin(π) ≈ 1.22e-16. Always use np.isclose() for float comparisons.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "NumPy trig functions expect:",
      options: ["Radians", "Degrees", "Gradians", "Auto-detected"],
      correctIndex: 0,
      explanation: "All trig ufuncs use radians. Convert with np.deg2rad() / np.rad2deg().",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Range of np.tanh(x)?",
      options: ["(-1, 1)", "(0, 1)", "(-∞, ∞)", "[0, 1]"],
      correctIndex: 0,
      explanation: "tanh maps all real numbers to the open interval (-1, 1). Zero-centered, which is why it was preferred over sigmoid.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "np.intersect1d([1,2,3], [2,3,4]) returns:",
      code: "import numpy as np\nprint(np.intersect1d([1,2,3], [2,3,4]))",
      options: ["[2 3]", "[1 2 3 4]", "[1 4]", "[2]"],
      correctIndex: 0,
      explanation: "Intersection returns elements present in BOTH arrays, sorted.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does np.unique(a, return_inverse=True) provide?",
      options: ["Indices to reconstruct original array from unique values", "Count of each unique value", "First occurrence index", "Nothing useful"],
      correctIndex: 0,
      explanation: "inverse indices satisfy: unique_values[inverse_indices] == original_array. Useful for label encoding.",
      randomize: true,
    },
    {
      id: "q6",
      type: "multiple-choice",
      prompt: "Why prefer arctan2(y, x) over arctan(y/x)?",
      options: ["Correctly determines quadrant from separate x,y signs", "Faster computation", "Handles complex numbers", "Returns degrees instead of radians"],
      correctIndex: 0,
      explanation: "arctan(y/x) loses sign information (can't distinguish Q1/Q3 or Q2/Q4). arctan2 uses both arguments independently to return the correct angle in [-π, π].",
      randomize: true,
    }
  ]}
/>
