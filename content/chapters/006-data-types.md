---
title: "Data Types"
slug: "006-data-types"
description: "NumPy's type system isn't just about correctness — it's about memory, speed, and not silently destroying your gradients."
track: "numpy-foundations"
order: 3
read_time: 14
code_time: 10
execution_timeout: 5
prerequisites: ["001-arrays-and-shapes"]
---

# Data Types

Okay so you know arrays have shapes. But every array also has a **dtype** — a single data type that every element shares. This isn't some boring metadata field. It's the reason NumPy is fast, and it's also the reason your neural network can silently produce garbage if you pick wrong.

Let me show you what I mean.

## The Type Family

NumPy has way more types than Python's built-in `int` and `float`. Each one uses a specific number of bytes:

<PyRunner
  cellId="006-cell-1"
  defaultCode={`import numpy as np

# NumPy picks a default based on your input
a = np.array([1, 2, 3])          # integers → int64 (on 64-bit)
b = np.array([1.0, 2.0, 3.0])    # floats → float64
c = np.array([True, False, True]) # bools → bool_

print(f"Integer array:  dtype={a.dtype}, {a.itemsize} bytes per element")
print(f"Float array:    dtype={b.dtype}, {b.itemsize} bytes per element")
print(f"Boolean array:  dtype={c.dtype}, {c.itemsize} byte per element")

# But you can force a specific type
small = np.array([1, 2, 3], dtype=np.int8)
print(f"\nForced int8:    dtype={small.dtype}, {small.itemsize} byte per element")
print(f"That's 8x less memory than int64 for the same values!")
`}
/>

Here's the full family at a glance:

| Category | Types | Bytes | When you'd use it |
|----------|-------|-------|-------------------|
| Boolean | `bool_` | 1 | Masks, dropout flags |
| Integer | `int8` → `int64` | 1–8 | Indices, labels, counts |
| Unsigned | `uint8` → `uint64` | 1–8 | Image pixels (0–255) |
| Float | `float16`, `float32`, `float64` | 2–8 | **Weights, activations, gradients** |
| Complex | `complex64`, `complex128` | 8–16 | Signal processing, Fourier stuff |

Notice I bolded the float row? That's where your neural network lives. Weights are `float32` or `float64`. Almost never anything else.

## Why dtype Matters for Memory

Let's say you're training on ImageNet — 1.2 million images, each 224×224×3 pixels. How much RAM does that take?

<PyRunner
  cellId="006-cell-2"
  defaultCode={`import numpy as np

n_images = 1_200_000
pixels_per_image = 224 * 224 * 3

# If you naively load everything as float64...
f64_bytes = n_images * pixels_per_image * 8  # 8 bytes per float64
print(f"float64: {f64_bytes / 1e9:.1f} GB 😱")

# As float32 (what PyTorch/TensorFlow actually use)...
f32_bytes = n_images * pixels_per_image * 4
print(f"float32: {f32_bytes / 1e9:.1f} GB")

# As uint8 (raw pixel values are 0-255 anyway)...
u8_bytes = n_images * pixels_per_image * 1
print(f"uint8:   {u8_bytes / 1e9:.1f} GB ← store as this, convert to float32 per batch")

print(f"\nLesson: dtype choice = {f64_bytes/u8_bytes:.0f}x memory difference")
`}
/>

See? Same data, 8x memory difference just from picking the right type. This is why every ML framework defaults to `float32` — it's the sweet spot between precision and memory. `float64` is almost never worth it for training.

## Type Promotion: When Types Mix

When you combine arrays of different dtypes, NumPy automatically promotes to the "safer" type. The hierarchy goes: `bool < int < uint < float < complex`.

<PyRunner
  cellId="006-cell-3"
  defaultCode={`import numpy as np

int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.5, 2.5, 3.5], dtype=np.float32)

result = int_arr + float_arr
print(f"int32 + float32 → {result.dtype}")  # float32 wins

# You can check what NumPy would promote to without actually doing it
print(f"\nnp.result_type(np.int8, np.float32) = {np.result_type(np.int8, np.float32)}")
print(f"np.result_type(np.float32, np.float64) = {np.result_type(np.float32, np.float64)}")

# In neural networks, this matters when mixing integer labels with float weights
labels = np.array([0, 1, 2], dtype=np.int64)    # class labels
weights = np.random.randn(3).astype(np.float32)  # model weights
combined = labels + weights
print(f"\nint64 labels + float32 weights → {combined.dtype}")
`}
/>

## The Silent Killer: Integer Overflow

This one bites everyone eventually. Integer types in NumPy **silently wrap around** when they overflow. No error, no warning — just wrong numbers.

<PyRunner
  cellId="006-cell-4"
  defaultCode={`import numpy as np

# uint8 can hold 0-255. What happens at 256?
print(f"np.uint8(255) + 1 = {np.uint8(255) + np.uint8(1)}")  # 0?! 

# int8 can hold -128 to 127
print(f"np.int8(127) + 1 = {np.int8(127) + np.int8(1)}")    # -128?!

# This is INSANE if it happens in your gradient computation
# Imagine your loss counter overflowing mid-training 💀
loss_sum = np.array(0, dtype=np.int32)
for i in range(100_000):
    loss_sum = loss_sum + np.int32(50_000)
print(f"\nSum of 100k × 50k as int32: {loss_sum}")
print(f"Expected: {100_000 * 50_000}")
print(f"Overflowed by: {100_000 * 50_000 - int(loss_sum):,}")
`}
/>

<Callout type="danger" title="Rule for Neural Networks">
Always use `float32` or `float64` for anything involving gradients, losses, or weights. Integer types are fine for indices and labels, but never for computation that accumulates.
</Callout>

## Converting Types with .astype()

`.astype()` creates a copy with the new type. It truncates floats to ints (doesn't round!):

<PyRunner
  cellId="006-cell-5"
  defaultCode={`import numpy as np

probabilities = np.array([0.9, 0.1, 0.8, 0.3, 0.6])

# Convert probabilities to binary predictions (threshold at 0.5)
predictions = (probabilities > 0.5).astype(np.int32)
print(f"Probabilities: {probabilities}")
print(f"Predictions:   {predictions}")

# float64 → float32 (common before feeding to a model)
weights_f64 = np.random.randn(1000, 1000)
weights_f32 = weights_f64.astype(np.float32)
print(f"\nMemory saved: {weights_f64.nbytes/1e6:.1f} MB → {weights_f32.nbytes/1e6:.1f} MB")

# Watch out: astype truncates, doesn't round!
values = np.array([1.7, 2.9, 3.1])
print(f"\n{values} → int: {values.astype(np.int32)}")  # [1, 2, 3] not [2, 3, 3]!
print(f"If you want rounding: {np.round(values).astype(np.int32)}")
`}
/>

## What ML Frameworks Actually Use

<PyRunner
  cellId="006-cell-6"
  defaultCode={`import numpy as np

# Simulating a tiny neural network layer
rng = np.random.default_rng(42)

# Inputs: float32 (standard for training)
X = rng.standard_normal((32, 784)).astype(np.float32)  # batch of 32 MNIST images

# Weights: float32
W = rng.standard_normal((784, 128)).astype(np.float32) * 0.01
b = np.zeros(128, dtype=np.float32)

# Forward pass
Z = X @ W + b
print(f"Input:   {X.dtype}, {X.nbytes/1e3:.0f} KB")
print(f"Weights: {W.dtype}, {W.nbytes/1e3:.0f} KB")
print(f"Output:  {Z.dtype}, {Z.nbytes/1e3:.0f} KB")

# Labels: int64 (class indices)
y = rng.integers(0, 10, size=32).astype(np.int64)
print(f"Labels:  {y.dtype} (just indices, no computation needed)")

print(f"\nTotal for one batch: {(X.nbytes + W.nbytes + Z.nbytes)/1e3:.0f} KB")
print(f"With float64 it would be: {(X.nbytes + W.nbytes + Z.nbytes)*2/1e3:.0f} KB")
`}
/>

See the pattern? **float32 for computation, int64 for labels**. That's the convention across PyTorch, TensorFlow, JAX — all of them. Now you know why.

<Quiz
  chapterSlug="006-data-types"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is a dtype in NumPy?",
      options: ["A Python class", "A specification of how each element is stored in memory — its binary format and byte size", "A shape descriptor", "A performance setting"],
      correctIndex: 1,
      explanation: "dtype determines the binary representation of every element. float32 = 4 bytes IEEE 754, int64 = 8 bytes signed integer, etc. All elements in an array share the same dtype.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3])\nprint(a.dtype)",
      options: ["int64", "float64", "int32", "object"],
      correctIndex: 0,
      explanation: "NumPy infers dtype from input values. Plain Python integers become int64 on 64-bit systems. Pass dtype=np.float32 to override.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why does deep learning almost always use float32 instead of float64?",
      options: ["float64 is less precise", "float32 uses half the memory and computes faster on GPUs, with sufficient precision (~7 digits) for gradient-based optimization", "float64 causes overflow", "NumPy doesn't support float64"],
      correctIndex: 1,
      explanation: "float32 = 4 bytes, float64 = 8 bytes. For a model with 100M parameters, that's 400MB vs 800MB just for weights. GPU tensor cores are optimized for FP32. The extra precision of FP64 doesn't improve training.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1.5, 2.7, 3.2])\nprint(a.astype(np.int32))",
      options: ["[1 2 3]", "[2 3 3]", "[1.5 2.7 3.2]", "Error"],
      correctIndex: 0,
      explanation: "astype(int) TRUNCATES toward zero — it does NOT round. 1.5 becomes 1, 2.7 becomes 2, 3.2 becomes 3. If you want rounding, use np.round(values).astype(int).",
      randomize: false,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What happens when uint8 overflows past 255?",
      options: ["Raises OverflowError", "Silently wraps around to 0 — no error, no warning", "Clamps at 255", "Promotes to int16"],
      correctIndex: 1,
      explanation: "NumPy integer types silently wrap: uint8(255) + 1 = 0. This is extremely dangerous in neural network code where accumulated losses or gradient sums can silently produce wrong values.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\nprint(np.uint8(255) + np.uint8(1))",
      options: ["0", "256", "255", "Error"],
      correctIndex: 0,
      explanation: "uint8 range is 0-255. 255+1=256 wraps to 0. Silent overflow — one of the most dangerous bugs in numerical code.",
      randomize: false,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "When you add an int32 array and a float32 array, what dtype is the result?",
      options: ["int32", "float64 — NumPy promotes to the safer type that can represent both", "object", "Error — can't mix types"],
      correctIndex: 1,
      explanation: "NumPy promotes to the wider type. int + float → float64 (on most platforms). The hierarchy is roughly: bool < int < uint < float < complex.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3], dtype=np.float32)\nprint(a.nbytes)",
      options: ["12", "24", "3", "4"],
      correctIndex: 0,
      explanation: "3 elements times 4 bytes (float32) = 12 bytes. nbytes = size times itemsize.",
      randomize: false,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "Which dtype should you use for class labels (0-9)?",
      options: ["float64", "int64 or int32 — they're discrete indices, not continuous values", "float32", "complex128"],
      correctIndex: 1,
      explanation: "Labels are categorical indices. Using float wastes memory and can cause subtle bugs. Integer types clearly communicate this is a category, not a measurement.",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([True, False, True])\nprint(a.dtype, a.nbytes)",
      options: ["bool 3", "int64 24", "bool 24", "int8 3"],
      correctIndex: 0,
      explanation: "Boolean arrays use 1 byte per element (not 1 bit). 3 elements = 3 bytes. Used for masks, dropout flags, and boolean indexing.",
      randomize: false,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "What does .astype() do?",
      options: ["Changes the variable's Python type", "Returns a NEW array with elements converted to the specified dtype; original is unchanged", "Modifies the array in place", "Changes the shape"],
      correctIndex: 1,
      explanation: "astype always creates a copy. a.astype(np.float32) returns a new float32 array; a itself is unchanged. This costs memory — be aware when converting large arrays.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3], dtype=np.float64)\nb = a.astype(np.float32)\nprint(b.dtype, b is a)",
      options: ["float32 False", "float64 True", "float32 True", "float64 False"],
      correctIndex: 0,
      explanation: "astype returns a NEW array (b is not a). Even converting float64 to float32 creates a copy because the binary representation differs.",
      randomize: false,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Why is uint8 the right dtype for raw image pixels?",
      options: ["It's the fastest", "Pixel values are 0-255 by definition; uint8 stores exactly this range in 1 byte per pixel, using 4x less memory than float32", "GPUs require uint8", "It prevents overflow"],
      correctIndex: 1,
      explanation: "Images store pixel intensities as 0-255. uint8 matches this perfectly. Convert to float32 only when feeding into the network (divide by 255.0). Store on disk as uint8.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.zeros(3, dtype=np.int32)\nprint(a.sum(), a.dtype)",
      options: ["0 int32", "0.0 float64", "3 int32", "0 float32"],
      correctIndex: 0,
      explanation: "Zeros sum to 0. dtype is preserved through reduction operations like sum(). Result stays int32.",
      randomize: false,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What is type promotion?",
      options: ["Converting Python types to NumPy", "NumPy's automatic rules for determining the output dtype when combining arrays of different dtypes", "Upgrading hardware", "Changing array shapes"],
      correctIndex: 1,
      explanation: "When you compute int32_array + float32_array, NumPy promotes to float64. The hierarchy is roughly: bool < int < uint < float < complex. You can check with np.result_type().",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\nprint(np.result_type(np.int8, np.float32))",
      options: ["float64", "int8", "float32", "int32"],
      correctIndex: 0,
      explanation: "np.result_type predicts the output dtype without computing. int8 + float32 promotes to float64 on most platforms. Useful for debugging dtype issues before they cause silent bugs.",
      randomize: false,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "Why should you never use integer types for loss accumulation or gradient computation?",
      options: ["Integers are slower", "Integer overflow wraps silently, producing completely wrong values with no error; floats have much larger range and won't silently corrupt your training", "Integers can't represent negative numbers", "NumPy doesn't support integer math"],
      correctIndex: 1,
      explanation: "A loss sum that overflows int32 wraps to a negative number. Your optimizer then moves weights in the WRONG direction. Float32 has range up to ~3.4e38 — virtually impossible to overflow in practice.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1.0, 2.0, 3.0])\nprint(a.itemsize)",
      options: ["8", "4", "3", "1"],
      correctIndex: 0,
      explanation: "Default float dtype is float64 = 8 bytes per element. itemsize reports bytes per element, not total array size.",
      randomize: false,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "In mixed-precision training, why keep loss computation and gradient accumulation in float32 even when forward pass uses float16?",
      options: ["Convention", "float16 has limited range (max ~65504) and precision; accumulating many small gradients in float16 loses information and can overflow or underflow", "GPUs require it", "float16 is deprecated"],
      correctIndex: 1,
      explanation: "FP16 can only represent ~3.3 decimal digits. Summing thousands of small gradients loses precision. FP32 master weights prevent this while still getting FP16 speed in the forward pass.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "You load 1 million images (224x224x3 pixels each). How much RAM does this take as uint8 vs float64?",
      options: ["Same amount", "uint8: ~143 GB, float64: ~1.1 TB — an 8x difference from dtype alone", "float64 is smaller", "uint8: ~1 GB, float64: ~8 GB"],
      correctIndex: 1,
      explanation: "1M x 224 x 224 x 3 = 150M pixels. uint8: 150M bytes = ~143 GB. float64: 150M x 8 = ~1.2 TB. This is why you store images as uint8 and convert to float32 per-batch during training.",
      randomize: true,
    }
  ]}
/>
