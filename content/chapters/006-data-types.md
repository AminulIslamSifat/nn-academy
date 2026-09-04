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
      prompt: "Why do ML frameworks default to float32 instead of float64?",
      options: [
        "Half the memory with negligible precision loss for training",
        "float64 is slower to compute",
        "float32 is more accurate",
        "GPU hardware only supports float32"
      ],
      correctIndex: 0,
      explanation: "float32 uses 4 bytes vs 8 for float64. The precision difference (~7 vs ~16 decimal digits) doesn't matter for gradient descent — but the memory savings are huge at scale.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What does np.uint8(255) + np.uint8(1) return?",
      code: "import numpy as np\nprint(np.uint8(255) + np.uint8(1))",
      options: ["0", "256", "OverflowError", "-1"],
      correctIndex: 0,
      explanation: "uint8 wraps around silently: 255 + 1 = 256 mod 256 = 0. No error, no warning. This is why you never accumulate gradients in integer types.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does .astype(np.int32) do to np.array([1.7, 2.9])?",
      options: [
        "Truncates to [1, 2]",
        "Rounds to [2, 3]",
        "Raises an error",
        "Returns [1.7, 2.9] unchanged"
      ],
      correctIndex: 0,
      explanation: "astype truncates toward zero — it chops off the decimal. Use np.round() first if you want proper rounding.",
      randomize: true,
    }
  ]}
/>
