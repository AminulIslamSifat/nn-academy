---
title: "Copy vs View"
slug: "007-copy-vs-view"
description: "The most dangerous thing in NumPy isn't a bug — it's a feature. Views share memory, and modifying one silently changes the other."
track: "numpy-foundations"
order: 4
read_time: 14
code_time: 10
execution_timeout: 5
prerequisites: ["002-indexing-and-slicing"]
---

# Copy vs View

I need you to really pay attention here, because this is the thing that will make you stare at your screen at 3 AM wondering why your training data got corrupted.

When you slice a NumPy array, you don't get a new array. You get a **view** — a window into the same memory. Change the view, you change the original. *Silently.*

## The Trap

<PyRunner
  cellId="007-cell-1"
  defaultCode={`import numpy as np

# You have some training data
data = np.array([10, 20, 30, 40, 50])

# "Extract" a portion for processing
subset = data[1:4]  # looks like a new array, right?
print(f"data:   {data}")
print(f"subset: {subset}")

# Modify the subset
subset[0] = 999

# Wait... what happened to data?!
print(f"\nAfter subset[0] = 999:")
print(f"data:   {data}")    # [10, 999, 30, 40, 50] — IT CHANGED
print(f"subset: {subset}")

print(f"\nThey share memory: {np.shares_memory(data, subset)}")
`}
/>

Yeah. `subset` wasn't a copy. It was a view — just a different way of looking at the same bytes in RAM. When you wrote `999` into `subset[0]`, you wrote it directly into `data`'s memory.

This is *by design*. It's what makes slicing instant — no data copying, just a new header pointing to the same buffer. But it's also what makes it dangerous.

## The Fix: .copy()

When you need an independent array, say so explicitly:

<PyRunner
  cellId="007-cell-2"
  defaultCode={`import numpy as np

data = np.array([10, 20, 30, 40, 50])

# Explicit copy — independent memory
subset = data[1:4].copy()

print(f"Shares memory: {np.shares_memory(data, subset)}")  # False!

subset[0] = 999
print(f"data:   {data}")    # [10, 20, 30, 40, 50] — untouched ✓
print(f"subset: {subset}")  # [999, 30, 40]
`}
/>

## How to Tell If You Have a View

Every array has a `.base` attribute. If it's `None`, the array owns its own memory. If it points to another array, you're looking at a view:

<PyRunner
  cellId="007-cell-3"
  defaultCode={`import numpy as np

original = np.arange(12).reshape(3, 4)
slice_view = original[:, 1]     # column 1 — view
explicit_copy = original.copy() # full copy
partial_copy = original[0].copy() # row copy

print(f"slice_view.base is original: {slice_view.base is original}")
print(f"explicit_copy.base is None:  {explicit_copy.base is None}")
print(f"partial_copy.base is None:   {partial_copy.base is None}")

# .flags.owndata is another way to check
print(f"\nslice_view owns data: {slice_view.flags.owndata}")
print(f"explicit_copy owns data: {explicit_copy.flags.owndata}")
`}
/>

## The Asymmetry That Catches Everyone

Here's the thing that makes this extra tricky: **basic slicing returns views, but fancy indexing returns copies.**

<PyRunner
  cellId="007-cell-4"
  defaultCode={`import numpy as np

a = np.arange(10)

# Basic slicing → VIEW
view = a[2:5]
print(f"a[2:5] shares memory: {np.shares_memory(a, view)}")      # True

# Fancy indexing (list of indices) → COPY
copy = a[[2, 3, 4]]
print(f"a[[2,3,4]] shares memory: {np.shares_memory(a, copy)}")  # False

# Boolean mask → COPY
mask_copy = a[a > 5]
print(f"a[a>5] shares memory: {np.shares_memory(a, mask_copy)}") # False

# So this modifies original:
a[2:5] = [99, 99, 99]
print(f"\nAfter a[2:5] = [99,99,99]: {a}")

# But this does NOT:
a[[2, 3, 4]] = [88, 88, 88]  # modifies a COPY, then throws it away
print(f"After a[[2,3,4]] = [88,88,88]: {a}")  # unchanged!
`}
/>

Wait — actually that last one is wrong. Let me re-check...

<PyRunner
  cellId="007-cell-5"
  defaultCode={`import numpy as np

a = np.arange(10)

# Actually, fancy indexing on the LEFT side of assignment DOES modify original
# It's fancy indexing on the RIGHT side that returns a copy
a[[2, 3, 4]] = [88, 88, 88]
print(f"Fancy index assignment: {a}")  # DOES modify! 

# The copy behavior is only when READING
b = a[[2, 3, 4]]  # this is a copy
b[0] = 999
print(f"Modifying the copy doesn't affect a: {a}")  # unchanged
`}
/>

Okay good — I almost taught you something wrong there. Fancy indexing *reads* return copies, but fancy indexing *writes* still modify the original. The distinction matters.

<Callout type="danger" title="The 3 AM Debugging Scenario">
You split your dataset into train/test. You normalize the training set. But you sliced instead of copied — so you just normalized the test set too. Your evaluation metrics are meaningless. Welcome to the club.
</Callout>

## Why Views Exist: Performance

Views aren't a bug — they're a feature. Creating a view is O(1), zero memory allocation. For large arrays, this matters enormously:

<PyRunner
  cellId="007-cell-6"
  defaultCode={`import numpy as np
import time

big = np.random.randn(10_000_000)

# View: instant, zero memory
start = time.perf_counter()
view = big[::2]  # every other element — 5M elements
view_time = time.perf_counter() - start

# Copy: allocates 40 MB of new memory
start = time.perf_counter()
copy = big[::2].copy()
copy_time = time.perf_counter() - start

print(f"View:  {view_time*1e6:.1f} μs, memory={view.nbytes/1e6:.0f} MB (shared)")
print(f"Copy:  {copy_time*1e3:.1f} ms, memory={copy.nbytes/1e6:.0f} MB (new)")
print(f"Speed difference: {copy_time/view_time:.0f}x")
`}
/>

## Practical: Safe Train/Test Split

Here's how you do it right:

<PyRunner
  cellId="007-cell-7"
  defaultCode={`import numpy as np

rng = np.random.default_rng(42)

# Full dataset
X = rng.standard_normal((100, 5))
y = rng.integers(0, 2, 100)

# WRONG way (views — normalization leaks!)
# X_train = X[:80]
# X_test = X[80:]
# X_train -= X_train.mean(axis=0)  # this also shifts X_test!

# RIGHT way (copies — independent)
X_train = X[:80].copy()
X_test = X[80:].copy()
y_train = y[:80].copy()
y_test = y[80:].copy()

# Now safe to normalize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std  # uses TRAIN stats, but doesn't modify train

print(f"Train mean after norm: {X_train.mean(axis=0).round(6)}")  # ≈ 0
print(f"Test mean after norm:  {X_test.mean(axis=0).round(4)}")   # not 0 (different data)
print(f"Original X untouched:  {X.mean():.4f}")  # original distribution preserved
`}
/>

See? `.copy()` isn't optional ceremony — it's the difference between valid experiments and silent data leakage.

<Quiz
  chapterSlug="007-copy-vs-view"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "You slice a training set, normalize it, and your test accuracy is suspiciously high. What probably happened?",
      options: [
        "The slice was a view — normalization modified the test set too",
        "NumPy has a bug in normalization",
        "Your model overfit",
        "The test set was too small"
      ],
      correctIndex: 0,
      explanation: "Basic slicing returns a view sharing memory with the original. In-place operations on the 'train' slice also modified the 'test' slice. Always .copy() before splitting.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "After: a = np.arange(5); b = a[1:3]; b[0] = 99 — what is a?",
      code: "import numpy as np\na = np.arange(5)\nb = a[1:3]\nb[0] = 99\nprint(a)",
      options: ["[ 0 99  2  3  4]", "[0 1 2 3 4]", "[ 0  1 99  3  4]", "Error"],
      correctIndex: 0,
      explanation: "b is a view of a[1:3]. b[0] is a[1]. Setting it to 99 modifies a in place.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Which operation returns a COPY (not a view)?",
      options: [
        "a[[1, 3, 5]] (fancy indexing)",
        "a[1:5] (basic slice)",
        "a.reshape(2, 3) (usually)",
        "a.T (transpose)"
      ],
      correctIndex: 0,
      explanation: "Fancy indexing (with lists/arrays) always returns a copy. Basic slicing, reshape, and transpose return views when possible.",
      randomize: true,
    }
  ]}
/>
