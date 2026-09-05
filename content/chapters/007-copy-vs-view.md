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
      prompt: "What is the fundamental difference between a view and a copy in NumPy?",
      options: ["Views are faster to create", "A view shares memory with the original array; a copy allocates new independent memory", "Views have different dtypes", "Copies are always 2D"],
      correctIndex: 1,
      explanation: "Views are zero-cost references to existing data. Copies allocate new memory and duplicate every element. Modifying a view changes the original; modifying a copy does not.",
      randomize: true,
    },
    {
      id: "q2",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(5)\nb = a[1:4]\nb[0] = 99\nprint(a)",
      options: ["[ 0 99  2  3  4]", "[0 1 2 3 4]", "[99 99 99 99 99]", "Error"],
      correctIndex: 0,
      explanation: "b = a[1:4] is a basic slice, which returns a VIEW. b[0] refers to a[1]. Setting b[0]=99 modifies a[1] because they share memory.",
      randomize: false,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Which operation returns a COPY (not a view)?",
      options: ["a[1:5] (basic slice)", "a[[1, 3, 5]] (fancy indexing with an array of indices)", "a.T (transpose)", "a.reshape(2, 3) (when possible)"],
      correctIndex: 1,
      explanation: "Fancy indexing (indexing with an array or list of indices) ALWAYS returns a copy. Basic slicing, transpose, and reshape return views when the data layout allows it.",
      randomize: true,
    },
    {
      id: "q4",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(6).reshape(2, 3)\nb = a.copy()\nb[0, 0] = 99\nprint(a[0, 0])",
      options: ["0", "99", "6", "Error"],
      correctIndex: 0,
      explanation: ".copy() creates independent data. Modifying b does NOT affect a. This is the safe way to work with array subsets.",
      randomize: false,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "You slice a training set, normalize it in-place, and your test accuracy is suspiciously high. What probably happened?",
      options: ["Your model overfit", "The slice was a view — in-place normalization modified the test portion of the original array too", "NumPy has a bug", "The test set was too small"],
      correctIndex: 1,
      explanation: "Basic slicing returns a view sharing memory. X_train = X[:80] means X_train and X[:80] point to the same memory. Normalizing X_train in-place also normalizes the first 80 rows of X, leaking information into any subsequent split.",
      randomize: true,
    },
    {
      id: "q6",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(10)\nb = a[::2]\nprint(np.shares_memory(a, b))",
      options: ["True", "False", "None", "Error"],
      correctIndex: 0,
      explanation: "Step slicing (::2) still returns a view. The selected elements are at regular strides, so NumPy can represent them without copying.",
      randomize: false,
    },
    {
      id: "q7",
      type: "multiple-choice",
      prompt: "How can you definitively check if two arrays share memory?",
      options: ["a == b", "np.shares_memory(a, b)", "a.dtype == b.dtype", "len(a) == len(b)"],
      correctIndex: 1,
      explanation: "np.shares_memory(a, b) returns True if the arrays reference overlapping memory regions. This is the definitive test for view relationships.",
      randomize: true,
    },
    {
      id: "q8",
      type: "code-output",
      prompt: "What is the output?",
      code: "import numpy as np\na = np.arange(12).reshape(3, 4)\nb = a.T\nprint(b.shape, np.shares_memory(a, b))",
      options: ["(4, 3) True", "(4, 3) False", "(3, 4) True", "Error"],
      correctIndex: 0,
      explanation: "Transpose returns a view with swapped strides. Shape changes from (3,4) to (4,3), but no data is copied — memory is shared.",
      randomize: false,
    },
    {
      id: "q9",
      type: "multiple-choice",
      prompt: "When should you explicitly use .copy()?",
      options: ["Always, for safety", "When you need to modify an array without affecting the original, or before passing data to functions that modify in-place", "Never — views are always better", "Only for large arrays"],
      correctIndex: 1,
      explanation: "Defensive copying prevents subtle mutation bugs. Key situations: splitting train/test, applying in-place transforms, passing to external functions that might modify data.",
      randomize: true,
    },
    {
      id: "q10",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.array([1, 2, 3, 4, 5])\nb = a[[0, 2, 4]]\nb[0] = 99\nprint(a[0])",
      options: ["1", "99", "0", "Error"],
      correctIndex: 0,
      explanation: "Fancy indexing (a[[0,2,4]]) returns a COPY. Modifying b[0] does NOT affect a[0]. This is different from basic slicing.",
      randomize: false,
    },
    {
      id: "q11",
      type: "multiple-choice",
      prompt: "Does reshape return a view or a copy?",
      options: ["Always a copy", "View when possible (contiguous data compatible with new shape); copy when the data layout requires rearrangement", "Always a view", "Depends on dtype"],
      correctIndex: 1,
      explanation: "reshape tries to return a view. If the data is contiguous and the new shape is compatible with the stride pattern, you get a view. Otherwise, NumPy must copy.",
      randomize: true,
    },
    {
      id: "q12",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(8).reshape(2, 4)\nb = a[:, ::2]\nb[0, 0] = 99\nprint(a[0, 0])",
      options: ["99", "0", "Error", "None"],
      correctIndex: 0,
      explanation: "Column slicing with step (a[:, ::2]) returns a view when possible. b[0,0] maps to a[0,0], so modifying b modifies a.",
      randomize: false,
    },
    {
      id: "q13",
      type: "multiple-choice",
      prompt: "Why is understanding views critical for memory-efficient deep learning code?",
      options: ["Views are prettier in print output", "Views avoid allocating new memory for intermediate results — crucial when working with datasets that barely fit in RAM or GPU memory", "Views are faster to compute", "There's no memory difference"],
      correctIndex: 1,
      explanation: "A view of a 1GB array costs 0 extra bytes. A copy costs another 1GB. In batch processing pipelines, unnecessary copies can exhaust memory. Understanding views lets you write zero-copy data pipelines.",
      randomize: true,
    },
    {
      id: "q14",
      type: "code-output",
      prompt: "What is the result?",
      code: "import numpy as np\na = np.arange(6)\nb = a.reshape(2, 3)\nprint(b.base is a)",
      options: ["True", "False", "None", "Error"],
      correctIndex: 0,
      explanation: "reshape returned a view. b.base points to the original array a, confirming they share memory.",
      randomize: false,
    },
    {
      id: "q15",
      type: "multiple-choice",
      prompt: "What operation ALWAYS returns a copy?",
      options: ["Basic slicing (a[1:3])", "Fancy indexing (a[np.array([0,2,4])])", "Transpose (a.T)", "Reshape (a.reshape(...))"],
      correctIndex: 1,
      explanation: "Fancy indexing must copy because the selected elements may be scattered non-contiguously in memory. There's no way to represent arbitrary selections as a strided view.",
      randomize: true,
    },
    {
      id: "q16",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.ones((3, 3))\nb = a[1:]\nb[:] = 0\nprint(a[2, 0])",
      options: ["0.0", "1.0", "Error", "None"],
      correctIndex: 0,
      explanation: "b = a[1:] is a view of rows 1 and 2. b[:] = 0 sets those rows to zero in-place. a[2,0] was part of that view, so it's now 0.",
      randomize: false,
    },
    {
      id: "q17",
      type: "multiple-choice",
      prompt: "What is the performance cost of .copy()?",
      options: ["Zero", "O(n) time and O(n) memory — every element must be read from source and written to new memory", "O(1)", "Depends on dtype only"],
      correctIndex: 1,
      explanation: "Copying is linear in array size. For a 100M element float32 array, that's 400MB read + 400MB write. In tight loops, unnecessary copies dominate runtime.",
      randomize: true,
    },
    {
      id: "q18",
      type: "code-output",
      prompt: "What prints?",
      code: "import numpy as np\na = np.arange(10)\nb = a[3:8]\nc = b[1:3]\nc[0] = 77\nprint(a[4])",
      options: ["77", "4", "3", "Error"],
      correctIndex: 0,
      explanation: "c is a view of b, which is a view of a. All three share the same underlying memory. c[0] corresponds to b[1] which corresponds to a[4]. Chain of views.",
      randomize: false,
    },
    {
      id: "q19",
      type: "multiple-choice",
      prompt: "In PyTorch, what is the equivalent concept to NumPy views?",
      options: ["Clones", "Tensors share storage by default; .clone() creates an independent copy, similar to NumPy's .copy()", "Pointers", "References"],
      correctIndex: 1,
      explanation: "PyTorch tensors behave like NumPy views by default. Slicing shares storage. You must explicitly call .clone() to get independent data. Same mental model applies.",
      randomize: true,
    },
    {
      id: "q20",
      type: "multiple-choice",
      prompt: "Why does boolean indexing (a[a > 0]) return a copy instead of a view?",
      options: ["Performance optimization", "Selected elements are typically non-contiguous in memory, making a strided view representation impossible", "Design choice for safety", "Boolean masks are inherently slow"],
      correctIndex: 1,
      explanation: "Views require regular stride patterns (fixed step sizes between elements). Boolean selection picks arbitrary scattered elements that can't be expressed as a strided window into the original array.",
      randomize: true,
    }
  ]}
/>
