#!/usr/bin/env python3
"""Generate ufunc chapters 016-027."""
import os
CONTENT_DIR = "/home/sifat/hdd/projects/numpy-nn-academy/src/content/chapters"

def w(slug, content):
    p = os.path.join(CONTENT_DIR, f"{slug}.mdx")
    if os.path.exists(p): print(f"  SKIP {slug}"); return
    open(p,'w').write(content); print(f"  ✓ {slug}")

w("016-ufunc-intro", open('/dev/stdin').read() if False else """---
title: "Universal Functions (ufunc) Intro"
slug: "016-ufunc-intro"
description: "What ufuncs are, why they're fast, and how NumPy's element-wise engine works."
track: "ufuncs"
order: 1
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["003-vectorization"]
---

# Universal Functions (ufunc) Intro

A **ufunc** operates on ndarrays element-by-element in compiled C loops.

## What Makes a ufunc Special

<PyRunner cellId="016-cell-1" defaultCode={`import numpy as np, time
a = np.arange(1_000_000, dtype=np.float64)

start = time.perf_counter()
r1 = [x**2 for x in a]
t1 = time.perf_counter() - start

start = time.perf_counter()
r2 = np.square(a)
t2 = time.perf_counter() - start

print(f"Python loop: {t1*1000:.1f} ms")
print(f"np.square:   {t2*1000:.3f} ms")
print(f"Speedup:     {t1/t2:.0f}x")`}/>

## Inspecting ufuncs

<PyRunner cellId="016-cell-2" defaultCode={`import numpy as np
print(f"np.add.nin:      {np.add.nin}")
print(f"np.add.nout:     {np.add.nout}")
print(f"np.add.identity: {np.add.identity}")
ufuncs = [n for n in dir(np) if isinstance(getattr(np,n,None), np.ufunc)]
print(f"Total ufuncs: {len(ufuncs)}")`}/>

## The out Parameter

<PyRunner cellId="016-cell-3" defaultCode={`import numpy as np
a = np.arange(5, dtype=np.float64)
result = np.empty_like(a)
np.multiply(a, 2, out=result)
print(f"Zero-alloc result: {result}")
np.add(a, 10, out=a)
print(f"In-place add: {a}")`}/>

<Quiz chapterSlug="016-ufunc-intro" questions={[
  {id:"q1",type:"multiple-choice",prompt:"What is np.add.identity?",options:["0","1","None","-1"],correctIndex:0,explanation:"Identity for addition is 0: x+0=x.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"Why are ufuncs faster than Python loops?",options:["Pre-compiled C loops without per-element Python overhead","They use GPU","They skip errors","Less memory"],correctIndex:0,explanation:"ufuncs run tight C loops over contiguous memory.",randomize:true},
  {id:"q3",type:"code-output",prompt:"What does np.add.nin return?",code:"import numpy as np\nprint(np.add.nin)",options:["2","1","0","Error"],correctIndex:0,explanation:"nin=number of inputs. add takes 2.",randomize:true}
]}/>
""")

w("017-ufunc-create-function", """---
title: "Creating Custom ufuncs"
slug: "017-ufunc-create-function"
description: "Build your own ufuncs with frompyfunc and vectorize. Understand the tradeoffs."
track: "ufuncs"
order: 2
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Creating Custom ufuncs

## np.frompyfunc

<PyRunner cellId="017-cell-1" defaultCode={`import numpy as np
my_ufunc = np.frompyfunc(lambda x: x**2 + 2*x + 1, 1, 1)
a = np.array([1, 2, 3, 4, 5])
result = my_ufunc(a)
print(f"Result: {result}")
print(f"dtype: {result.dtype}")  # object!
print(f"As float: {result.astype(np.float64)}")`}/>

## np.vectorize

<PyRunner cellId="017-cell-2" defaultCode={`import numpy as np
@np.vectorize
def classify(x):
    if x < 0: return -1
    elif x < 5: return 0
    else: return 1
print(classify(np.array([-3, 0, 2, 5, 8])))`}/>

<Callout type="warning" title="Not Actually Faster">
vectorize and frompyfunc still call Python per element. For real speed, use native ufuncs or numba.
</Callout>

## Speed Comparison

<PyRunner cellId="017-cell-3" defaultCode={`import numpy as np, time
n = 100_000
a = np.random.randn(n)
vec = np.vectorize(lambda x: x**2 if x > 0 else 0)
for name, fn in [("loop", lambda a: [x**2 if x>0 else 0 for x in a]),
                 ("vectorize", vec),
                 ("native", lambda a: np.where(a>0, a**2, 0))]:
    t = time.perf_counter(); fn(a); print(f"{name:12s}: {(time.perf_counter()-t)*1000:.1f} ms")`}/>

<Quiz chapterSlug="017-ufunc-create-function" questions={[
  {id:"q1",type:"multiple-choice",prompt:"Does np.vectorize make functions faster?",options:["No — convenience wrapper only","Yes significantly","Only small arrays","Only with numba"],correctIndex:0,explanation:"vectorize is a fancy for-loop, not a speed optimization.",randomize:true},
  {id:"q2",type:"code-output",prompt:"What dtype does frompyfunc return?",code:"import numpy as np\nf=np.frompyfunc(lambda x:x*2,1,1)\nprint(f(np.array([1,2,3])).dtype)",options:["object","float64","int64","Error"],correctIndex:0,explanation:"frompyfunc always returns object dtype.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"For real performance with custom ops use:",options:["Native ufuncs or numba","np.vectorize","frompyfunc","List comprehensions"],correctIndex:0,explanation:"Native ufuncs run in C. Numba JIT-compiles to machine code.",randomize:true}
]}/>
""")

w("018-ufunc-arithmetic", """---
title: "Arithmetic ufuncs"
slug: "018-ufunc-arithmetic"
description: "add, subtract, multiply, divide, power, mod — element-wise math operations."
track: "ufuncs"
order: 3
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Arithmetic ufuncs

## Basic Operations

<PyRunner cellId="018-cell-1" defaultCode={`import numpy as np
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])
print(f"add:      {np.add(a, b)}")
print(f"subtract: {np.subtract(a, b)}")
print(f"multiply: {np.multiply(a, b)}")
print(f"divide:   {np.divide(a, b)}")
print(f"power:    {np.power(b, 2)}")
print(f"mod:      {np.mod(a, 3)}")
print(f"a+b == np.add(a,b): {np.array_equal(a+b, np.add(a,b))}")`}/>

## Division Variants

<PyRunner cellId="018-cell-2" defaultCode={`import numpy as np
a = np.array([7, -7, 8, -8])
b = np.array([3, 3, 3, 3])
print(f"true_divide:  {np.true_divide(a, b)}")
print(f"floor_divide: {np.floor_divide(a, b)}")
print(f"remainder:    {np.remainder(a, b)}")
print(f"divmod:       {np.divmod(a, b)}")`}/>

## In-Place with out=

<PyRunner cellId="018-cell-3" defaultCode={`import numpy as np
a = np.arange(5, dtype=np.float64)
np.add(a, 10, out=a)
print(f"After +10: {a}")
np.multiply(a, 2, out=a)
print(f"After ×2:  {a}")`}/>

<Quiz chapterSlug="018-ufunc-arithmetic" questions={[
  {id:"q1",type:"code-output",prompt:"np.floor_divide(-7, 3) returns:",code:"import numpy as np\nprint(np.floor_divide(-7,3))",options:["-3","-2","-1","-4"],correctIndex:0,explanation:"Floor rounds toward -inf: -7/3=-2.33→-3.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"Is a+b equivalent to np.add(a,b)?",options:["Yes, operators dispatch to ufuncs","No, different behavior","Only integers","Only 1D"],correctIndex:0,explanation:"Operators on arrays call the corresponding ufunc.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"np.divmod(10,3) returns:",options:["(array(3), array(1))","(3.33,1)","(3,1)","Error"],correctIndex:0,explanation:"divmod returns (quotient, remainder) as arrays.",randomize:true}
]}/>
""")

w("019-ufunc-rounding", """---
title: "Rounding ufuncs"
slug: "019-ufunc-rounding"
description: "around, floor, ceil, trunc, fix — five rounding modes with different negative behavior."
track: "ufuncs"
order: 4
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["018-ufunc-arithmetic"]
---

# Rounding ufuncs

## Five Rounding Modes

<PyRunner cellId="019-cell-1" defaultCode={`import numpy as np
vals = np.array([2.7, 2.3, -2.7, -2.3, 2.5, -2.5])
print(f"{'Val':>5s} {'around':>7s} {'floor':>6s} {'ceil':>6s} {'trunc':>6s}")
for v in vals:
    print(f"{v:5.1f} {np.around(v):7.0f} {np.floor(v):6.0f} {np.ceil(v):6.0f} {np.trunc(v):6.0f}")`}/>

<Callout type="info" title="Banker's Rounding">
np.around uses round-half-to-even: 2.5→2, 3.5→4. Reduces statistical bias.
</Callout>

## around with Decimals

<PyRunner cellId="019-cell-2" defaultCode={`import numpy as np
a = np.array([3.14159, 2.71828, 1.41421])
print(f"dec=2: {np.around(a, 2)}")
print(f"dec=1: {np.around(a, 1)}")
print(f"dec=-1: {np.around(np.array([15,25,35]), -1)}")`}/>

<Quiz chapterSlug="019-ufunc-rounding" questions={[
  {id:"q1",type:"code-output",prompt:"np.around(2.5) returns:",code:"import numpy as np\nprint(np.around(2.5))",options:["2.0","3.0","2","Error"],correctIndex:0,explanation:"Banker's rounding: 2.5→nearest even=2.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"np.floor(-2.3) returns:",options:["-3.0","-2.0","-2","-3"],correctIndex:0,explanation:"Floor rounds toward -inf: -2.3→-3.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"Which rounds toward zero?",options:["trunc","floor","ceil","around"],correctIndex:0,explanation:"trunc truncates toward zero: -2.7→-2.",randomize:true}
]}/>
""")

w("020-ufunc-logs", """---
title: "Logarithmic ufuncs"
slug: "020-ufunc-logs"
description: "log, log2, log10, log1p, exp, expm1 — essential for softmax and numerical stability."
track: "ufuncs"
order: 5
read_time: 10
code_time: 8
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Logarithmic ufuncs

## Basic Logs

<PyRunner cellId="020-cell-1" defaultCode={`import numpy as np
a = np.array([1, 2, 4, 8, 16, 100], dtype=float)
print(f"ln:    {np.log(a).round(4)}")
print(f"log2:  {np.log2(a).round(4)}")
print(f"log10: {np.log10(a).round(4)}")
print(f"exp(ln(a))==a: {np.allclose(np.exp(np.log(a)), a)}")`}/>

## Numerical Stability

<PyRunner cellId="020-cell-2" defaultCode={`import numpy as np
tiny = 1e-15
print(f"log(1+tiny): {np.log(1+tiny)}")   # loses precision
print(f"log1p(tiny): {np.log1p(tiny)}")     # accurate
print(f"exp(tiny)-1: {np.exp(tiny)-1}")     # loses precision
print(f"expm1(tiny): {np.expm1(tiny)}")     # accurate`}/>

## Stable Softmax

<PyRunner cellId="020-cell-3" defaultCode={`import numpy as np
logits = np.array([1000, 1001, 1002])
print(f"Naive:    {np.exp(logits)/np.exp(logits).sum()}")  # nan!
shifted = logits - logits.max()
e = np.exp(shifted)
print(f"Stable:   {e/e.sum()}")`}/>

<Callout type="danger" title="Cross-Entropy Safety">
log(0) = -inf crashes training. Always clip probabilities: np.clip(probs, 1e-12, 1.0).
</Callout>

<Quiz chapterSlug="020-ufunc-logs" questions={[
  {id:"q1",type:"code-output",prompt:"np.log(0) returns:",code:"import numpy as np\nprint(np.log(0))",options:["-inf","0","nan","Error"],correctIndex:0,explanation:"log(0)=-infinity.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"Why use log1p(x) instead of log(1+x)?",options:["Maintains precision for tiny x","Faster","Identical","Handles negatives"],correctIndex:0,explanation:"When x~1e-15, 1+x rounds to 1.0. log1p computes accurately.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"Why subtract max in stable softmax?",options:["Prevents exp overflow, same ratios","Faster","Changes distribution","Optional"],correctIndex:0,explanation:"exp(x-max)/sum = exp(x)/sum. Same result, no overflow.",randomize:true}
]}/>
""")

w("021-ufunc-summations", """---
title: "Summation ufuncs"
slug: "021-ufunc-summations"
description: "sum, nansum, cumsum — axis-based aggregation for loss computation and statistics."
track: "ufuncs"
order: 6
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Summation ufuncs

## sum Along Axes

<PyRunner cellId="021-cell-1" defaultCode={`import numpy as np
m = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(f"Total:   {m.sum()}")
print(f"axis=0:  {m.sum(axis=0)}")
print(f"axis=1:  {m.sum(axis=1)}")
print(f"keepdims: {m.sum(axis=1, keepdims=True).shape}")`}/>

## nansum

<PyRunner cellId="021-cell-2" defaultCode={`import numpy as np
data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
print(f"sum:    {np.sum(data)}")
print(f"nansum: {np.nansum(data)}")`}/>

## cumsum

<PyRunner cellId="021-cell-3" defaultCode={`import numpy as np
print(f"cumsum: {np.cumsum([1,2,3,4,5])}")
m = np.array([[1,2],[3,4],[5,6]])
print(f"cumsum axis=0:\n{np.cumsum(m, axis=0)}")`}/>

## Batch MSE Loss

<PyRunner cellId="021-cell-4" defaultCode={`import numpy as np
pred = np.random.randn(32, 10)
target = np.random.randn(32, 10)
per_sample = np.sum((pred-target)**2, axis=1)
print(f"Mean batch loss: {per_sample.mean():.4f}")`}/>

<Quiz chapterSlug="021-ufunc-summations" questions={[
  {id:"q1",type:"code-output",prompt:"np.sum([[1,2],[3,4]], axis=0) returns:",code:"import numpy as np\nprint(np.sum([[1,2],[3,4]],axis=0))",options:["[4 6]","[3 7]","10","[[1 2]]"],correctIndex:0,explanation:"axis=0 sums columns: [1+3, 2+4]=[4,6].",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"np.sum([1, np.nan, 3]) returns:",options:["nan","4","0","Error"],correctIndex:0,explanation:"NaN propagates. Use nansum to ignore.",randomize:true},
  {id:"q3",type:"code-output",prompt:"np.cumsum([1,2,3,4]) returns:",code:"import numpy as np\nprint(np.cumsum([1,2,3,4]))",options:["[ 1  3  6 10]","[1 2 3 4]","10","[1 3 6]"],correctIndex:0,explanation:"Running sum: [1, 1+2, 1+2+3, 1+2+3+4].",randomize:true}
]}/>
""")

w("022-ufunc-products", """---
title: "Product ufuncs"
slug: "022-ufunc-products"
description: "prod, nanprod, cumprod — multiplication aggregation for probabilities and factorials."
track: "ufuncs"
order: 7
read_time: 6
code_time: 5
execution_timeout: 5
prerequisites: ["021-ufunc-summations"]
---

# Product ufuncs

## prod and nanprod

<PyRunner cellId="022-cell-1" defaultCode={`import numpy as np
print(f"prod: {np.prod([1,2,3,4,5])}")  # 120
data = np.array([2.0, np.nan, 4.0])
print(f"prod:    {np.prod(data)}")
print(f"nanprod: {np.nanprod(data)}")`}/>

## cumprod: Factorials

<PyRunner cellId="022-cell-2" defaultCode={`import numpy as np
print(f"Factorials: {np.cumprod(np.arange(1,11))}")
probs = np.array([0.9, 0.8, 0.7, 0.6])
print(f"Chain probs: {np.cumprod(probs).round(4)}")`}/>

<Callout type="warning" title="Overflow">
Products grow fast! Use log space: np.exp(np.sum(np.log(a))) for large products.
</Callout>

<Quiz chapterSlug="022-ufunc-products" questions={[
  {id:"q1",type:"code-output",prompt:"np.prod([2,3,4]) returns:",code:"import numpy as np\nprint(np.prod([2,3,4]))",options:["24","9","12","Error"],correctIndex:0,explanation:"2×3×4=24.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"How to compute large products without overflow?",options:["exp(sum(log(a)))","float128","Split chunks","Impossible"],correctIndex:0,explanation:"Log space: product→sum of logs, exp at end.",randomize:true},
  {id:"q3",type:"code-output",prompt:"np.cumprod([2,3,4]) returns:",code:"import numpy as np\nprint(np.cumprod([2,3,4]))",options:["[ 2  6 24]","[2 3 4]","24","[2 6 12]"],correctIndex:0,explanation:"Running product: [2, 2×3, 2×3×4].",randomize:true}
]}/>
""")

w("023-ufunc-differences", """---
title: "Difference ufuncs"
slug: "023-ufunc-differences"
description: "diff, gradient — discrete derivatives for numerical analysis and gradient checking."
track: "ufuncs"
order: 8
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["021-ufunc-summations"]
---

# Difference ufuncs

## np.diff

<PyRunner cellId="023-cell-1" defaultCode={`import numpy as np
a = np.array([1, 3, 6, 10, 15])
print(f"diff:     {np.diff(a)}")
print(f"diff n=2: {np.diff(a, n=2)}")`}/>

## np.gradient: Central Differences

<PyRunner cellId="023-cell-2" defaultCode={`import numpy as np
x = np.linspace(0, 2*np.pi, 100)
dy_num = np.gradient(np.sin(x), x)
dy_exact = np.cos(x)
print(f"Max error vs cos(x): {np.abs(dy_num-dy_exact).max():.6f}")`}/>

## Numerical Gradient Check

<PyRunner cellId="023-cell-3" defaultCode={`import numpy as np
def f(x): return x**2
def num_grad(f, x, eps=1e-7):
    return (f(x+eps) - f(x-eps)) / (2*eps)
x = 3.0
print(f"Numerical: {num_grad(f,x):.8f}")
print(f"Analytic:  {2*x:.8f}")
print(f"Match: {np.isclose(num_grad(f,x), 2*x)}")`}/>

<Quiz chapterSlug="023-ufunc-differences" questions={[
  {id:"q1",type:"code-output",prompt:"np.diff([1,4,9,16]) returns:",code:"import numpy as np\nprint(np.diff([1,4,9,16]))",options:["[3 5 7]","[1 4 9 16]","[4 9 16]","[1 2 3 4]"],correctIndex:0,explanation:"Consecutive diffs: 4-1=3, 9-4=5, 16-9=7.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"np.gradient uses:",options:["Central differences","Forward only","Backward only","Symbolic"],correctIndex:0,explanation:"Central differences for interior, forward/backward at edges.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"Output length of np.diff on array of length N?",options:["N-1","N","N+1","Depends"],correctIndex:0,explanation:"N-1 consecutive differences from N elements.",randomize:true}
]}/>
""")

w("024-ufunc-lcm-gcd", """---
title: "LCM & GCD ufuncs"
slug: "024-ufunc-lcm-gcd"
description: "Least common multiple and greatest common divisor — element-wise with reduce."
track: "ufuncs"
order: 9
read_time: 6
code_time: 5
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# LCM & GCD ufuncs

<PyRunner cellId="024-cell-1" defaultCode={`import numpy as np
a = np.array([12, 18, 24])
b = np.array([8, 12, 36])
print(f"GCD: {np.gcd(a, b)}")
print(f"LCM: {np.lcm(a, b)}")
print(f"gcd(48,18) = {np.gcd(48, 18)}")
print(f"lcm(4,6)   = {np.lcm(4, 6)}")`}/>

## Reduce Over Array

<PyRunner cellId="024-cell-2" defaultCode={`import numpy as np
nums = np.array([12, 18, 24, 36])
print(f"GCD of all: {np.gcd.reduce(nums)}")
print(f"LCM of all: {np.lcm.reduce(nums)}")`}/>

<Quiz chapterSlug="024-ufunc-lcm-gcd" questions={[
  {id:"q1",type:"code-output",prompt:"np.gcd(48, 18) returns:",code:"import numpy as np\nprint(np.gcd(48,18))",options:["6","12","3","144"],correctIndex:0,explanation:"GCD(48,18)=6.",randomize:true},
  {id:"q2",type:"code-output",prompt:"np.lcm(4, 6) returns:",code:"import numpy as np\nprint(np.lcm(4,6))",options:["12","2","24","6"],correctIndex:0,explanation:"LCM(4,6)=12.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"np.gcd.reduce([12,18,24]) computes:",options:["GCD of all three","Pairwise GCDs","First two only","Error"],correctIndex:0,explanation:"reduce: gcd(gcd(12,18),24)=gcd(6,24)=6.",randomize:true}
]}/>
""")

w("025-ufunc-trigonometric", """---
title: "Trigonometric ufuncs"
slug: "025-ufunc-trigonometric"
description: "sin, cos, tan, arcsin, arccos, arctan — all in radians with degree conversion."
track: "ufuncs"
order: 10
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["016-ufunc-intro"]
---

# Trigonometric ufuncs

All trig functions expect **radians**.

## Basic Trig

<PyRunner cellId="025-cell-1" defaultCode={`import numpy as np
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi])
print(f"Degrees: {np.rad2deg(angles).round(1)}")
print(f"sin:     {np.sin(angles).round(4)}")
print(f"cos:     {np.cos(angles).round(4)}")`}/>

## Inverse Trig

<PyRunner cellId="025-cell-2" defaultCode={`import numpy as np
vals = np.array([0, 0.5, 0.7071, 1.0])
print(f"arcsin: {np.rad2deg(np.arcsin(vals)).round(1)}°")
print(f"arccos: {np.rad2deg(np.arccos(vals)).round(1)}°")
x = np.array([1, 0, -1, 0])
y = np.array([0, 1, 0, -1])
print(f"arctan2: {np.rad2deg(np.arctan2(y, x))}°")`}/>

## Degree Conversion

<PyRunner cellId="025-cell-3" defaultCode={`import numpy as np
deg = np.array([0, 30, 45, 60, 90, 180, 360])
rad = np.deg2rad(deg)
print(f"deg2rad: {rad.round(4)}")
print(f"Round-trip OK: {np.allclose(deg, np.rad2deg(rad))}")`}/>

<Quiz chapterSlug="025-ufunc-trigonometric" questions={[
  {id:"q1",type:"code-output",prompt:"np.sin(np.pi) returns:",code:"import numpy as np\nprint(np.sin(np.pi))",options:["~1.2e-16","0.0 exactly","1.0","Error"],correctIndex:0,explanation:"Float precision: sin(pi)≈1.22e-16, not exactly 0.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"NumPy trig functions expect:",options:["Radians","Degrees","Gradians","Auto"],correctIndex:0,explanation:"All use radians. Use deg2rad() to convert.",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"np.arctan2(1, 0) returns:",options:["pi/2","0","pi","Undefined"],correctIndex:0,explanation:"arctan2(y=1,x=0)=pi/2 (90 degrees).",randomize:true}
]}/>
""")

w("026-ufunc-hyperbolic", """---
title: "Hyperbolic ufuncs"
slug: "026-ufunc-hyperbolic"
description: "sinh, cosh, tanh and inverses. Why tanh is a classic activation function."
track: "ufuncs"
order: 11
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["025-ufunc-trigonometric"]
---

# Hyperbolic ufuncs

## sinh, cosh, tanh

<PyRunner cellId="026-cell-1" defaultCode={`import numpy as np
x = np.linspace(-3, 3, 7)
print(f"x:    {x.round(2)}")
print(f"sinh: {np.sinh(x).round(4)}")
print(f"cosh: {np.cosh(x).round(4)}")
print(f"tanh: {np.tanh(x).round(4)}")
print(f"cosh²-sinh²=1: {np.allclose(np.cosh(x)**2 - np.sinh(x)**2, 1)}")`}/>

## tanh as Activation

<PyRunner cellId="026-cell-2" defaultCode={`import numpy as np
x = np.linspace(-5, 5, 11)
for xi in x:
    bar = "█" * int((np.tanh(xi)+1)*15)
    print(f"  {xi:+5.1f} → {np.tanh(xi):+.4f} {bar}")
print(f"\ntanh(0)={np.tanh(0)} (zero-centered!)")`}/>

## Inverse Hyperbolic

<PyRunner cellId="026-cell-3" defaultCode={`import numpy as np
v = np.array([0, 0.5, 0.9, -0.5])
print(f"arcsinh: {np.arcsinh(v).round(4)}")
print(f"arctanh: {np.arctanh(v).round(4)}")`}/>

<Quiz chapterSlug="026-ufunc-hyperbolic" questions={[
  {id:"q1",type:"code-output",prompt:"np.tanh(0) returns:",code:"import numpy as np\nprint(np.tanh(0))",options:["0.0","1.0","-1.0","0.5"],correctIndex:0,explanation:"tanh(0)=0. Zero-centered output.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"Range of np.tanh(x)?",options:["(-1, 1)","(0, 1)","(-inf, inf)","[0, 1]"],correctIndex:0,explanation:"tanh maps reals to open interval (-1,1).",randomize:true},
  {id:"q3",type:"multiple-choice",prompt:"Identity relating sinh and cosh?",options:["cosh²-sinh²=1","sinh²+cosh²=1","sinh×cosh=1","sinh+cosh=e^x"],correctIndex:0,explanation:"Fundamental identity: cosh²(x)-sinh²(x)=1.",randomize:true}
]}/>
""")

w("027-ufunc-set-operations", """---
title: "Set Operation ufuncs"
slug: "027-ufunc-set-operations"
description: "unique, intersect1d, union1d, setdiff1d, isin — set theory on arrays."
track: "ufuncs"
order: 12
read_time: 8
code_time: 6
execution_timeout: 5
prerequisites: ["014-array-sort"]
---

# Set Operation ufuncs

## unique

<PyRunner cellId="027-cell-1" defaultCode={`import numpy as np
a = np.array([3, 1, 2, 3, 1, 4, 2, 5, 3])
print(f"unique:       {np.unique(a)}")
print(f"with counts:  {np.unique(a, return_counts=True)}")
vals, inv = np.unique(a, return_inverse=True)
print(f"reconstruct:  {vals[inv]}")
print(f"match: {np.array_equal(vals[inv], a)}")`}/>

## Set Operations

<PyRunner cellId="027-cell-2" defaultCode={`import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([3, 4, 5, 6, 7])
print(f"intersect: {np.intersect1d(a, b)}")
print(f"union:     {np.union1d(a, b)}")
print(f"setdiff:   {np.setdiff1d(a, b)}")
print(f"setxor:    {np.setxor1d(a, b)}")`}/>

## Membership Testing

<PyRunner cellId="027-cell-3" defaultCode={`import numpy as np
a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
test = np.array([2, 4, 9, 1])
mask = np.isin(a, test)
print(f"isin mask: {mask}")
print(f"matching:  {a[mask]}")
print(f"not in:    {a[~mask]}")`}/>

<Quiz chapterSlug="027-ufunc-set-operations" questions={[
  {id:"q1",type:"code-output",prompt:"np.intersect1d([1,2,3],[2,3,4]) returns:",code:"import numpy as np\nprint(np.intersect1d([1,2,3],[2,3,4]))",options:["[2 3]","[1 2 3 4]","[1 4]","[2]"],correctIndex:0,explanation:"Intersection: elements in both.",randomize:true},
  {id:"q2",type:"multiple-choice",prompt:"np.unique(a, return_inverse=True) gives:",options:["Indices to reconstruct original","Counts","First occurrence","Nothing"],correctIndex:0,explanation:"inverse indices: unique[inverse]==original.",randomize:true},
  {id:"q3",type:"code-output",prompt:"np.setdiff1d([1,2,3,4],[3,4,5]) returns:",code:"import numpy as np\nprint(np.setdiff1d([1,2,3,4],[3,4,5]))",options:["[1 2]","[3 4]","[5]","[1 2 5]"],correctIndex:0,explanation:"In first but not second: [1,2].",randomize:true}
]}/>
""")

print("\n✅ Track 3 (ufuncs) complete")
