---
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

<Quiz
  chapterSlug="025-ufunc-trigonometric"
  questions={[
    {
      id: "q1",
      type: "code-output",
      prompt: "np.sin(np.pi) returns:",
      code: "import numpy as np\nprint(np.sin(np.pi))",
      options: ["~1.2e-16","0.0 exactly","1.0","Error"],
      correctIndex: 0,
      explanation: "Float precision: sin(pi)≈1.22e-16, not exactly 0.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "NumPy trig functions expect:",
      options: ["Radians","Degrees","Gradians","Auto"],
      correctIndex: 0,
      explanation: "All use radians. Use deg2rad() to convert.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "np.arctan2(1, 0) returns:",
      options: ["pi/2","0","pi","Undefined"],
      correctIndex: 0,
      explanation: "arctan2(y=1,x=0)=pi/2 (90 degrees).",
      randomize: true,
    }
  ]}
/>
