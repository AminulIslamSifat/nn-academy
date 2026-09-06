---
title: "Static Analysis with Ghidra"
description: "Decompiling binaries, navigating code, and identifying functions without executing anything."
read_time: 13
code_time: 7
---

## What Is Static Analysis?

Analyzing a binary without running it. You read the disassembly or decompiled code to understand logic, find vulnerabilities, and identify malware behavior.

## Ghidra Overview

NSA open-source reverse engineering suite. Key features:
- Disassembler and decompiler
- Cross-reference tracking
- Data type recovery
- Scripting (Java/Python)
- Collaborative projects

## The Analysis Workflow

1. Import the binary
2. Auto-analyze (default analyzers)
3. Navigate to entry point or main
4. Follow cross-references to interesting strings/functions
5. Rename variables and functions as you understand them
6. Identify vulnerability patterns

## Finding Interesting Code

Strings are your roadmap:
- Error messages reveal logic paths
- URLs and IPs indicate C2
- Crypto constants reveal algorithms
- Format strings may indicate format string bugs

```python
# Ghidra Python script: find all strings containing "http"
from ghidra.program.util import DefinedDataIterator
for data in DefinedDataIterator.definedStrings(currentProgram):
    if "http" in str(data.getValue()).lower():
        print(f"0x{data.getAddress()}: {data.getValue()}")
```

<Callout type="tip" title="Rename everything">As you understand a function, rename it. As you understand a variable, rename it. Future-you (and teammates) will thank present-you.</Callout>

## Identifying Vulnerability Patterns

| Pattern | What to Look For |
|---|---|
| Buffer overflow | Fixed-size stack buffer + unchecked copy (strcpy, memcpy) |
| Format string | User input passed directly to printf-family as format |
| Integer overflow | Arithmetic before allocation size calculation |
| Use-after-free | Pointer used after kfree/free |
| Command injection | User input concatenated into system/popen call |

## Interactive: Vulnerability Pattern Matcher

<PyRunner
  cellId="137-static-analysis-ghidra-cell-1"
  defaultCode={`code_snippets = [
    {"func": "handle_input", "code": "char buf[64]; strcpy(buf, user_input);", "risk": "Buffer overflow: fixed buffer + unchecked copy"},
    {"func": "log_message", "code": "printf(user_message);", "risk": "Format string: user input as format string"},
    {"func": "alloc_buffer", "code": "size_t total = width * height; char *buf = malloc(total);", "risk": "Integer overflow: multiplication may wrap before malloc"},
    {"func": "process_data", "code": "free(ptr); ... use(ptr->field);", "risk": "Use-after-free: pointer dereferenced after free"},
    {"func": "run_command", "code": "sprintf(cmd, \"ls %s\", user_path); system(cmd);", "risk": "Command injection: unsanitized input in system()"},
    {"func": "safe_copy", "code": "strncpy(buf, input, sizeof(buf)-1); buf[sizeof(buf)-1]=0;", "risk": None},
]

print("Static Analysis: Vulnerability Pattern Scanner")
print("=" * 60)

vuln_count = 0
for snippet in code_snippets:
    if snippet["risk"]:
        vuln_count += 1
        print(f"\n  [VULN] {snippet[func]}()")
        print(f"    Code: {snippet[code]}")
        print(f"    Issue: {snippet[risk]}")
    else:
        print(f"\n  [SAFE] {snippet[func]}()")
        print(f"    Code: {snippet[code]}")

print(f"\n--- Summary ---")
print(f"Functions analyzed: {len(code_snippets)}")
print(f"Vulnerabilities found: {vuln_count}")
print(f"Safe patterns: {len(code_snippets) - vuln_count}")`}
  timeout={8}
  title="Vulnerability Pattern Scanner"
/>

## Summary

Static analysis with Ghidra lets you understand binaries without execution. Follow strings, rename aggressively, and learn to recognize vulnerability patterns in disassembly. This skill is foundational for malware analysis and exploit development.

<Quiz
  chapterSlug="137-static-analysis-ghidra"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the primary advantage of static analysis over dynamic analysis?",
      options: ["It is faster", "It examines all code paths without needing to trigger them at runtime", "It requires less skill", "It works on encrypted binaries"],
      correctIndex: 1,
      explanation: "Static analysis examines the code directly, covering all paths including those that are difficult to trigger dynamically. Dynamic analysis only sees paths that are actually executed.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why are strings important in static analysis?",
      options: ["They are the only readable part of a binary", "They serve as landmarks to navigate code and reveal functionality like URLs, error messages, and crypto constants", "They are always encrypted", "They indicate the compiler version"],
      correctIndex: 1,
      explanation: "Strings are human-readable anchors in a binary. Following cross-references to strings reveals the code that uses them, making navigation and understanding much faster.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What vulnerability pattern does strcpy(buf, user_input) with a fixed-size buf indicate?",
      options: ["Format string vulnerability", "Buffer overflow", "SQL injection", "Race condition"],
      correctIndex: 1,
      explanation: "strcpy copies without bounds checking. If user_input exceeds the buffer size, it overwrites adjacent stack memory, potentially allowing code execution.",
      randomize: true,
    }
  ]}
/>
