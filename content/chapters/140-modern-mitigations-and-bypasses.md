---
title: "Modern Mitigations & Bypasses"
description: "ASLR, DEP/NX, stack canaries, CFI — how they work and how attackers bypass them."
read_time: 13
code_time: 7
---

## The Arms Race

Every exploitation technique spawned a mitigation. Every mitigation spawned a bypass. Understanding both sides is essential.

## Mitigations

### ASLR (Address Space Layout Randomization)
Randomizes base addresses of stack, heap, libraries, and executable. Makes hardcoded addresses useless.

### DEP/NX (Data Execution Prevention)
Marks data pages (stack, heap) as non-executable. Shellcode on the stack cannot run.

### Stack Canaries
Places a random value between buffers and the return address. Overflow corrupts the canary before the return address, and the function aborts before returning.

### CFI (Control Flow Integrity)
Ensures indirect calls/jumps go to valid targets. Prevents ROP/JOP chains.

## Bypass Techniques

| Mitigation | Bypass |
|---|---|
| ASLR | Information leak (format string, partial overwrite), brute force (32-bit) |
| DEP/NX | Return-Oriented Programming (ROP), ret2libc |
| Stack Canary | Leak canary value, overwrite without touching canary |
| CFI | Shadow stack bypass, type confusion |

<Callout type="info" title="ROP is the universal DEP bypass">Instead of injecting shellcode, ROP chains together existing code snippets (gadgets) ending in ret. Each gadget performs a small operation, and together they achieve arbitrary computation without executing injected code.</Callout>

## Interactive: Mitigation Status Checker

<PyRunner
  cellId="140-modern-mitigations-bypasses-cell-1"
  defaultCode={`def check_mitigations(binary_info):
    results = []
    
    mitigations = {
        "NX": {"enabled": binary_info.get("nx", False), 
               "bypass": "ROP / ret2libc",
               "impact": "Blocks shellcode execution on stack/heap"},
        "ASLR": {"enabled": binary_info.get("aslr", False),
                 "bypass": "Info leak, partial overwrite, brute force (32-bit)",
                 "impact": "Randomizes memory layout"},
        "Canary": {"enabled": binary_info.get("canary", False),
                   "bypass": "Leak canary, overwrite without touching it",
                   "impact": "Detects stack buffer overflows"},
        "PIE": {"enabled": binary_info.get("pie", False),
                "bypass": "Info leak of base address",
                "impact": "Randomizes executable base address"},
        "RELRO": {"level": binary_info.get("relro", "none"),
                  "bypass": "Partial RELRO: GOT overwrite still possible",
                  "impact": "Protects GOT entries from overwrite"},
    }
    
    return mitigations

# Simulate checksec output for different binaries
binaries = [
    {"name": "legacy_app", "nx": False, "aslr": False, "canary": False, "pie": False, "relro": "none"},
    {"name": "modern_app", "nx": True, "aslr": True, "canary": True, "pie": True, "relro": "full"},
    {"name": "partial_hardened", "nx": True, "aslr": True, "canary": False, "pie": False, "relro": "partial"},
]

print("Binary Security Mitigation Assessment")
print("=" * 60)

for binary in binaries:
    mits = check_mitigations(binary)
    enabled = sum(1 for k,v in mits.items() if k != "RELRO" and v["enabled"])
    total = len(mits) - 1  # RELRO handled separately
    
    print(f"\n  {binary[name]}:")
    for name, info in mits.items():
        if name == "RELRO":
            status = info["level"].upper()
            color = "OK" if info["level"] == "full" else "PARTIAL" if info["level"] == "partial" else "NONE"
        else:
            status = "ENABLED" if info["enabled"] else "DISABLED"
            color = "OK" if info["enabled"] else "MISSING"
        
        print(f"    [{color:7s}] {name:8s}: {status}")
        if not info.get("enabled", True) or (name == "RELRO" and info["level"] != "full"):
            print(f"             Bypass: {info[bypass]}")

print("\n--- Exploit Difficulty Rating ---")
for binary in binaries:
    score = sum([
        binary.get("nx", False),
        binary.get("aslr", False),
        binary.get("canary", False),
        binary.get("pie", False),
        binary.get("relro") == "full",
    ])
    difficulty = "TRIVIAL" if score <= 1 else "MODERATE" if score <= 3 else "HARD" if score <= 4 else "VERY HARD"
    print(f"  {binary[name]:20s}: {difficulty} ({score}/5 mitigations)")`}
  timeout={8}
  title="Mitigation Assessment Tool"
/>

## Summary

Modern mitigations make exploitation significantly harder but not impossible. The attacker needs information leaks to defeat ASLR, ROP to defeat NX, and canary leaks to defeat stack protection. Defense in depth means enabling all mitigations — each one raises the bar.

<Quiz
  chapterSlug="140-modern-mitigations-bypasses"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does ASLR prevent?",
      options: ["Buffer overflows", "Using hardcoded memory addresses by randomizing layout at runtime", "Format string attacks", "SQL injection"],
      correctIndex: 1,
      explanation: "ASLR randomizes the base addresses of the stack, heap, libraries, and executable. Hardcoded addresses from exploit development become invalid across runs.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How does ROP bypass DEP/NX?",
      options: ["It disables DEP at runtime", "It chains existing code gadgets ending in ret to perform computation without injecting new executable code", "It encrypts the shellcode", "It moves shellcode to an executable page"],
      correctIndex: 1,
      explanation: "ROP uses short sequences of existing instructions (gadgets) that end with ret. By chaining return addresses on the stack, it achieves arbitrary computation using only existing executable code, never injecting new code.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the most common way to bypass ASLR?",
      options: ["Brute force on 64-bit systems", "Information leak that reveals a runtime address", "Disabling it via registry", "Using larger buffers"],
      correctIndex: 1,
      explanation: "An information leak (via format string, partial overwrite, or other vulnerability) reveals a runtime address. From one known address, the attacker calculates all others since relative offsets remain constant.",
      randomize: true,
    }
  ]}
/>
