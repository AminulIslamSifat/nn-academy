---
title: "Assembly & Debugging Basics"
description: "Reading x86 assembly, using GDB/x64dbg, and understanding how compiled code actually runs."
read_time: 14
code_time: 8
---

## Why Learn Assembly?

High-level languages abstract away what the CPU actually does. Reverse engineering and exploitation require understanding the machine level: registers, stack frames, memory layout, and instruction flow.

## x86 Registers (64-bit)

| Register | Purpose |
|---|---|
| RAX | Return value, accumulator |
| RBX | Base register (callee-saved) |
| RCX | First argument (Windows) / fourth arg (Linux) |
| RDX | Second argument (Windows) / third arg (Linux) |
| RSI | Source index / second arg (Linux) |
| RDI | Destination index / first arg (Linux) |
| RBP | Base pointer (frame pointer) |
| RSP | Stack pointer |
| RIP | Instruction pointer |

### Linux x64 Calling Convention
Arguments passed in: RDI, RSI, RDX, RCX, R8, R9. Return value in RAX.

## The Stack

The stack grows **downward** (toward lower addresses). Each function call creates a **stack frame**:

```
High addresses
  [caller frame]
  [return address]     <- pushed by CALL
  [saved RBP]          <- old base pointer
  [local variables]    <- current frame
  [function args]      <- for next call
Low addresses           <- RSP points here
```

## Basic Instructions

```nasm
mov rax, 42       ; rax = 42
push rbp          ; save base pointer
sub rsp, 32       ; allocate 32 bytes for locals
call function     ; push RIP, jump to function
add rsp, 32       ; deallocate locals
pop rbp           ; restore base pointer
ret               ; pop RIP, return to caller
```

## Using GDB

```bash
# Start debugging
gdb ./program

# Key commands
break main        # breakpoint at main
run               # start execution
next              # step over (source level)
step              # step into
info registers    # show all registers
x/10x $rsp       # examine 10 words at stack pointer
disassemble       # show assembly for current function
continue          # resume execution
```

<Callout type="info" title="GDB with source">Compile with gcc -g -O0 for debug symbols and no optimization. This maps assembly back to source lines and preserves variable names.</Callout>

## Interactive: Stack Frame Simulator

<PyRunner
  cellId="136-assembly-debugging-basics-cell-1"
  defaultCode={`class StackSimulator:
    def __init__(self):
        self.stack = []
        self.rsp = 0xFFFF
        self.operations = []
    
    def push(self, value, label=""):
        self.rsp -= 8
        self.stack.append((self.rsp, value, label))
        self.operations.append(f"PUSH {label or value} -> RSP=0x{self.rsp:04X}")
    
    def pop(self):
        if self.stack:
            addr, val, label = self.stack.pop()
            self.rsp += 8
            self.operations.append(f"POP  {label or val} <- RSP=0x{self.rsp:04X}")
            return val
        return None
    
    def display(self):
        print("Stack State (top = RSP):")
        print("-" * 45)
        for addr, val, label in reversed(self.stack[-8:]):
            marker = " <- RSP" if addr == self.rsp else ""
            print(f"  0x{addr:04X}: 0x{val:016X}  {label}{marker}")
        print("-" * 45)

sim = StackSimulator()

print("x86-64 Stack Frame Simulation")
print("=" * 50)
print("\nSimulating: main() calls vulnerable_func(buf)")
print()

# Simulate call sequence
sim.push(0xDEADBEEF, "main return addr")
sim.push(0x7FFFFFFFE0, "saved RBP (main)")
sim.push(0x4141414141414141, "buf[0-7] AAAA...")
sim.push(0x4141414141414141, "buf[8-15] AAAA...")
sim.push(0x4141414141414141, "buf[16-23] AAAA...")
sim.push(0x4141414141414141, "buf[24-31] AAAA...")

print("After entering vulnerable_func with 32-byte buffer:")
sim.display()

print("\nOverflow: writing 48 bytes into 32-byte buffer...")
# Overwrite saved RBP and return address
sim.stack[-6] = (sim.stack[-6][0], 0x4242424242424242, "OVERFLOW -> saved RBP")
sim.stack[-7] = (sim.stack[-7][0], 0x00007FFFF7E12345, "OVERFLOW -> return addr!")

print("\nAfter overflow:")
sim.display()
print("\n  [!] Return address overwritten!")
print("  [!] When ret executes, control jumps to attacker address")
print("  [!] This is the basis of stack buffer overflow exploits")`}
  timeout={8}
  title="Stack Frame Simulator"
/>

## Summary

Assembly is the language of exploitation and reverse engineering. Understand registers, the stack, calling conventions, and basic GDB usage. These foundations unlock everything in the rest of this track.

<Quiz
  chapterSlug="136-assembly-debugging-basics"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "In x86-64 Linux, which register holds the first function argument?",
      options: ["RAX", "RCX", "RDI", "RSI"],
      correctIndex: 2,
      explanation: "The System V AMD64 ABI (Linux) passes the first six integer arguments in RDI, RSI, RDX, RCX, R8, R9. Windows uses a different convention (RCX, RDX, R8, R9).",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Which direction does the x86 stack grow?",
      options: ["Upward (toward higher addresses)", "Downward (toward lower addresses)", "It depends on the OS", "Sideways"],
      correctIndex: 1,
      explanation: "The x86 stack grows downward toward lower memory addresses. PUSH decrements RSP, POP increments it. This is fundamental to understanding buffer overflows.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does the CALL instruction do?",
      options: ["Jumps to a label without saving state", "Pushes the return address onto the stack and jumps to the target", "Allocates stack space", "Returns from a function"],
      correctIndex: 1,
      explanation: "CALL pushes the address of the next instruction (return address) onto the stack, then jumps to the target function. RET pops this address back into RIP to return.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why compile with -g -O0 for debugging?",
      options: ["It makes the program faster", "It includes debug symbols and disables optimizations so assembly maps cleanly to source", "It enables network debugging", "It encrypts the binary"],
      correctIndex: 1,
      explanation: "-g adds debug symbols (variable names, source line mappings). -O0 disables optimizations that reorder or eliminate code. Together they make debugging tractable.",
      randomize: true,
    }
  ]}
/>
