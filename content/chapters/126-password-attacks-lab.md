---
title: "Password Attacks Lab"
description: "Understand how password hashes are cracked and why hashing parameters matter, with hands-on simulations."
read_time: 13
code_time: 8
---

## Why Passwords Still Matter

Despite MFA and passkeys, passwords remain the most common authentication factor. When a database leaks, the attacker gets hashes — and the race to crack them begins.

This chapter is about understanding that race from both sides: how attackers crack, and how defenders make cracking impractical.

## How Password Hashing Works

A proper password hash is:

1. **One-way** — you cannot reverse the hash to get the password.
2. **Salted** — a unique random value per user prevents rainbow tables.
3. **Slow** — computationally expensive to resist brute force.

```
hash = PBKDF2(password + salt, iterations=600000)
```

The **iterations** parameter is the defender's main lever. More iterations = more time per guess = fewer guesses per second.

<Callout type="danger" title="MD5 and SHA-256 are not password hashes">They are fast by design. A GPU can compute billions of SHA-256 hashes per second. Password hashing requires intentionally slow functions: bcrypt, scrypt, Argon2, or PBKDF2 with high iterations.</Callout>

## Attack Types

### Brute Force

Try every possible combination. Feasible for short passwords, impractical for long ones.

| Password Length | Possible Combinations (lowercase + digits) |
|---|---|
| 6 chars | ~2.2 billion |
| 8 chars | ~2.8 trillion |
| 12 chars | ~3.2 x 10^18 |

### Dictionary Attack

Try words from a list (rockyou.txt has 14 million passwords). Most people use words, names, and simple patterns, making this devastatingly effective.

### Rule-Based Attack

Apply mutations to dictionary words: capitalize, append numbers, leet-speak substitutions. `password` becomes `Password1`, `P@ssw0rd`, `password2024!`.

### Rainbow Tables (Defeated by Salting)

Precomputed hash-to-password mappings. A unique salt per user makes precomputed tables useless because the attacker must recompute for each salt.

## Interactive: Password Entropy Calculator

<PyRunner
  cellId="126-password-attacks-lab-cell-1"
  defaultCode={`import math
import string

def password_entropy(password):
    charset = 0
    if any(c in string.ascii_lowercase for c in password):
        charset += 26
    if any(c in string.ascii_uppercase for c in password):
        charset += 26
    if any(c in string.digits for c in password):
        charset += 10
    if any(c in string.punctuation for c in password):
        charset += 32
    if charset == 0:
        return 0
    return len(password) * math.log2(charset)

def crack_time(entropy, guesses_per_sec):
    total = 2 ** entropy
    seconds = total / guesses_per_sec
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 31536000:
        return f"{seconds/86400:.1f} days"
    else:
        return f"{seconds/31536000:.1f} years"

passwords = [
    "password",
    "Password1",
    "Tr0ub4dor&3",
    "correct-horse-battery-staple",
    "aB3$xZ9!mK2@pL7#",
]

# Attack speeds
scenarios = [
    ("MD5 (GPU)", 10_000_000_000),     # 10 billion/sec
    ("SHA-256 (GPU)", 5_000_000_000),   # 5 billion/sec
    ("bcrypt (cost 12)", 10_000),       # 10 thousand/sec
    ("PBKDF2 (600k iter)", 5_000),      # 5 thousand/sec
]

print("Password Entropy & Crack Time Estimator")
print("=" * 65)
for pwd in passwords:
    ent = password_entropy(pwd)
    print(f"\n'{pwd}' -> {ent:.1f} bits of entropy")
    for name, speed in scenarios:
        time_str = crack_time(ent, speed)
        print(f"  vs {name:25s}: {time_str}")`}
  timeout={8}
  title="Password Entropy Calculator"
/>

## Interactive: PBKDF2 Hashing Demo

<PyRunner
  cellId="126-password-attacks-lab-cell-2"
  defaultCode={`import hashlib
import os
import time

password = "MyS3cur3Pass!"
salt = os.urandom(16)  # 128-bit random salt

# Demonstrate PBKDF2 with different iteration counts
iterations_list = [1_000, 10_000, 100_000, 600_000]

print("PBKDF2 Hashing with Increasing Iterations")
print("=" * 55)
print(f"Password: {password}")
print(f"Salt (hex): {salt.hex()}\n")

for iterations in iterations_list:
    start = time.time()
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, dklen=32)
    elapsed = time.time() - start
    print(f"  Iterations: {iterations:>8,} | Time: {elapsed:.4f}s | Hash: {dk.hex()[:32]}...")

# Show that different salts produce different hashes
salt2 = os.urandom(16)
dk1 = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100_000, dklen=32)
dk2 = hashlib.pbkdf2_hmac('sha256', password.encode(), salt2, 100_000, dklen=32)

print(f"\nSame password, different salts:")
print(f"  Salt 1 hash: {dk1.hex()[:32]}...")
print(f"  Salt 2 hash: {dk2.hex()[:32]}...")
print(f"  Hashes match: {dk1 == dk2} (salt makes each unique)")

# Estimate attack cost
print(f"\nAt 5,000 guesses/sec (PBKDF2 600k iter):")
print(f"  1 million passwords = {1_000_000 / 5_000:.0f} seconds = {1_000_000 / 5_000 / 3600:.1f} hours")`}
  timeout={8}
  title="PBKDF2 Hashing Lab"
/>

## Defensive Recommendations

1. **Use Argon2id** (preferred) or bcrypt with cost >= 12.
2. **Enforce length** — minimum 12 characters matters more than complexity rules.
3. **Check against breached lists** — use the HaveIBeenPwned k-anonymity API.
4. **Rate limit login attempts** — slow down online attacks.
5. **MFA** — even a cracked password should not grant access alone.

<Callout type="tip" title="Length beats complexity">A 16-character passphrase like 'correct-horse-battery-staple' has more entropy than 'P@ss1' and is easier to remember. Encourage passphrases over short complex passwords.</Callout>

## Summary

Password cracking is a numbers game: entropy vs. guesses per second. Defenders win by increasing the cost per guess (slow hashing), increasing password entropy (length), and adding layers (MFA). Understanding the attacker's math helps you configure the right defenses.

<Quiz
  chapterSlug="126-password-attacks-lab"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is a salt added to password hashes?",
      options: ["To make the hash shorter", "To prevent rainbow table attacks by making each hash unique", "To encrypt the password", "To speed up the hashing process"],
      correctIndex: 1,
      explanation: "A unique random salt per user means the same password produces different hashes for different users. This defeats precomputed rainbow tables because the attacker must recompute for each unique salt.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why are MD5 and SHA-256 unsuitable for password hashing?",
      options: ["They produce hashes that are too long", "They are designed to be fast, allowing billions of guesses per second on GPUs", "They cannot handle special characters", "They are not available in Python"],
      correctIndex: 1,
      explanation: "MD5 and SHA-256 are optimized for speed. Password hashing needs to be intentionally slow to make brute force impractical. Use bcrypt, scrypt, Argon2, or PBKDF2 with high iterations.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is a rule-based password attack?",
      options: ["Trying random characters", "Applying transformations like capitalization and number appending to dictionary words", "Using rainbow tables", "Intercepting passwords over the network"],
      correctIndex: 1,
      explanation: "Rule-based attacks take dictionary words and apply mutation rules (capitalize first letter, append digits, leet-speak substitutions) to generate likely password candidates.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which factor has the most impact on password security?",
      options: ["Requiring at least one special character", "Password length", "Requiring mixed case", "Changing passwords every 30 days"],
      correctIndex: 1,
      explanation: "Length increases entropy exponentially. A 16-character passphrase is vastly more secure than an 8-character complex password. NIST guidelines now emphasize length over complexity rules.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does increasing PBKDF2 iterations do?",
      options: ["Makes the hash shorter", "Increases the computational cost per hash, slowing brute force attacks", "Adds more salt automatically", "Encrypts the hash output"],
      correctIndex: 1,
      explanation: "Each iteration repeats the hash computation. More iterations mean each password guess takes longer, reducing the attacker's guesses per second and making brute force impractical.",
      randomize: true,
    }
  ]}
/>
