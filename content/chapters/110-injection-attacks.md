---
title: "Injection Attacks"
description: "SQL injection, command injection, and XSS — why mixing code and data is the root cause, with hands-on demonstrations."
read_time: 13
code_time: 8
---

## The Root Cause

Injection happens when **untrusted data is interpreted as code**. The fix is always to keep data and instructions separate. This single principle eliminates SQL injection, command injection, XSS, and every other injection variant.

## SQL Injection

### The Classic Example

If you build a query by concatenating user input, an attacker changes its meaning:

```python
# Vulnerable: name is concatenated directly
query = "SELECT * FROM users WHERE name = '" + user_input + "'"
```

Feeding `' OR '1'='1` makes the condition always true:

```sql
SELECT * FROM users WHERE name = '' OR '1'='1'
```

This returns **every row** in the users table.

### The Fix: Parameterized Queries

```python
# Safe: parameter binding separates code from data
cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
```

The database driver sends the statement structure and the data separately. The input is never interpreted as SQL.

### Blind SQL Injection

When the application does not display query results directly, attackers infer data from behavior:

- **Boolean-based** — `AND 1=1` vs `AND 1=2` changes the response.
- **Time-based** — `AND SLEEP(5)` delays the response if the condition is true.

```sql
-- Time-based blind: does the admin user exist?
SELECT * FROM users WHERE id = 1 AND IF(SUBSTRING(password,1,1)='a', SLEEP(5), 0)
```

<Callout type="warning" title="ORMs are not immune">Object-Relational Mappers protect against injection when you use their query builder. But raw SQL strings, even inside an ORM, are still vulnerable. Never concatenate user input into raw SQL.</Callout>

## Command Injection

Same idea at the OS level. If you pass user input into a shell command, an attacker appends their own commands:

```python
# Vulnerable
import os
os.system("ping " + user_input)
# Input: "8.8.8.8; cat /etc/passwd"
# Executes: ping 8.8.8.8; cat /etc/passwd
```

The fix: never invoke a shell with interpolated input. Use argument lists:

```python
# Safe: argument list, no shell
import subprocess
subprocess.run(["ping", "-c", "1", target], shell=False)
```

## Cross-Site Scripting (XSS)

When a page reflects user input as HTML, an attacker injects script that runs in other users' browsers.

### Types

| Type | How It Works | Persistence |
|---|---|---|
| **Reflected** | Input echoed in the immediate response | None |
| **Stored** | Input saved and served to other users | Database |
| **DOM-based** | Client-side JS processes the input unsafely | None |

### Example

```html
<!-- Vulnerable: user input inserted as HTML -->
<div>Welcome, <span id="name"></span></div>
<script>
  document.getElementById("name").innerHTML = getParam("name");
</script>

<!-- Attacker sends: ?name=<img src=x onerror=alert(document.cookie)> -->
```

### Defenses

1. **Encode output** for its context (HTML, attribute, URL, JavaScript).
2. **Content-Security-Policy** header restricts what scripts can run.
3. **Use frameworks** that auto-escape (React, Vue, Angular).
4. **HttpOnly cookies** prevent JavaScript from reading session tokens.

## Interactive: SQL Injection Simulator

<PyRunner
  cellId="110-injection-attacks-cell-1"
  defaultCode={`# Simulate SQL injection vs parameterized queries

users_db = [
    {"id": 1, "name": "alice", "role": "user", "email": "alice@example.com"},
    {"id": 2, "name": "bob", "role": "admin", "email": "bob@example.com"},
    {"id": 3, "name": "charlie", "role": "user", "email": "charlie@example.com"},
]

def vulnerable_query(name_input):
    """Simulates string concatenation (vulnerable)"""
    # In real SQL: SELECT * FROM users WHERE name = '{input}'
    if name_input == "' OR '1'='1":
        return users_db  # Returns ALL users
    return [u for u in users_db if u["name"] == name_input]

def safe_query(name_input):
    """Simulates parameterized query (safe)"""
    # Input is treated as a literal value, never as SQL
    return [u for u in users_db if u["name"] == name_input]

print("SQL Injection Demonstration")
print("=" * 55)

test_inputs = ["alice", "' OR '1'='1", "bob"]

for inp in test_inputs:
    print(f"\nInput: '{inp}'")

    vuln_result = vulnerable_query(inp)
    safe_result = safe_query(inp)

    print(f"  Vulnerable query returns {len(vuln_result)} row(s):")
    for u in vuln_result:
        print(f"    {u['name']} ({u['role']})")

    print(f"  Safe query returns {len(safe_result)} row(s):")
    for u in safe_result:
        print(f"    {u['name']} ({u['role']})")

print("\n--- Key Takeaway ---")
print("Parameterized queries treat input as DATA, never as CODE.")
print("The vulnerable query returned ALL users with ' OR '1'='1.")
print("The safe query correctly found no match.")`}
  timeout={8}
  title="SQL Injection Simulator"
/>

## Interactive: XSS Payload Encoder

<PyRunner
  cellId="110-injection-attacks-cell-2"
  defaultCode={`import html

# Simulate XSS payloads and encoding defenses

payloads = [
    '<script>alert(document.cookie)</script>',
    '<img src=x onerror=alert(1)>',
    '"><svg onload=alert(1)>',
    "javascript:alert(1)",
]

print("XSS Encoding Defense Demonstration")
print("=" * 60)

for payload in payloads:
    encoded = html.escape(payload)
    print(f"\n  Raw payload: {payload}")
    print(f"  HTML-encoded: {encoded}")
    print(f"  Safe to render: {'Yes' if '<' not in encoded and '>' not in encoded else 'Needs review'}")

# Demonstrate context-aware encoding
print("\n--- Context Matters ---")
user_input = '" onmouseover="alert(1)'

contexts = {
    "HTML body": html.escape(user_input),
    "HTML attribute": user_input.replace('"', '&quot;'),
    "URL parameter": user_input.replace('"', '%22').replace(' ', '+'),
}

for ctx, encoded in contexts.items():
    print(f"  {ctx:18s}: {encoded}")

print("\n--- Content-Security-Policy ---")
csp = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
print(f"  CSP: {csp}")
print("  Effect: Blocks inline scripts, external scripts, and eval().")
print("  Even if XSS payload executes, it cannot exfiltrate data to external domains.")`}
  timeout={8}
  title="XSS Encoding Lab"
/>

## The Universal Rule

<Callout type="danger" title="Never trust input">Validate on the way in, and encode or parameterize on the way out. Every injection vulnerability exists because this rule was violated somewhere in the code path.</Callout>

## Summary

Injection is one bug class with many faces: SQL injection, command injection, XSS, SSTI, LDAP injection. The root cause is always the same — mixing untrusted data with code. The fix is always the same — keep them separate through parameterization, encoding, and safe APIs.

<Quiz
  chapterSlug="110-injection-attacks"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the root cause of all injection vulnerabilities?",
      options: ["Weak passwords", "Untrusted data being interpreted as code", "Missing SSL certificates", "Outdated operating systems"],
      correctIndex: 1,
      explanation: "Injection occurs when untrusted data is mixed with code or commands. The attacker's input changes the meaning of the query, command, or markup. Keeping data and instructions separate eliminates injection.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "How do parameterized queries prevent SQL injection?",
      options: ["They encrypt the query", "They send the SQL structure and the data separately so input is never interpreted as SQL", "They filter out special characters", "They use a different database engine"],
      correctIndex: 1,
      explanation: "Parameterized queries (prepared statements) send the query structure to the database first, then bind the data values separately. The database never interprets the data as part of the SQL command.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What type of XSS stores the malicious payload in a database?",
      options: ["Reflected XSS", "Stored XSS", "DOM-based XSS", "Blind XSS"],
      correctIndex: 1,
      explanation: "Stored XSS (persistent XSS) saves the malicious input in the database. Every user who views the affected page receives the payload, making it more dangerous than reflected XSS.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What does the Content-Security-Policy header help prevent?",
      options: ["SQL injection", "Cross-site scripting by restricting which scripts can execute", "Command injection", "Directory traversal"],
      correctIndex: 1,
      explanation: "CSP restricts which sources can execute scripts, load styles, and make connections. Even if an XSS payload is injected, CSP can prevent it from executing or exfiltrating data.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Why is subprocess.run with shell=False safer than os.system?",
      options: ["It runs faster", "It passes arguments as a list without invoking a shell, preventing command chaining", "It encrypts the command", "It requires root privileges"],
      correctIndex: 1,
      explanation: "os.system invokes a shell, allowing command chaining with ; && or |. subprocess.run with shell=False passes arguments directly to the program, so special characters are not interpreted as shell syntax.",
      randomize: true,
    }
  ]}
/>
