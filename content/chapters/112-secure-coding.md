---
title: "Secure Coding Practices"
description: "Building security into code review, dependency management, secrets handling, and the CI/CD pipeline."
read_time: 12
code_time: 7
---

## Shift Left

Fixing a vulnerability in design costs orders of magnitude less than fixing it after deployment. Security must be woven into the workflow, not audited at the end.

```
Design -> Code -> Review -> CI -> Deploy -> Runtime
  $1       $10     $50     $100   $1000    $10000+
```

## Input Validation & Output Encoding

Validate all input at the boundary. Encode all output for its context (HTML, URL, SQL, shell). These two habits remove the majority of injection bugs.

```python
# Input validation: whitelist approach
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError("Invalid email format")
    return email

# Output encoding: context-specific
import html
def render_user_input(user_input):
    return html.escape(user_input)  # HTML context
```

<Callout type="tip" title="Whitelist over blacklist">Blocking known-bad patterns (blacklisting) always misses something. Allowing only known-good patterns (whitelisting) is far more robust. Validate format, length, type, and range.</Callout>

## Dependency Hygiene

- Keep a **software bill of materials**.
- Pin versions and audit for known CVEs.
- Remove unused dependencies — each is added attack surface.
- Review changelogs before updating, especially for major version bumps.

```bash
# Python: check for known vulnerabilities
pip-audit

# Node.js: check the dependency tree
npm audit

# Both: automate in CI
pip-audit --fail-on-vuln  # fails CI if vulnerability found
```

## Secrets Management

Never commit secrets to version control. Use a secret manager or environment injection at deploy time, and rotate keys.

### Common Mistakes

```python
# NEVER do this
API_KEY = "sk-live-4eC39HqLyjWDarjtT1zdp7dc"
DB_PASSWORD = "hunter2"

# Do this instead
import os
API_KEY = os.environ["API_KEY"]  # Injected at deploy time
```

<Callout type="warning" title="The .env trap">A committed `.env` file lives forever in git history even after deletion. If you ever committed a secret, treat it as compromised and rotate it immediately. Use `git filter-branch` or BFG Repo-Cleaner to remove it from history, and rotate the key.</Callout>

### Secret Detection in CI

Add tools like **gitleaks** or **trufflehog** to your CI pipeline to catch secrets before they are merged:

```yaml
# GitHub Actions example
- name: Secret scan
  run: gitleaks detect --source . --verbose
```

## Automate the Checks

Add these to CI so regressions are caught before merge:

| Check | Tool Examples | Catches |
|---|---|---|
| SAST | Semgrep, Bandit, ESLint | Code-level vulnerabilities |
| Dependency scan | pip-audit, npm audit | Known CVEs in libraries |
| Secret detection | gitleaks, trufflehog | Committed credentials |
| Container scan | Trivy, Grype | Vulnerable base images |
| License check | license-checker | Legal compliance |

## Interactive: Dependency Vulnerability Scanner

<PyRunner
  cellId="112-secure-coding-cell-1"
  defaultCode={`# Simulate scanning dependencies against known-bad versions
dependencies = [
    {"name": "django", "version": "3.2.14", "ecosystem": "pypi"},
    {"name": "requests", "version": "2.28.0", "ecosystem": "pypi"},
    {"name": "express", "version": "4.17.1", "ecosystem": "npm"},
    {"name": "lodash", "version": "4.17.20", "ecosystem": "npm"},
    {"name": "log4j-core", "version": "2.14.1", "ecosystem": "maven"},
    {"name": "flask", "version": "2.2.2", "ecosystem": "pypi"},
]

# Simulated CVE database
known_vulns = {
    ("django", "3.2.14"): "CVE-2022-34265: SQL injection in Trunc/Extract",
    ("lodash", "4.17.20"): "CVE-2021-23337: Command injection via template",
    ("log4j-core", "2.14.1"): "CVE-2021-44228: Log4Shell RCE (CRITICAL)",
}

print("Dependency Vulnerability Scan")
print("=" * 60)

vulns_found = 0
for dep in dependencies:
    key = (dep["name"], dep["version"])
    if key in known_vulns:
        vulns_found += 1
        severity = "CRITICAL" if "CRITICAL" in known_vulns[key] else "HIGH"
        print(f"  [VULN] {dep['name']}=={dep['version']} ({dep['ecosystem']})")
        print(f"         {known_vulns[key]}")
        print(f"         Severity: {severity}")
    else:
        print(f"  [OK]   {dep['name']}=={dep['version']} ({dep['ecosystem']})")

print(f"\nScan complete: {vulns_found} vulnerable dependencies out of {len(dependencies)}")
if vulns_found > 0:
    print("\nRemediation:")
    print("  - Upgrade django to >= 3.2.15")
    print("  - Upgrade lodash to >= 4.17.21")
    print("  - UPGRADE log4j-core to >= 2.17.1 IMMEDIATELY (Log4Shell)")`}
  timeout={8}
  title="Dependency Vulnerability Scanner"
/>

## Interactive: Secret Detection with Regex

<PyRunner
  cellId="112-secure-coding-cell-2"
  defaultCode={`import re

# Simulated code snippets that might appear in a PR
code_samples = [
    'db_url = "postgres://admin:SuperSecret123@prod-db:5432/app"',
    'API_KEY = "sk-proj-abc123def456ghi789"',
    'aws_secret = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
    'connection = os.environ["DATABASE_URL"]  # Good practice',
    'token = config.get("auth_token")  # From secret manager',
]

# Secret detection patterns
patterns = [
    (r'["\'](?:postgres|mysql|mongodb)://[^:]+:[^@]+@', "Database connection string with credentials"),
    (r'["\']sk-[a-zA-Z0-9\-]{20,}["\']', "API secret key"),
    (r'["\'](?:AKIA|ASIA)[A-Z0-9]{16}["\']', "AWS access key ID"),
    (r'(?:password|passwd|pwd|secret|token|key)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret assignment"),
]

print("Secret Detection Scanner")
print("=" * 60)

findings = 0
for i, code in enumerate(code_samples, 1):
    detected = []
    for pattern, label in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            detected.append(label)

    if detected:
        findings += 1
        print(f"\n  Line {i}: SECRET DETECTED")
        print(f"    Code: {code[:60]}...")
        for d in detected:
            print(f"    Type: {d}")
    else:
        print(f"\n  Line {i}: Clean")
        print(f"    Code: {code[:60]}")

print(f"\n--- Results ---")
print(f"Scanned: {len(code_samples)} code samples")
print(f"Secrets found: {findings}")
print(f"Recommendation: Add gitleaks to CI to catch these before merge")`}
  timeout={8}
  title="Secret Detection Scanner"
/>

## Code Review Security Checklist

When reviewing code, check for:

1. **Input handling** — Is all user input validated? Is output encoded for context?
2. **Authentication** — Are auth checks present on every endpoint?
3. **Authorization** — Does the code verify the user owns the resource they are accessing?
4. **Secrets** — Are there any hardcoded credentials?
5. **Dependencies** — Are new dependencies pinned and audited?
6. **Error handling** — Do errors leak information? Do they fail secure?
7. **Logging** — Are security-relevant events logged? Are sensitive values excluded from logs?

## Summary

Secure coding is a set of repeatable habits plus automated guardrails. Validate input, encode output, manage dependencies, protect secrets, and automate the checks in CI. Culture and tooling together keep vulnerabilities from shipping.

<Quiz
  chapterSlug="112-secure-coding"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is whitelisting preferred over blacklisting for input validation?",
      options: ["Whitelisting is faster", "Blacklisting always misses some attack patterns; whitelisting only allows known-good input", "Whitelisting requires less code", "Blacklisting is deprecated"],
      correctIndex: 1,
      explanation: "Blacklisting tries to block known-bad patterns, but attackers always find new variations. Whitelisting defines exactly what is acceptable and rejects everything else, which is fundamentally more robust.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What should you do if you accidentally commit a secret to git?",
      options: ["Delete the file and commit again", "Treat it as compromised, rotate the key, and remove it from git history", "Add it to .gitignore", "Rename the variable"],
      correctIndex: 1,
      explanation: "Deleting the file does not remove it from git history. The secret is permanently exposed. You must rotate the key immediately and use tools like BFG Repo-Cleaner to purge it from history.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the purpose of a Software Bill of Materials (SBOM)?",
      options: ["To track developer productivity", "To inventory all components so you can quickly assess CVE exposure", "To replace dependency scanning", "To encrypt the dependency tree"],
      correctIndex: 1,
      explanation: "An SBOM lists every component in your software. When a new CVE is published, you can immediately check whether you ship the affected component and version.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Where should secrets be stored in a production application?",
      options: ["In the source code for easy access", "In a .env file committed to the repository", "In a secrets manager or environment variables injected at deploy time", "In a Kubernetes ConfigMap"],
      correctIndex: 2,
      explanation: "Secrets belong in encrypted, access-controlled stores (HashiCorp Vault, AWS Secrets Manager, environment injection at deploy time). Committing them to code or config files exposes them to anyone with repository access.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What does 'shift left' mean in the context of security?",
      options: ["Moving servers to a different data center", "Integrating security checks early in the development lifecycle", "Shifting security responsibility to the operations team", "Reducing the number of security tools"],
      correctIndex: 1,
      explanation: "Shift left means moving security activities earlier in the development lifecycle: threat modeling during design, SAST during coding, dependency scanning in CI. The earlier a vulnerability is found, the cheaper it is to fix.",
      randomize: true,
    }
  ]}
/>
