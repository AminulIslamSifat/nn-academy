---
title: "DevSecOps & CI/CD Security"
description: "Shifting security left, securing the pipeline, and protecting the software supply chain."
read_time: 12
code_time: 7
---

## The Shift-Left Philosophy

Traditional security reviewed code at the end of development — after architecture decisions were locked in. **DevSecOps** embeds security checks at every stage: design, commit, build, test, deploy, and runtime.

The cost curve is brutal: fixing a vulnerability found in production costs **30-100x** more than catching it during design.

## The CI/CD Pipeline as Attack Surface

A modern pipeline touches many systems:

1. **Source control** (GitHub, GitLab)
2. **CI server** (Jenkins, GitHub Actions, GitLab CI)
3. **Container registry** (Docker Hub, ECR, GCR)
4. **Artifact store** (Nexus, Artifactory)
5. **Deployment target** (Kubernetes, AWS, GCP)

Each of these is a target. The SolarWinds attack (2020) compromised the **build system** to inject malware into signed updates, reaching 18,000 organizations.

<Callout type="danger" title="The pipeline is a privileged target">A compromised CI system can push malicious code to every environment. Treat pipeline credentials with the same care as production secrets.</Callout>

## Static Analysis (SAST)

SAST tools scan source code for known vulnerability patterns without executing it.

| Tool | Language Focus |
|---|---|
| Semgrep | Multi-language, rule-based |
| Bandit | Python |
| ESLint security plugin | JavaScript |
| SpotBugs + FindSecBugs | Java |

```yaml
# GitHub Actions example: run Semgrep on every PR
- name: Semgrep scan
  run: semgrep ci --config=p/owasp-top-ten
```

## Dynamic Analysis (DAST)

DAST tools attack a **running** application from the outside, like a real attacker. OWASP ZAP is the most common open-source DAST tool. It finds issues SAST cannot: authentication bypasses, runtime injection, misconfigured headers.

## Software Composition Analysis (SCA)

Your project depends on libraries, which depend on other libraries. SCA tools map this **dependency tree** and flag known CVEs.

```bash
# npm audit checks the Node.js dependency tree
npm audit

# pip-audit for Python
pip-audit
```

The **dependency confusion** attack exploits package managers: an attacker publishes a private-sounding package name to a public registry with a higher version number, and the build system pulls the malicious public version instead of the internal one.

## Securing the Pipeline Itself

### Secrets in CI

- Use the CI platform's secret store (GitHub Secrets, Vault integration), never plain-text in YAML.
- Scope secrets to the minimum environments and branches.
- Rotate on any suspicion of leak.

### Pinned Actions

```yaml
# Bad: mutable tag
- uses: actions/checkout@v4

# Good: pinned to commit SHA
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11
```

A mutable tag like `@v4` can be republished. Pinning to the commit SHA guarantees you run the code you reviewed.

### Branch Protection

Require pull request reviews, status checks, and signed commits before merging to main. This prevents a single compromised account from pushing malicious code directly.

## Interactive: Simulate a Supply Chain Check

<PyRunner
  cellId="119-devsecops-cicd-security-cell-1"
  defaultCode={`import re

# Simulated dependency lockfile entries
deps = [
  {"name": "express", "version": "4.18.2", "source": "npm"},
  {"name": "lodash", "version": "4.17.20", "source": "npm"},
  {"name": "requests", "version": "2.31.0", "source": "pypi"},
  {"name": "urllib3", "version": "1.26.5", "source": "pypi"},
  {"name": "left-pad", "version": "1.3.0", "source": "npm"},
]

# Known-bad versions (simulated CVE database)
known_bad = {
  ("lodash", "4.17.20"): "CVE-2021-23337: Command injection",
  ("urllib3", "1.26.5"): "CVE-2023-43804: Cookie header leak",
}

def check_dependency(dep):
    key = (dep["name"], dep["version"])
    if key in known_bad:
      return "VULNERABLE", known_bad[key]
    return "OK", None

print("Supply Chain Dependency Audit")
print("=" * 55)
vuln_count = 0
for dep in deps:
    status, detail = check_dependency(dep)
    marker = "X" if status == "VULNERABLE" else " "
    print(f"  [{marker}] {dep['name']:12s} v{dep['version']:10s} ({dep['source']:5s}) -> {status}")
    if detail:
        print(f"      {detail}")
        vuln_count += 1

print(f"\nResult: {vuln_count} vulnerable dependencies found out of {len(deps)}")
print("Action: Update lodash to >=4.17.21, urllib3 to >=1.26.17")`}
  timeout={8}
  title="Dependency Vulnerability Scanner"
/>

## Interactive: Detect Secrets in Code

<PyRunner
  cellId="119-devsecops-cicd-security-cell-2"
  defaultCode={`import re

# Simulated code snippet that might appear in a PR
code = """
import os
API_KEY = "sk-live-4eC39HqLyjWDarjtT1zdp7dc"
DB_PASSWORD = "hunter2"
connection = f"postgres://admin:{DB_PASSWORD}@db:5432/prod"
webhook_url = "https://hooks.slack.com/services/T00/B00/xxxyyyzzz"
"""

# Regex patterns for common secret formats
patterns = [
  (r'[Aa][Pp][Ii][_\-]?[Kk][Ee][Yy]\s*=\s*["\']([^"\']+)["\']', "API Key"),
  (r'[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]\s*=\s*["\']([^"\']+)["\']', "Password"),
  (r'[Pp][Oo][Ss][Tt][Gg][Rr][Ee][Ss]://[^\s]+:[^\s]+@', "Connection string with creds"),
  (r'https://hooks\.slack\.com/services/[^\s"\']+', "Slack webhook URL"),
]

print("Secret Scanner Results")
print("=" * 50)
found = 0
for pattern, label in patterns:
    matches = re.findall(pattern, code)
    for m in matches:
        # Mask the secret for display
        masked = m[:4] + "*" * (len(m) - 8) + m[-4:] if len(m) > 8 else "****"
        print(f"  FOUND [{label}]: {masked}")
        found += 1

if found == 0:
    print("  No secrets detected.")
else:
    print(f"\n  {found} potential secret(s) found. Block this commit!")`}
  timeout={8}
  title="Secret Detection Scanner"
/>

## The SBOM

A **Software Bill of Materials** lists every component in your artifact, like a nutrition label. Formats include **SPDX** and **CycloneDX**. Generate one at build time so that when a new CVE drops, you can answer "are we affected?" in minutes instead of days.

<Callout type="info" title="SBOMs are becoming mandatory">The US Executive Order on Cybersecurity (2021) and the EU Cyber Resilience Act both push for SBOMs in software sold to governments. Even if you are not selling software, an SBOM accelerates your own incident response.</Callout>

## Summary

DevSecOps is not a tool you buy — it is a set of automated gates in the pipeline: SAST at commit, SCA at build, DAST before deploy, secret scanning everywhere, pinned dependencies, and an SBOM for rapid CVE triage. The pipeline itself must be hardened like any production system.

<Quiz
  chapterSlug="119-devsecops-cicd-security"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What was the key technique in the SolarWinds supply chain attack?",
      options: ["Phishing employees for credentials", "Compromising the build system to inject malware into signed updates", "Exploiting a zero-day in the SolarWinds web interface", "DNS hijacking of the update server"],
      correctIndex: 1,
      explanation: "The attackers compromised SolarWinds build infrastructure and injected a backdoor (SUNBURST) into digitally signed software updates, which were then distributed to all customers.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should you pin GitHub Actions to a commit SHA instead of a version tag?",
      options: ["Tags are slower to resolve", "A mutable tag can be republished with different code", "SHA pinning is required by GitHub", "Tags do not support caching"],
      correctIndex: 1,
      explanation: "A tag like @v4 can be moved to point at different code. Pinning to the full commit SHA ensures you always run the exact code you audited.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does a dependency confusion attack exploit?",
      options: ["Weak encryption in package registries", "The package manager preferring a higher version from a public registry over a private one", "Buffer overflows in the dependency resolver", "Unauthenticated access to private registries"],
      correctIndex: 1,
      explanation: "If a private package name is published publicly with a higher version number, some package managers will pull the public (attacker-controlled) version instead of the internal one.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the primary purpose of a Software Bill of Materials (SBOM)?",
      options: ["To speed up the build process", "To list all components so you can quickly determine CVE exposure", "To encrypt the dependency tree", "To replace the need for SCA tools"],
      correctIndex: 1,
      explanation: "An SBOM is an inventory of every component in your software. When a new CVE is published, you can immediately check if you ship the affected component.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Where should CI/CD secrets be stored?",
      options: ["In the pipeline YAML file for visibility", "In a .env file committed to the repo", "In the CI platform secret store or an external vault", "In a Kubernetes ConfigMap"],
      correctIndex: 2,
      explanation: "Secrets belong in encrypted, access-controlled stores (GitHub Secrets, HashiCorp Vault). Committing them to code or config files exposes them to anyone with repo access.",
      randomize: true,
    }
  ]}
/>
