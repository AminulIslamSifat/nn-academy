---
title: "The OWASP Top 10"
description: "The ten most critical web application risks and how each one gets exploited in practice."
read_time: 12
code_time: 6
---

## What Is OWASP?

The Open Web Application Security Project maintains a consensus list of the most critical web risks, updated every few years based on real-world vulnerability data. It is the de-facto starting point for any web security assessment.

## The Current Top 10

1. **Broken Access Control** — acting on data you should not reach.
2. **Cryptographic Failures** — exposing or mishandling sensitive data.
3. **Injection** — untrusted input executed as code or queries.
4. **Insecure Design** — flaws in the design, not the implementation.
5. **Security Misconfiguration** — default creds, open cloud storage, verbose errors.
6. **Vulnerable & Outdated Components** — shipping known-bad libraries.
7. **Identification & Auth Failures** — weak sessions and credential handling.
8. **Software & Data Integrity Failures** — insecure CI/CD and deserialization.
9. **Logging & Monitoring Failures** — breaches go unnoticed.
10. **Server-Side Request Forgery** — forcing the server to fetch attacker-chosen URLs.

<Callout type="info" title="Broken Access Control is number one">Access-control failures have topped the list because they are pervasive and often invisible to automated scanners. Always test authorization, not just authentication.</Callout>

## Deep Dive: Broken Access Control

Access control enforces that users can only do what they are authorized to do. When it breaks:

- **Horizontal escalation** — User A accesses User B's data by changing an ID.
- **Vertical escalation** — A regular user accesses admin functionality.
- **IDOR** — Insecure Direct Object References expose internal IDs.

```python
# Vulnerable: no ownership check
@app.route("/api/orders/<order_id>")
def get_order(order_id):
    return db.get_order(order_id)  # Any user can access any order

# Fixed: verify ownership
@app.route("/api/orders/<order_id>")
def get_order(order_id):
    order = db.get_order(order_id)
    if order.user_id != current_user.id:
        abort(403)
    return order
```

## Deep Dive: Injection

Injection was number one for years and remains in the top three. The root cause is mixing untrusted data with code. See the dedicated chapter on injection attacks for SQL injection, command injection, and XSS in depth.

## Deep Dive: Security Misconfiguration

This is the most common finding in assessments:

- Default admin accounts still active.
- Cloud storage buckets open to the public.
- Verbose error messages revealing stack traces.
- Unnecessary features enabled (directory listing, debug mode).
- Missing security headers (CSP, HSTS, X-Frame-Options).

<Callout type="warning" title="Default credentials are still a thing">Scanners routinely find admin panels with admin/admin. Change every default credential during deployment and disable accounts you do not use.</Callout>

## Deep Dive: Cryptographic Failures

Sensitive data exposed through:

- Transmitting data over HTTP instead of HTTPS.
- Storing passwords with fast hashes (MD5, SHA-1) or in plaintext.
- Using weak algorithms (DES, RC4, ECB mode).
- Hardcoding encryption keys in source code.

## How to Use This List

Treat each item as a checklist during design review and testing:

1. **Design phase** — for each feature, ask which Top 10 items could apply.
2. **Code review** — check for the specific patterns that cause each item.
3. **Testing** — actively try to trigger each one in your application.
4. **Monitoring** — ensure you have detection for the attacks each item represents.

## Interactive: OWASP Configuration Checker

<PyRunner
  cellId="109-owasp-top-ten-cell-1"
  defaultCode={`# Simulate checking an application config against OWASP Top 10
config = {
    "https_enforced": True,
    "password_hashing": "bcrypt",
    "access_control_checked": False,
    "error_messages": "verbose",
    "security_headers": {"CSP": False, "HSTS": True, "X-Frame-Options": True},
    "dependency_scan": False,
    "logging_enabled": True,
    "debug_mode": True,
    "default_admin_disabled": False,
}

checks = [
    ("A02 Cryptographic Failures", lambda c: c["https_enforced"] and c["password_hashing"] in ("bcrypt", "argon2", "scrypt")),
    ("A01 Broken Access Control", lambda c: c["access_control_checked"]),
    ("A05 Security Misconfiguration", lambda c: c["error_messages"] != "verbose" and not c["debug_mode"] and c["default_admin_disabled"]),
    ("A06 Vulnerable Components", lambda c: c["dependency_scan"]),
    ("A09 Logging Failures", lambda c: c["logging_enabled"]),
]

print("OWASP Top 10 Configuration Assessment")
print("=" * 55)
passed = 0
failed = 0
for name, check_fn in checks:
    result = check_fn(config)
    status = "PASS" if result else "FAIL"
    if result:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}")

print(f"\nScore: {passed}/{len(checks)} checks passed")
if failed > 0:
    print(f"\nAction items:")
    if not config["access_control_checked"]:
        print("  - Implement and test authorization checks on every endpoint")
    if config["error_messages"] == "verbose":
        print("  - Switch to generic error messages in production")
    if config["debug_mode"]:
        print("  - Disable debug mode in production")
    if not config["default_admin_disabled"]:
        print("  - Remove or disable default admin account")
    if not config["dependency_scan"]:
        print("  - Add dependency scanning to CI pipeline")`}
  timeout={8}
  title="OWASP Configuration Checker"
/>

## Interactive: Attack Surface Mapper

<PyRunner
  cellId="109-owasp-top-ten-cell-2"
  defaultCode={`# Map application endpoints to OWASP risk categories
endpoints = [
    {"path": "/api/user/profile", "auth": True, "input": True, "data_sensitivity": "medium"},
    {"path": "/api/admin/users", "auth": True, "input": False, "data_sensitivity": "high"},
    {"path": "/api/orders/<id>", "auth": True, "input": True, "data_sensitivity": "medium"},
    {"path": "/api/upload", "auth": True, "input": True, "data_sensitivity": "low"},
    {"path": "/api/search", "auth": False, "input": True, "data_sensitivity": "low"},
    {"path": "/api/payment", "auth": True, "input": True, "data_sensitivity": "high"},
]

def assess_risk(ep):
    risks = []
    if ep["data_sensitivity"] == "high" and not ep["auth"]:
        risks.append("A01: High-sensitivity data without auth")
    if ep["input"]:
        risks.append("A03: User input (injection risk)")
    if ep["data_sensitivity"] == "high":
        risks.append("A02: Sensitive data handling")
    if not ep["auth"]:
        risks.append("A07: Unauthenticated endpoint")
    return risks

print("Attack Surface Risk Mapping")
print("=" * 60)
for ep in endpoints:
    risks = assess_risk(ep)
    risk_level = "HIGH" if len(risks) >= 2 else "MED" if len(risks) == 1 else "LOW"
    print(f"\n  {ep['path']} [{risk_level}]")
    for r in risks:
        print(f"    - {r}")

print("\n--- Priority Testing Order ---")
scored = [(ep["path"], len(assess_risk(ep))) for ep in endpoints]
scored.sort(key=lambda x: -x[1])
for i, (path, score) in enumerate(scored, 1):
    print(f"  {i}. {path} (risk score: {score})")`}
  timeout={8}
  title="Attack Surface Mapper"
/>

## Summary

The OWASP Top 10 gives you the vocabulary of web risk. Use it as a living checklist throughout the development lifecycle: during design, code review, testing, and monitoring. The next chapters drill into the individual items in depth.

<Quiz
  chapterSlug="109-owasp-top-ten"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is currently ranked as the number one OWASP Top 10 risk?",
      options: ["Injection", "Broken Access Control", "Cryptographic Failures", "Security Misconfiguration"],
      correctIndex: 1,
      explanation: "Broken Access Control moved to number one because access-control failures are pervasive, often invisible to automated scanners, and can expose entire datasets.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is an Insecure Direct Object Reference (IDOR)?",
      options: ["A type of SQL injection", "Exposing internal object IDs that allow users to access resources they should not", "A cross-site scripting attack", "A misconfigured firewall rule"],
      correctIndex: 1,
      explanation: "IDOR occurs when an application uses user-supplied IDs to access objects directly without verifying authorization. Changing the ID lets a user access another user's data.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Which OWASP item covers default credentials, open cloud storage, and verbose errors?",
      options: ["Injection", "Broken Access Control", "Security Misconfiguration", "Cryptographic Failures"],
      correctIndex: 2,
      explanation: "Security Misconfiguration (A05) covers deployment and configuration weaknesses: default accounts, unnecessary features, verbose errors, and missing security headers.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why is the OWASP Top 10 described as a consensus list?",
      options: ["It is voted on by the public", "It is based on real-world vulnerability data and expert agreement", "It is mandated by governments", "It is generated by AI"],
      correctIndex: 1,
      explanation: "The OWASP Top 10 is compiled from real-world vulnerability data, industry surveys, and expert consensus. It reflects what the community agrees are the most critical risks.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "When should you check against the OWASP Top 10?",
      options: ["Only during annual audits", "Throughout the lifecycle: design, code review, testing, and monitoring", "Only after a breach", "Only during penetration tests"],
      correctIndex: 1,
      explanation: "The Top 10 is most effective when used continuously: during design (threat modeling), code review (pattern checking), testing (active exploitation attempts), and monitoring (detection rules).",
      randomize: true,
    }
  ]}
/>
