---
title: "Penetration Testing"
description: "The methodology of ethical hacking: scoped, authorized attacks that find weaknesses before adversaries do."
read_time: 12
code_time: 7
---

## What a Pentest Is (and Is Not)

A penetration test is an **authorized**, **scoped** simulation of an attack. It differs from real hacking in one crucial way: you have written permission and clear rules of engagement.

A vulnerability scan is automated and broad. A pentest is manual, deep, and goal-oriented: "Can I reach the payment database from the internet?"

## The Methodology

### 1. Reconnaissance

Gather public information without touching the target:

- **OSINT** — WHOIS, DNS records, employee LinkedIn profiles, GitHub repos.
- **Passive scanning** — Shodan, Censys for exposed services.
- **Social engineering recon** — organizational structure, email formats.

### 2. Scanning & Enumeration

Active probing of the target:

- Port scanning (nmap) to find open services.
- Service version detection.
- Web crawling to map endpoints and parameters.
- Vulnerability scanning for known CVEs.

### 3. Exploitation

Attempt to gain access:

- Exploit known vulnerabilities in identified services.
- Test web application for injection, broken access control.
- Attempt credential attacks (within scope).
- Chain vulnerabilities for greater impact.

### 4. Post-Exploitation

Demonstrate impact:

- Escalate privileges.
- Move laterally to other systems.
- Access sensitive data (and document proof without exfiltrating).
- Establish persistence (if in scope).

### 5. Reporting

The deliverable that drives remediation:

- Executive summary (business impact, no jargon).
- Technical findings with reproduction steps.
- Risk ratings (Critical/High/Medium/Low).
- Prioritized remediation recommendations.

<Callout type="info" title="The report is the product">A pentest that finds holes but cannot communicate them clearly wastes everyone's effort. Evidence, reproduction steps, and prioritized fixes are the deliverable.</Callout>

## Boxes and Access Levels

| Type | Knowledge Given | Simulates |
|---|---|---|
| **Black box** | None | External attacker with no insider info |
| **Grey box** | Partial (user credentials, network diagrams) | Insider or compromised user |
| **White box** | Full (source code, architecture, credentials) | Thorough code-level review |

Grey box is the most common and cost-effective: it simulates a realistic attacker who has gained initial foothold.

## Ethics & Law

Always have:

- **Signed scope** — exactly which systems, IPs, and applications are in scope.
- **Time window** — when testing may occur.
- **Authorization letter** — legal protection for the tester.
- **Emergency contact** — who to call if something breaks.
- **Rules of engagement** — what is off-limits (DoS, social engineering, physical).

<Callout type="danger" title="Testing outside scope is illegal">Even with good intentions, scanning or attacking systems not in your signed authorization violates computer fraud laws. Scope is not a suggestion — it is a legal boundary.</Callout>

## Interactive: Pentest Findings Parser

<PyRunner
  cellId="113-penetration-testing-cell-1"
  defaultCode={`import re

# Simulated pentest findings
findings_raw = """
[CRITICAL] SQL Injection in /api/search
  CVSS: 9.8 | CWE-89
  Endpoint: /api/search?q=
  Impact: Full database access, data exfiltration
  Evidence: Input ' OR 1=1-- returned all records

[HIGH] Broken Access Control in /api/admin/users
  CVSS: 8.1 | CWE-284
  Endpoint: /api/admin/users
  Impact: Regular users can access admin functions
  Evidence: Authenticated as user, accessed admin endpoint, got 200

[HIGH] Outdated Apache Tomcat 9.0.73
  CVSS: 7.5 | CVE-2023-XXXX
  Service: Apache Tomcat on port 8080
  Impact: Known RCE vulnerability
  Evidence: Version detected via nmap -sV

[MEDIUM] Missing Content-Security-Policy header
  CVSS: 5.3 | CWE-693
  Endpoint: All responses
  Impact: Increases XSS exploitation impact
  Evidence: Response headers inspected, no CSP present

[LOW] Server version disclosure
  CVSS: 3.1 | CWE-200
  Endpoint: All responses
  Impact: Information disclosure aids attacker reconnaissance
  Evidence: Server: Apache/2.4.54 (Ubuntu) in response headers
"""

# Parse findings
findings = []
for block in findings_raw.strip().split("\n\n"):
    lines = block.strip().split("\n")
    if not lines:
        continue
    severity_title = lines[0].strip()
    match = re.match(r'\[(\w+)\] (.+)', severity_title)
    if match:
        finding = {"severity": match.group(1), "title": match.group(2)}
        for line in lines[1:]:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                finding[key.strip().lower()] = val.strip()
        findings.append(finding)

# Display report
severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
findings.sort(key=lambda f: severity_order.get(f["severity"], 99))

print("Penetration Test Findings Report")
print("=" * 60)
print(f"Total findings: {len(findings)}\n")

for i, f in enumerate(findings, 1):
    print(f"  {i}. [{f['severity']}] {f['title']}")
    if 'cvss' in f:
        print(f"     CVSS: {f['cvss']}")
    if 'impact' in f:
        print(f"     Impact: {f['impact']}")
    print()

# Summary stats
from collections import Counter
sev_counts = Counter(f["severity"] for f in findings)
print("--- Summary ---")
for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    print(f"  {sev}: {sev_counts.get(sev, 0)}")

print("\n--- Priority Remediation Order ---")
for i, f in enumerate(findings, 1):
    print(f"  {i}. Fix {f['title']} ({f['severity']})")`}
  timeout={8}
  title="Pentest Findings Parser"
/>

## Tools of the Trade

| Phase | Tools |
|---|---|
| Recon | Maltego, theHarvester, Shodan |
| Scanning | Nmap, Nikto, Burp Suite, OWASP ZAP |
| Exploitation | Metasploit, Burp Repeater, custom scripts |
| Post-exploitation | Mimikatz, BloodHound, Netcat |
| Reporting | Pwndoc, Serpico, custom templates |

<Callout type="tip" title="Tools assist, skills decide">A tool finds the low-hanging fruit. Your understanding of how systems work finds the chained, creative, and business-impacting vulnerabilities that tools miss.</Callout>

## Summary

Pentesting applies attacker technique under legal constraints to find and fix weaknesses before adversaries do. Master the methodology, respect the scope, and remember that the report is the product.

<Quiz
  chapterSlug="113-penetration-testing"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the key difference between a pentest and malicious hacking?",
      options: ["Pentests use better tools", "A pentest has written authorization and a defined scope", "Pentests are always automated", "Malicious hackers do not write reports"],
      correctIndex: 1,
      explanation: "The defining difference is authorization. A pentest operates under a signed scope with clear rules of engagement. Without written authorization, the same activities are illegal.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What type of pentest gives the tester partial knowledge like user credentials?",
      options: ["Black box", "Grey box", "White box", "Red team"],
      correctIndex: 1,
      explanation: "Grey box testing provides partial knowledge (like user credentials or network diagrams), simulating an attacker who has gained initial foothold or an insider with limited access.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the primary deliverable of a penetration test?",
      options: ["A list of exploited vulnerabilities", "A report with findings, evidence, risk ratings, and remediation recommendations", "A patched system", "A tool configuration file"],
      correctIndex: 1,
      explanation: "The report is the product. It must communicate findings clearly to both executives (business impact) and developers (reproduction steps and fixes). Without a good report, the pentest's value is lost.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why is the reconnaissance phase important?",
      options: ["It is required by law", "It maps the attack surface and identifies entry points before active testing", "It replaces the need for exploitation", "It generates the final report"],
      correctIndex: 1,
      explanation: "Reconnaissance gathers information about the target: open ports, services, technologies, and potential entry points. This intelligence guides the testing effort toward the most promising attack paths.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What should you do if you discover a vulnerability outside the agreed scope?",
      options: ["Exploit it anyway to show thoroughness", "Stop and contact the client to discuss expanding the scope", "Ignore it completely", "Publish it publicly"],
      correctIndex: 1,
      explanation: "Testing outside the signed scope is illegal. If you discover something outside scope, stop, document what you found passively, and contact the client to discuss whether to expand the authorization.",
      randomize: true,
    }
  ]}
/>
