---
title: "Security Automation with Python"
description: "Script common security tasks: IOC extraction, log analysis, entropy scoring, and report generation."
read_time: 12
code_time: 8
---

## Why Automate?

Security teams drown in repetitive work: parsing logs, extracting indicators, checking configurations, generating reports. Python is the lingua franca of security scripting because it is readable, has a rich standard library, and integrates with everything.

<Callout type="info" title="Automate the boring stuff">If you do a task more than three times, script it. The time investment pays for itself quickly and eliminates human error in repetitive analysis.</Callout>

## Common Security Scripting Tasks

| Task | Python Approach |
|---|---|
| Extract IOCs from logs | Regex + string parsing |
| Detect suspicious strings | Shannon entropy calculation |
| Parse firewall rules | Structured text parsing |
| Check SSL/TLS configs | Socket + ssl module |
| Generate reports | String formatting, JSON output |
| Correlate events | Dictionaries, sets, timestamps |

## Shannon Entropy for String Analysis

High-entropy strings in logs or binaries often indicate:

- Encoded/encrypted payloads
- Base64-encoded commands
- Randomly generated filenames (malware)
- Compressed data

$$H = -\sum_{i} p_i \log_2 p_i$$

Where $p_i$ is the frequency of character $i$ in the string.

## Interactive: IOC Extraction from Logs

<PyRunner
  cellId="127-security-automation-python-cell-1"
  defaultCode={`import re
from collections import Counter

# Simulated raw log blob from multiple sources
log_blob = """
2024-03-15 08:12:01 WARN Connection from 185.220.101.45 to internal 10.0.0.5:445
2024-03-15 08:12:05 ALERT DNS query for evil-c2.example.com from 10.0.0.5
2024-03-15 08:13:00 WARN HTTP request to http://malware-drop.example.net/payload.exe
2024-03-15 08:14:00 INFO User jdoe logged in from 172.16.0.20
2024-03-15 08:15:00 ALERT Outbound connection 10.0.0.5:49152 -> 91.219.236.174:8443
2024-03-15 08:16:00 WARN Email received from phisher@evil-domain.com with attachment invoice.pdf.exe
2024-03-15 08:17:00 INFO Scheduled task created: C:\temp\.hidden\beacon.exe
2024-03-15 08:18:00 ALERT DNS query for c2-callback.example.org from 10.0.0.5
2024-03-15 08:19:00 WARN Failed SSH login from 185.220.101.45 (attempt 47)
"""

# IOC extraction patterns
patterns = {
    "IP Address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "Domain": r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b',
    "URL": r'https?://[^\s"<>]+',
    "Email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "File Path": r'[A-Za-z]:\[^\s]+',
}

print("IOC Extraction from Log Data")
print("=" * 55)

all_iocs = {}
for ioc_type, pattern in patterns.items():
    matches = set(re.findall(pattern, log_blob))
    # Filter out common false positives
    if ioc_type == "IP Address":
        matches = {m for m in matches if not m.startswith("10.") and not m.startswith("172.")}
    if ioc_type == "Domain":
        matches = {m for m in matches if "example" in m}
    if matches:
        all_iocs[ioc_type] = sorted(matches)
        print(f"\n{ioc_type} ({len(matches)} found):")
        for m in sorted(matches):
            print(f"  {m}")

# Count IOC mentions for prioritization
all_text = log_blob
print("\n--- IOC Frequency (prioritization) ---")
for ioc_type, iocs in all_iocs.items():
    for ioc in iocs:
        count = all_text.count(ioc)
        print(f"  {ioc:35s} mentioned {count}x")`}
  timeout={8}
  title="IOC Extraction Script"
/>

## Interactive: Shannon Entropy Analyzer

<PyRunner
  cellId="127-security-automation-python-cell-2"
  defaultCode={`import math
from collections import Counter

def shannon_entropy(s):
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((count/length) * math.log2(count/length) for count in freq.values())

# Test strings - normal vs suspicious
test_strings = [
    ("Normal filename", "report_2024_q1.pdf"),
    ("Normal path", "/var/log/syslog"),
    ("Base64 encoded payload", "aGVsbG8gd29ybGQgdGhpcyBpcyBhIHNlY3JldCBwYXlsb2Fk"),
    ("Random malware name", "x7kQ9mP2vR4wL8nJ"),
    ("Encrypted-looking blob", "4f8a2c1e9b3d7f6a5e0c8d2b1a9f4e7d"),
    ("Normal command", "systemctl restart nginx"),
    ("Suspicious registry key", "HKCU\Software\Microsoft\Windows\CurrentVersion\Run\x7kQ9m"),
]

print("Shannon Entropy String Analyzer")
print("=" * 60)
print(f"{'Description':25s} {'Entropy':>8s} {'Assessment'}")
print(f"{'-'*25} {'-'*8} {'-'*20}")

for desc, s in test_strings:
    h = shannon_entropy(s)
    if h > 4.0:
        assessment = "SUSPICIOUS (high entropy)"
    elif h > 3.0:
        assessment = "Moderate (review)"
    else:
        assessment = "Normal"
    print(f"  {desc:25s} {h:8.2f}  {assessment}")

print("\nThreshold guide:")
print("  < 3.0  : Normal text/paths")
print("  3.0-4.0: Mixed content, review")
print("  > 4.0  : Likely encoded/encrypted/random")`}
  timeout={8}
  title="Entropy-Based String Analyzer"
/>

## Building a Security Toolkit

Structure your scripts for reuse:

```python
# security_utils.py
import re
import math
from collections import Counter

def extract_iocs(text, patterns=None):
    """Extract IOCs from text using regex patterns."""
    # ... implementation

def entropy(s):
    """Calculate Shannon entropy of a string."""
    # ... implementation

def parse_log_line(line):
    """Parse a syslog-format line into a dict."""
    # ... implementation
```

<Callout type="tip" title="Keep scripts modular">Write small, composable functions. A script that extracts IOCs, one that scores entropy, and one that formats reports can be combined in different ways for different investigations.</Callout>

## Summary

Python security automation saves hours of manual work and reduces errors. Start with IOC extraction and entropy analysis, build a reusable toolkit, and integrate with your SIEM and ticketing systems. The goal is to let the machine do the repetitive work so you can focus on analysis.

<Quiz
  chapterSlug="127-security-automation-python"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does a high Shannon entropy score in a string suggest?",
      options: ["The string is a normal file path", "The string is likely encoded, encrypted, or randomly generated", "The string is too short to analyze", "The string contains only numbers"],
      correctIndex: 1,
      explanation: "High entropy means the character distribution is close to uniform, which is characteristic of encoded data, encrypted blobs, or random strings rather than natural language or file paths.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should you filter internal IPs when extracting IOC candidates from logs?",
      options: ["Internal IPs are always malicious", "Internal IPs are expected traffic and would create false positives in the IOC list", "Regex cannot match internal IPs", "Internal IPs are encrypted"],
      correctIndex: 1,
      explanation: "Internal IPs (10.x, 172.16-31.x, 192.168.x) are normal infrastructure. Including them as IOCs would flood your threat intel with false positives.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the benefit of writing modular security scripts?",
      options: ["They run faster", "Small composable functions can be combined for different investigations and are easier to maintain", "They require less disk space", "They do not need testing"],
      correctIndex: 1,
      explanation: "Modular scripts with small, focused functions can be mixed and matched for different tasks. They are also easier to test, debug, and share with teammates.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which Python standard library module is most useful for IOC extraction from text?",
      options: ["numpy", "re (regular expressions)", "math", "itertools"],
      correctIndex: 1,
      explanation: "Regular expressions are the primary tool for pattern matching in text. IOC extraction relies on regex patterns to find IPs, domains, URLs, emails, and file paths in unstructured log data.",
      randomize: true,
    }
  ]}
/>
