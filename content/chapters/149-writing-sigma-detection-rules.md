---
title: "Writing Sigma Detection Rules"
description: "Creating portable detection rules that work across SIEMs to catch attacker techniques."
read_time: 12
code_time: 7
---

## What Is Sigma?

Sigma is a generic signature format for log-based detection. Write once, convert to Splunk SPL, Elastic Query DSL, QRadar AQL, or any SIEM query.

## Sigma Rule Structure

```yaml
title: Suspicious PowerShell Encoded Command
id: a1b2c3d4-e5f6-7890-abcd-ef1234567890
status: experimental
description: Detects encoded PowerShell commands often used by malware
author: Your Name
date: 2024/03/15
tags:
  - attack.execution
  - attack.t1059.001
logsource:
  category: process_creation
  product: windows
detection:
  selection:
    Image|endswith: powershell.exe
    CommandLine|contains: -enc
  condition: selection
level: high
```

## Writing Effective Rules

1. Map to MITRE ATT&CK
2. Use specific field names (not wildcards)
3. Test against known-good and known-bad data
4. Minimize false positives
5. Include references and examples

<Callout type="info" title="Contribute to the community">The SigmaHQ repository has thousands of community-contributed rules. Contributing your detections helps everyone and builds your reputation.</Callout>

## Interactive: Sigma Rule Builder

<PyRunner
  cellId="149-writing-sigma-detection-rules-cell-1"
  defaultCode={`templates = {
    "Process Creation": {
        "logsource": {"category": "process_creation", "product": "windows"},
        "fields": ["Image", "CommandLine", "ParentImage", "User"],
    },
    "File Creation": {
        "logsource": {"category": "file_event", "product": "windows"},
        "fields": ["TargetFilename", "Image", "CreationUtcTime"],
    },
    "Network Connection": {
        "logsource": {"category": "network_connection", "product": "windows"},
        "fields": ["Image", "DestinationIp", "DestinationPort", "SourceIp"],
    },
    "Authentication": {
        "logsource": {"category": "authentication", "product": "linux"},
        "fields": ["User", "SourceAddress", "Result"],
    },
}

print("Sigma Detection Rule Builder")
print("=" * 55)

for name, tmpl in templates.items():
    print(f"\n  {name}:")
    print(f"    Logsource: {tmpl[logsource]}")
    print(f"    Available fields: {, .join(tmpl[fields])}")

print("\n--- Example Rule Generation ---")
rule = """title: Mimikatz Execution Detection
id: example-001
status: experimental
description: Detects Mimikatz execution via process name or command line
logsource:
  category: process_creation
  product: windows
detection:
  selection_name:
    Image|endswith:
      - mimikatz.exe
      - mimikatz.x64.exe
  selection_cmdline:
    CommandLine|contains:
      - sekurlsa::logonpasswords
      - lsadump::dcsync
      - privilege::debug
  condition: selection_name or selection_cmdline
level: critical
tags:
  - attack.credential_access
  - attack.t1003.001"""

print(rule)`}
  timeout={8}
  title="Sigma Rule Builder"
/>

## Summary

Sigma makes detection portable and collaborative. Learn the format, map to ATT&CK, test thoroughly, and contribute to the community. Good detection rules are the blue team equivalent of good exploits.

<Quiz
  chapterSlug="149-writing-sigma-detection-rules"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the main advantage of Sigma over SIEM-specific queries?",
      options: ["Sigma is faster", "Sigma rules are portable across different SIEM platforms", "Sigma requires no conversion", "Sigma is encrypted"],
      correctIndex: 1,
      explanation: "Sigma is a generic format that can be converted to Splunk SPL, Elastic Query DSL, QRadar AQL, and other SIEM query languages. Write once, deploy everywhere.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should detection rules be mapped to MITRE ATT&CK?",
      options: ["It is required by law", "It enables coverage measurement and ensures detections align with known adversary techniques", "ATT&CK generates rules automatically", "It makes rules shorter"],
      correctIndex: 1,
      explanation: "Mapping to ATT&CK lets you measure detection coverage across techniques, identify gaps, and communicate findings in a standardized framework that both red and blue teams understand.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the most important quality of a good detection rule?",
      options: ["It catches everything", "Low false positive rate while catching real threats", "It uses complex regex", "It is very short"],
      correctIndex: 1,
      explanation: "A rule that generates too many false positives gets disabled. A good rule balances detection fidelity with low noise, ensuring analysts trust and act on alerts.",
      randomize: true,
    }
  ]}
/>
