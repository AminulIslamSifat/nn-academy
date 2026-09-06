---
title: "Building a Home SOC Lab"
description: "Setting up Wazuh, Elastic, or Splunk Free to practice detection engineering at home."
read_time: 12
code_time: 7
---

## Why a Home Lab?

Theory without practice does not stick. A home SOC lab lets you generate attacks, write detections, and build the muscle memory that employers value.

## Recommended Stack

| Component | Free Option | Purpose |
|---|---|---|
| SIEM | Wazuh / Elastic Security | Log aggregation, detection, alerting |
| Endpoint Agent | Wazuh Agent / Sysmon | Telemetry collection |
| Attack VM | Kali Linux / Parrot | Offensive tools |
| Target VMs | Windows 10 eval, Ubuntu | Victims |
| Network | VirtualBox host-only | Isolated lab network |

## Setup Steps

1. Create isolated virtual network
2. Deploy Wazuh manager (Docker or VM)
3. Install agents on target VMs
4. Configure Sysmon on Windows targets
5. Generate test attacks (Atomic Red Team)
6. Write and test detection rules
7. Build dashboards

<Callout type="tip" title="Atomic Red Team is your best friend">Atomic Red Team provides hundreds of mapped-to-ATT&CK test cases. Run them against your lab to validate detections. If your SIEM does not alert, your detection has a gap.</Callout>

## Practice Workflow

```
1. Pick an ATT&CK technique (e.g., T1059.001 PowerShell)
2. Run the Atomic Red Team test
3. Check if your SIEM detected it
4. If not, write/improve the detection rule
5. Re-run and verify
6. Document the detection
```

## Summary

A home SOC lab is the single best investment for aspiring blue teamers. Start simple (Wazuh + one Windows VM), generate attacks with Atomic Red Team, and iterate on detections. Document everything for your portfolio.

<Quiz
  chapterSlug="151-building-a-home-soc-lab"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is Atomic Red Team?",
      options: ["A commercial SIEM product", "A library of ATT&CK-mapped test cases for validating detections", "A red team consulting firm", "A malware analysis tool"],
      correctIndex: 1,
      explanation: "Atomic Red Team provides hundreds of simple, reproducible tests mapped to MITRE ATT&CK techniques. Running these tests validates whether your detection rules actually catch the intended behavior.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should lab networks be isolated?",
      options: ["To save bandwidth", "To prevent lab malware or attack traffic from reaching real networks", "Isolation is not necessary", "To comply with GDPR"],
      correctIndex: 1,
      explanation: "Lab environments run malware and offensive tools. Network isolation (host-only adapters, no internet routing) prevents accidental spread to production or home networks.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the recommended first step when building a home SOC lab?",
      options: ["Buy expensive hardware", "Deploy a free SIEM (Wazuh/Elastic) with one Windows target VM and Sysmon", "Get certified first", "Build a full enterprise replica"],
      correctIndex: 1,
      explanation: "Start simple and expand. A single SIEM instance, one Windows VM with Sysmon, and Atomic Red Team tests provide immediate hands-on learning without significant cost or complexity.",
      randomize: true,
    }
  ]}
/>
