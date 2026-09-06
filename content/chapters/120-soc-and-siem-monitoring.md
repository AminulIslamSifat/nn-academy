---
title: "SOC & SIEM Monitoring"
description: "How a Security Operations Center detects, triages, and escalates threats using log aggregation."
read_time: 12
code_time: 6
---

## What a SOC Does

A **Security Operations Center** is the team (and tooling) that watches for threats 24/7. Its core loop is: **collect** logs, **detect** anomalies, **triage** alerts, **investigate** incidents, and **respond**.

The SOC does not prevent attacks — it **detects and limits** them. Prevention is the job of firewalls, endpoint protection, and secure code. The SOC catches what prevention misses.

## Log Sources

A SOC ingests logs from every layer:

| Source | What It Tells You |
|---|---|
| Firewall / IDS | Blocked connections, intrusion attempts |
| Endpoint (EDR) | Process creation, file writes, registry changes |
| Authentication (AD / IdP) | Logins, failures, privilege escalation |
| Web server / WAF | HTTP requests, blocked payloads |
| DNS resolver | Lookups for known-bad domains |
| Cloud audit trail | API calls, config changes |

<Callout type="info" title="You cannot detect what you do not log">The first SOC maturity question is not 'do we have alerts?' but 'do we have logs?' If a server is not sending logs, an attacker can operate there invisibly.</Callout>

## The SIEM

A **Security Information and Event Management** system aggregates, normalizes, and correlates logs from all sources. Popular SIEMs include Splunk, Microsoft Sentinel, Elastic Security, and the open-source Wazuh.

### What a SIEM Actually Does

1. **Ingest** — receive logs via agents, syslog, or API.
2. **Normalize** — parse different formats into a common schema (IP, user, action, timestamp).
3. **Correlate** — combine events across sources. A failed SSH login from the same IP that later triggers a WAF alert is more interesting than either alone.
4. **Alert** — fire a notification when a rule or anomaly threshold is crossed.
5. **Retain** — keep logs for forensic and compliance purposes.

### Detection Rules

A detection rule is a query over the log stream:

```
# Pseudo-SIEM rule
ALERT WHEN count(failed_login) BY source_ip > 10 WITHIN 5m
  AND source_ip NOT IN whitelist
```

### Anomaly Detection

Rules catch known patterns. **Anomaly detection** builds a baseline of normal behavior and flags deviations. This is where the neural-network skills from Module 1 connect directly: an autoencoder or isolation forest can model normal log patterns and flag outliers.

<Callout type="tip" title="Start with high-fidelity rules">A SOC drowning in false positives is worse than no SOC. Start with a small set of high-confidence rules (brute force, impossible travel, known-bad IOCs) and expand from there.</Callout>

## Alert Triage

When an alert fires, the analyst asks:

1. **Is it a true positive?** Check context — is the user on vacation? Is the IP in a known-bad range?
2. **What is the scope?** One host or the whole fleet?
3. **What is the urgency?** A ransomware beacon is critical; a port scan from a scanner is low.
4. **What is the next step?** Contain, escalate, or close as false positive.

### The Triage Funnel

```
Raw events (millions/day)
  -> Correlated alerts (hundreds/day)
    -> Triage queue (dozens/day)
      -> Confirmed incidents (single digits/day)
```

Most alerts are noise. The SOC's job is to reduce the funnel efficiently without missing real threats.

## Interactive: Parse and Correlate Log Events

<PyRunner
  cellId="120-soc-siem-monitoring-cell-1"
  defaultCode={`import re
from collections import defaultdict

# Simulated raw security logs
raw_logs = [
  "2024-03-15 08:12:01 AUTH FAILED user=admin src=10.0.0.55",
  "2024-03-15 08:12:03 AUTH FAILED user=admin src=10.0.0.55",
  "2024-03-15 08:12:05 AUTH FAILED user=admin src=10.0.0.55",
  "2024-03-15 08:12:07 AUTH FAILED user=admin src=10.0.0.55",
  "2024-03-15 08:12:09 AUTH FAILED user=admin src=10.0.0.55",
  "2024-03-15 08:12:11 AUTH SUCCESS user=admin src=10.0.0.55",
  "2024-03-15 08:15:00 FIREWALL BLOCK src=10.0.0.55 dst=192.168.1.10 port=445",
  "2024-03-15 09:00:00 AUTH SUCCESS user=jdoe src=172.16.0.20",
  "2024-03-15 09:01:00 DNS QUERY domain=evil-c2.example.com src=10.0.0.55",
]

# Parse into structured events
events = []
for log in raw_logs:
    m = re.match(r'(\S+ \S+) (\w+) (\w+) (?:user=(\S+) )?(?:src=(\S+))?', log)
    if m:
        events.append({
            "time": m.group(1),
            "source": m.group(2),
            "action": m.group(3),
            "user": m.group(4) or "-",
            "ip": m.group(5) or "-",
            "raw": log,
        })

# Correlation: count failed logins per IP
failed_by_ip = defaultdict(int)
for e in events:
    if e["action"] == "FAILED":
        failed_by_ip[e["ip"]] += 1

print("SIEM Log Correlation Demo")
print("=" * 55)
print(f"Parsed {len(events)} events\n")

print("Failed login counts by IP:")
for ip, count in sorted(failed_by_ip.items(), key=lambda x: -x[1]):
    status = "ALERT: possible brute force" if count >= 5 else "normal"
    print(f"  {ip:15s} -> {count} failures [{status}]")

# Cross-reference: does the brute-force IP appear elsewhere?
suspect_ip = max(failed_by_ip, key=failed_by_ip.get)
print(f"\nCross-referencing {suspect_ip}:")
for e in events:
    if e["ip"] == suspect_ip and e["action"] != "FAILED":
        print(f"  [{e['source']}] {e['raw']}")`}
  timeout={8}
  title="Log Correlation Engine"
/>

## Key Metrics

- **MTTD** (Mean Time to Detect) — how long from compromise to alert.
- **MTTR** (Mean Time to Respond) — how long from alert to containment.
- **False positive rate** — percentage of alerts that are noise.

Industry median MTTD is still around **200 days** for breaches discovered by the victim. A mature SOC with good log coverage and tuned detections can bring that down to hours.

## Summary

The SOC is the detection and response layer. It depends on comprehensive logging, a SIEM for correlation, well-tuned detection rules, and a disciplined triage process. The goal is not to catch everything — it is to catch the important things fast.

<Quiz
  chapterSlug="120-soc-and-siem-monitoring"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the primary role of a SIEM?",
      options: ["Prevent all network intrusions", "Aggregate, correlate, and alert on security logs", "Replace the need for firewalls", "Encrypt log data at rest"],
      correctIndex: 1,
      explanation: "A SIEM collects logs from many sources, normalizes and correlates them, and generates alerts. It is a detection tool, not a prevention tool.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a high false positive rate indicate?",
      options: ["The SOC is overstaffed", "Detection rules need tuning to reduce noise", "The network has no threats", "Log retention is too short"],
      correctIndex: 1,
      explanation: "Too many false positives mean analysts waste time on noise and may miss real incidents. Rules need tuning to improve signal-to-noise ratio.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In the triage funnel, what happens between raw events and confirmed incidents?",
      options: ["Events are encrypted and archived", "Events are correlated into alerts, then triaged by analysts", "Events are sent directly to the CISO", "Events are deleted after 24 hours"],
      correctIndex: 1,
      explanation: "Raw events (millions) are correlated into alerts (hundreds), which analysts triage down to confirmed incidents (single digits). This funnel is the core SOC workflow.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "How does anomaly detection connect to neural networks?",
      options: ["Neural networks replace SIEMs entirely", "Anomaly detection models normal behavior and flags deviations, which is a classification task", "Neural networks are used to encrypt logs", "There is no connection"],
      correctIndex: 1,
      explanation: "Anomaly detection learns a baseline of normal behavior and flags outliers. This is fundamentally a classification or density-estimation problem where neural networks (autoencoders, isolation forests) apply directly.",
      randomize: true,
    }
  ]}
/>
