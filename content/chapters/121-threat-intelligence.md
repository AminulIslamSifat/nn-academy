---
title: "Threat Intelligence"
description: "Turning raw data about adversaries into actionable knowledge that drives defense decisions."
read_time: 11
code_time: 6
---

## What Threat Intel Is

Threat intelligence is **processed, analyzed, and contextualized** information about threats. Raw data (a list of bad IPs) is not intelligence. Intelligence answers questions like: "Who is targeting our sector, how, and what should we do about it?"

## The Intelligence Hierarchy

| Level | Audience | Example |
|---|---|---|
| **Strategic** | Executives, board | "Nation-state actors are targeting financial institutions in our region" |
| **Operational** | Security managers | "APT29 is using spear-phishing with malicious ISO attachments this quarter" |
| **Tactical** | SOC analysts | "Block these 14 IOCs associated with the current campaign" |
| **Technical** | Detection engineers | "This YARA rule detects the Cobalt Strike beacon variant in use" |

## IOCs vs TTPs

**Indicators of Compromise (IOCs)** are atomic artifacts: IP addresses, domain names, file hashes, email addresses. They are easy to share and block, but easy for attackers to change.

**Tactics, Techniques, and Procedures (TTPs)** describe *how* the attacker operates: their initial access method, lateral movement strategy, exfiltration channel. TTPs are harder to change and far more valuable for defense.

<Callout type="info" title="The Pyramid of Pain">Changing a hash is trivial for an attacker. Changing their entire TTP is expensive. Defending against TTPs forces the attacker to fundamentally retrain, not just swap an IP address.</Callout>

## MITRE ATT&CK

The **ATT&CK** framework is a matrix of adversary techniques organized by tactic. It is the common language of threat intel.

```
Tactic (Goal)          -> Technique (How)
─────────────────────────────────────────
Initial Access         -> T1566 Phishing
Execution              -> T1204 User Execution
Persistence            -> T1053 Scheduled Task
Privilege Escalation   -> T1548 Abuse Elevation Control
Lateral Movement       -> T1021 Remote Services
Exfiltration           -> T1041 Exfil Over C2 Channel
```

When you map a detection rule to an ATT&CK technique, you can measure your coverage: "We detect 3 of the 14 techniques this group is known to use."

## The Intelligence Lifecycle

1. **Planning & Direction** — what questions do we need answered?
2. **Collection** — gather raw data (OSINT, feeds, internal logs).
3. **Processing** — normalize, deduplicate, enrich.
4. **Analysis** — connect dots, assess confidence, attribute.
5. **Dissemination** — deliver to the right audience in the right format.
6. **Feedback** — was it useful? Adjust.

## Consuming Threat Feeds

Most organizations consume feeds from:

- **Commercial providers** (Recorded Future, Mandiant)
- **ISACs** (Information Sharing and Analysis Centers) per industry
- **Government agencies** (CISA alerts)
- **Open source** (Abuse.ch, AlienVault OTX)

<Callout type="warning" title="Feeds are noisy">A raw feed of 50,000 IOCs will flood your SIEM. Filter by relevance to your sector, age (IOCs decay fast), and confidence score before ingesting.</Callout>

## Interactive: IOC Enrichment and Scoring

<PyRunner
  cellId="121-threat-intelligence-cell-1"
  defaultCode={`import hashlib
from collections import defaultdict

# Simulated raw threat feed IOCs
raw_iocs = [
  {"type": "ip", "value": "185.220.101.45", "source": "Abuse.ch", "age_days": 2},
  {"type": "domain", "value": "evil-c2.example.com", "source": "AlienVault", "age_days": 5},
  {"type": "hash", "value": "a3f2b8c91d4e5f6a7b8c9d0e1f2a3b4c", "source": "Mandiant", "age_days": 30},
  {"type": "ip", "value": "91.219.236.174", "source": "Abuse.ch", "age_days": 1},
  {"type": "domain", "value": "old-campaign.example.net", "source": "CISA", "age_days": 90},
  {"type": "ip", "value": "185.220.101.45", "source": "CISA", "age_days": 2},
]

# Confidence scoring based on source reputation and age
def score_ioc(ioc):
    source_conf = {"Mandiant": 0.9, "CISA": 0.85, "Abuse.ch": 0.7, "AlienVault": 0.65}
    conf = source_conf.get(ioc["source"], 0.5)
    # IOCs decay: halve confidence after 30 days
    age_penalty = 0.5 if ioc["age_days"] > 30 else 1.0
    return round(conf * age_penalty, 2)

# Deduplicate by value
deduped = {}
for ioc in raw_iocs:
    key = ioc["value"]
    if key not in deduped:
        deduped[key] = ioc
        deduped[key]["sources"] = [ioc["source"]]
    else:
        deduped[key]["sources"].append(ioc["source"])

print("Threat Intel IOC Processing")
print("=" * 55)
print(f"Raw IOCs: {len(raw_iocs)} -> After dedup: {len(deduped)}\n")

for val, ioc in sorted(deduped.items(), key=lambda x: -score_ioc(x[1])):
    score = score_ioc(ioc)
    sources = ", ".join(ioc["sources"])
    status = "HIGH" if score >= 0.7 else "MED" if score >= 0.4 else "LOW"
    print(f"  [{status:4s}] {ioc['type']:7s} {val:35s} score={score} src={sources}")

# MITRE mapping example
print("\nMITRE ATT&CK mapping for this campaign:")
techniques = ["T1566.001 Spearphishing Attachment", "T1059.001 PowerShell", "T1071.001 Web Protocols"]
for t in techniques:
    print(f"  -> {t}")`}
  timeout={8}
  title="IOC Enrichment Pipeline"
/>

## Attribution and Its Limits

Attribution (naming the actor) is the hardest part. Confidence levels matter:

- **Confirmed** — multiple independent sources, forensic evidence.
- **Probable** — strong TTP overlap, infrastructure reuse.
- **Possible** — circumstantial, single-source.

<Callout type="tip" title="Attribution is rarely the point">For most defenders, knowing the TTPs and IOCs is enough to improve detection. Attribution matters more for law enforcement and strategic decisions than for the SOC.</Callout>

## Summary

Threat intelligence transforms raw adversary data into prioritized, actionable defense. Focus on TTPs over IOCs, map to ATT&CK for coverage measurement, and filter feeds by relevance before they flood your SIEM.

<Quiz
  chapterSlug="121-threat-intelligence"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why are TTPs more valuable than IOCs for defense?",
      options: ["TTPs are easier to collect", "IOCs are free while TTPs cost money", "TTPs are harder for attackers to change and describe fundamental behavior", "IOCs cannot be blocked by firewalls"],
      correctIndex: 2,
      explanation: "IOCs like IP addresses and hashes are trivial for an attacker to swap. TTPs describe their methodology, which requires significant effort to change, making TTP-based defenses more durable.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does the MITRE ATT&CK framework organize?",
      options: ["CVE severity scores", "Adversary techniques grouped by tactical objective", "Firewall rule templates", "Encryption algorithm comparisons"],
      correctIndex: 1,
      explanation: "ATT&CK is a matrix of adversary techniques organized by tactic (the goal, like Initial Access or Exfiltration), providing a common language for describing and measuring threat behavior.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the Pyramid of Pain?",
      options: ["A network topology diagram", "A model showing that defending against higher-level indicators (TTPs) is more costly for attackers to evade", "A risk assessment matrix", "A SIEM alert severity scale"],
      correctIndex: 1,
      explanation: "The Pyramid of Pain illustrates that while atomic IOCs (hashes, IPs) are easy for attackers to change, TTPs are hard. Defending at higher levels causes more pain to the adversary.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why should you filter threat feeds before ingesting them into a SIEM?",
      options: ["To reduce storage costs", "Raw feeds are noisy and can flood the SIEM with irrelevant or stale IOCs", "SIEMs cannot process more than 100 IOCs", "Filtering is required by compliance"],
      correctIndex: 1,
      explanation: "Raw feeds contain thousands of IOCs, many stale or irrelevant to your sector. Ingesting them unfiltered creates alert fatigue and buries real signals in noise.",
      randomize: true,
    }
  ]}
/>
