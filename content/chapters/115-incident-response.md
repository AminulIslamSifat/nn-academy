---
title: "Incident Response"
description: "The structured playbook for detecting, containing, eradicating, and recovering from a security breach."
read_time: 12
code_time: 7
---

## Why Have a Plan?

During a breach, panic is the enemy. A rehearsed incident-response process means the team knows exactly who does what, when. Organizations with tested IR plans contain breaches significantly faster and at lower cost.

## The IR Lifecycle (NIST SP 800-61)

### 1. Preparation

Before anything happens:

- Build the IR team and define roles.
- Deploy detection tooling (SIEM, EDR, network monitoring).
- Write playbooks for common scenarios (ransomware, phishing, data exfiltration).
- Conduct tabletop exercises.
- Establish communication channels and escalation paths.

### 2. Detection & Analysis

Identify the incident and scope it:

- **Triage the alert** — is it a true positive?
- **Determine scope** — which systems, users, and data are affected?
- **Identify the attack vector** — how did the attacker get in?
- **Preserve evidence** — capture logs, memory dumps, disk images.

### 3. Containment

Stop the spread:

- **Short-term** — isolate affected hosts from the network immediately.
- **Long-term** — apply fixes so the attacker cannot return (patch the vulnerability, revoke compromised credentials).

<Callout type="warning" title="Preserve evidence">If the incident may become legal, preserve forensic integrity: image disks, keep logs immutable, and document every action with timestamps.</Callout>

### 4. Eradication

Remove the root cause and attacker foothold:

- Delete malware and attacker tools.
- Remove persistence mechanisms (scheduled tasks, registry keys, backdoor accounts).
- Patch the exploited vulnerability.
- Reset all potentially compromised credentials.

### 5. Recovery

Restore services from clean backups:

- Rebuild affected systems from known-good images.
- Restore data from verified backups.
- Monitor closely for signs of re-compromise.
- Gradually restore services with enhanced monitoring.

### 6. Lessons Learned

The postmortem that improves defenses:

- What happened and when?
- How well did the team perform?
- What could have been detected earlier?
- What process or tool gaps were exposed?
- What changes will prevent recurrence?

## Detection Is a Data Problem

Centralized logging, baseline behavior, and alerting turn an invisible breach into a detectable one.

| Data Source | What It Reveals |
|---|---|
| Authentication logs | Unusual logins, brute force, privilege escalation |
| Endpoint (EDR) | Process creation, file modifications, registry changes |
| Network flows | C2 communication, lateral movement, exfiltration |
| DNS logs | Lookups for known-bad domains |
| Email gateway | Phishing delivery, malicious attachments |

Many modern SOC teams apply anomaly detection models here — the same classification and clustering techniques from the Neural Networks module.

## Interactive: IR Timeline and Metrics

<PyRunner
  cellId="115-incident-response-cell-1"
  defaultCode={`from datetime import datetime

# Simulated incident timeline
incident_events = [
    {"time": "2024-03-15 02:14:00", "phase": "Initial Compromise", "event": "SSH brute force succeeds on server-01"},
    {"time": "2024-03-15 02:15:00", "phase": "Initial Compromise", "event": "Payload downloaded and executed"},
    {"time": "2024-03-15 02:17:00", "phase": "Execution", "event": "C2 beacon established to 185.220.101.45"},
    {"time": "2024-03-15 02:20:00", "phase": "Persistence", "event": "Backdoor account svc_update created"},
    {"time": "2024-03-15 02:25:00", "phase": "Lateral Movement", "event": "Lateral movement to db-server-01 via SSH"},
    {"time": "2024-03-15 03:00:00", "phase": "Actions on Objectives", "event": "Database dump initiated"},
    {"time": "2024-03-15 08:30:00", "phase": "Detection", "event": "SOC analyst notices anomalous outbound traffic"},
    {"time": "2024-03-15 08:45:00", "phase": "Detection", "event": "Incident declared, IR team activated"},
    {"time": "2024-03-15 09:15:00", "phase": "Containment", "event": "server-01 and db-server-01 isolated from network"},
    {"time": "2024-03-15 10:00:00", "phase": "Containment", "event": "Compromised credentials revoked"},
    {"time": "2024-03-15 14:00:00", "phase": "Eradication", "event": "Malware removed, vulnerability patched"},
    {"time": "2024-03-16 09:00:00", "phase": "Recovery", "event": "Systems rebuilt and restored from backup"},
    {"time": "2024-03-18 10:00:00", "phase": "Lessons Learned", "event": "Postmortem conducted"},
]

print("Incident Response Timeline")
print("=" * 65)

current_phase = None
for e in incident_events:
    if e["phase"] != current_phase:
        current_phase = e["phase"]
        print(f"\n  [{current_phase}]")
    print(f"    {e['time']} - {e['event']}")

# Compute key metrics
fmt = "%Y-%m-%d %H:%M:%S"
compromise_time = datetime.strptime(incident_events[0]["time"], fmt)
detection_time = datetime.strptime("2024-03-15 08:30:00", fmt)
containment_time = datetime.strptime("2024-03-15 09:15:00", fmt)
recovery_time = datetime.strptime("2024-03-16 09:00:00", fmt)

mttd = (detection_time - compromise_time).total_seconds() / 3600
mttc = (containment_time - detection_time).total_seconds() / 3600
mttr = (recovery_time - compromise_time).total_seconds() / 3600

print(f"\n--- Key Metrics ---")
print(f"  MTTD (Mean Time to Detect):    {mttd:.1f} hours")
print(f"  MTTC (Mean Time to Contain):   {mttc:.1f} hours")
print(f"  MTTR (Mean Time to Recover):   {mttr:.1f} hours")
print(f"  Total incident duration:       {mttr:.1f} hours")

print(f"\n--- Attack Duration Before Detection ---")
print(f"  Attacker was active for {mttd:.1f} hours before detection")
print(f"  Industry median MTTD: ~200 days (external discovery)")
print(f"  This incident: {mttd:.1f} hours (internal detection - good!)")`}
  timeout={8}
  title="IR Timeline & Metrics"
/>

## Interactive: IR Phase Classifier

<PyRunner
  cellId="115-incident-response-cell-2"
  defaultCode={`# Classify incident response actions into NIST phases

actions = [
    "Deploy EDR agents to all endpoints",
    "Write a ransomware response playbook",
    "Conduct a tabletop exercise",
    "SOC analyst receives alert for unusual outbound traffic",
    "Verify the alert is a true positive",
    "Isolate affected hosts from the network",
    "Block the attacker IP at the firewall",
    "Delete malware files and remove persistence",
    "Reset all compromised credentials",
    "Patch the exploited vulnerability",
    "Restore systems from verified backups",
    "Monitor restored systems for re-compromise",
    "Conduct postmortem and document lessons",
]

phase_map = {
    "Deploy EDR agents to all endpoints": "Preparation",
    "Write a ransomware response playbook": "Preparation",
    "Conduct a tabletop exercise": "Preparation",
    "SOC analyst receives alert for unusual outbound traffic": "Detection & Analysis",
    "Verify the alert is a true positive": "Detection & Analysis",
    "Isolate affected hosts from the network": "Containment",
    "Block the attacker IP at the firewall": "Containment",
    "Delete malware files and remove persistence": "Eradication",
    "Reset all compromised credentials": "Eradication",
    "Patch the exploited vulnerability": "Eradication",
    "Restore systems from verified backups": "Recovery",
    "Monitor restored systems for re-compromise": "Recovery",
    "Conduct postmortem and document lessons": "Lessons Learned",
}

print("NIST IR Lifecycle - Action Classification")
print("=" * 55)

phases = ["Preparation", "Detection & Analysis", "Containment", "Eradication", "Recovery", "Lessons Learned"]
for phase in phases:
    phase_actions = [a for a in actions if phase_map.get(a) == phase]
    print(f"\n  {phase}:")
    for a in phase_actions:
        print(f"    - {a}")

print(f"\n--- Phase Summary ---")
for phase in phases:
    count = len([a for a in actions if phase_map.get(a) == phase])
    print(f"  {phase:25s}: {count} actions")`}
  timeout={8}
  title="IR Phase Classifier"
/>

## Communication During an Incident

- **Internal** — IR team, management, legal, PR.
- **External** — law enforcement (if required), regulators (GDPR 72-hour rule), affected customers.
- **Documentation** — every decision, action, and timestamp in the incident log.

<Callout type="tip" title="Pre-draft your communications">During a breach is the worst time to write a customer notification. Have templates ready for common scenarios so you can adapt rather than create from scratch.</Callout>

## Summary

Incident response is preparation meeting chaos. A tested lifecycle limits damage, preserves evidence, and turns every incident into an improvement. The six phases — preparation, detection, containment, eradication, recovery, and lessons learned — are the framework that keeps the team effective under pressure.

<Quiz
  chapterSlug="115-incident-response"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the first phase of the NIST incident response lifecycle?",
      options: ["Detection & Analysis", "Preparation", "Containment", "Recovery"],
      correctIndex: 1,
      explanation: "Preparation comes first: building the team, deploying tools, writing playbooks, and conducting exercises. Without preparation, the response to a real incident will be chaotic and slow.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does MTTD measure?",
      options: ["Mean Time to Deploy", "Mean Time to Detect — how long from compromise to detection", "Mean Time to Defend", "Mean Time to Document"],
      correctIndex: 1,
      explanation: "MTTD (Mean Time to Detect) measures the duration between when a compromise occurs and when it is detected. Lower MTTD means less time for the attacker to operate and cause damage.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why is evidence preservation important during containment?",
      options: ["To speed up recovery", "Because findings may be needed for legal proceedings and forensic analysis", "To reduce the cost of the incident", "To satisfy the SIEM vendor"],
      correctIndex: 1,
      explanation: "If the incident leads to legal action, evidence must be forensically sound. Modifying or destroying evidence during containment can make it inadmissible in court.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the purpose of the lessons learned phase?",
      options: ["To assign blame for the incident", "To identify process and tool gaps and improve future response", "To close the incident ticket", "To calculate the financial cost"],
      correctIndex: 1,
      explanation: "Lessons learned (postmortem) reviews how the incident was handled, what could have been detected earlier, and what changes will prevent recurrence. It is about improvement, not blame.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "Under GDPR, how quickly must a data breach be reported to the supervisory authority?",
      options: ["Within 24 hours", "Within 72 hours", "Within 7 days", "Within 30 days"],
      correctIndex: 1,
      explanation: "GDPR Article 33 requires notification to the supervisory authority within 72 hours of becoming aware of a personal data breach, unless the breach is unlikely to result in risk to individuals.",
      randomize: true,
    }
  ]}
/>
