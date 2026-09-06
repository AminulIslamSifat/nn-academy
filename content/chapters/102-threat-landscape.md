---
title: "The Threat Landscape"
description: "Who attacks systems, what they want, and the common categories of modern attacks."
read_time: 14
code_time: 8
---

## Understanding Your Adversaries

Effective defense starts with understanding who is attacking and why. Different adversaries have different capabilities, motivations, and targets. A defense that stops a script kiddie may be trivial for a nation-state to bypass.

### Adversary Categories

- **Script kiddies** — use existing tools with little understanding; motivated by curiosity or notoriety.
- **Cybercriminals** — financially motivated; run attacks as organized businesses with ROI calculations.
- **Hacktivists** — ideologically motivated disruption; target organizations they oppose politically or socially.
- **Nation-states (APTs)** — espionage, sabotage, and strategic advantage; patient, well-funded, and highly capable.
- **Insiders** — employees or contractors with legitimate access; motivated by grievance, coercion, or greed.

<Callout type="warning" title="Insiders are the hardest to detect">Insider threats bypass perimeter defenses because they already have authorized access. Behavioral analytics and least-privilege access are your primary controls.</Callout>

## Common Attack Categories

Modern attacks fall into several broad categories. Most real-world incidents combine multiple techniques:

1. **Malware** — malicious software including viruses, worms, trojans, ransomware, and spyware.
2. **Phishing & Social Engineering** — manipulating humans to steal credentials, deliver payloads, or authorize fraudulent transactions.
3. **Denial of Service (DoS/DDoS)** — overwhelming availability through volume, protocol abuse, or application-layer exhaustion.
4. **Man-in-the-Middle (MitM)** — intercepting or altering communication between two parties.
5. **Supply Chain Attacks** — compromising a trusted dependency, vendor, or update mechanism to reach downstream targets.
6. **Web Application Attacks** — SQL injection, XSS, CSRF, and other exploits targeting application logic.

<Callout type="info" title="Ransomware is a business model">Ransomware-as-a-service (RaaS) franchises let unskilled actors rent attack tooling, splitting profits with developers. This lowered the barrier to entry and exploded incident volumes after 2019.</Callout>

## The Cyber Kill Chain

Most targeted attacks follow a predictable pattern called the kill chain. Understanding each phase lets you place defenses where they interrupt the attack lifecycle:

1. **Reconnaissance** — attacker gathers information about the target (OSINT, scanning, social media).
2. **Weaponization** — attacker creates a payload tailored to the target's vulnerabilities.
3. **Delivery** — payload reaches the target via email, website, USB, or supply chain.
4. **Exploitation** — payload executes and exploits a vulnerability.
5. **Installation** — malware establishes persistence on the compromised system.
6. **Command & Control (C2)** — attacker establishes communication channel for remote control.
7. **Actions on Objectives** — attacker achieves their goal (data theft, encryption, sabotage).

Defenses can interrupt any stage. Email filtering blocks delivery. Patching prevents exploitation. Network segmentation limits C2. Backups neutralize ransomware actions.

## Worked Example: Mapping Incidents to Kill Chain Phases

Security analysts must quickly identify which kill chain phase an observed indicator represents. The exercise below parses sample threat intelligence and classifies each observation.

<PyRunner
  cellId="threat-landscape-cell-1"
  defaultCode={`kill_chain_phases = [
    'Reconnaissance',
    'Weaponization', 
    'Delivery',
    'Exploitation',
    'Installation',
    'Command and Control',
    'Actions on Objectives'
]

indicators = [
    ('Shodan scan of public IP range', 'Reconnaissance'),
    ('Crafted PDF with embedded exploit', 'Weaponization'),
    ('Spear-phishing email with attachment', 'Delivery'),
    ('CVE-2024-1234 triggered on endpoint', 'Exploitation'),
    ('Registry key added for persistence', 'Installation'),
    ('Beacon traffic to 198.51.100.23:443', 'Command and Control'),
    ('Database exfiltration over DNS tunnel', 'Actions on Objectives'),
]

print('=== Kill Chain Phase Mapping ===')
print()
for indicator, phase in indicators:
    phase_num = kill_chain_phases.index(phase) + 1
    print(f'Phase {phase_num}: {phase}')
    print(f'  Indicator: {indicator}')
    print()

print(f'Total phases covered: {len(set(p for _, p in indicators))}/{len(kill_chain_phases)}')`}
  timeout={8}
  title="Map Threat Indicators to Kill Chain Phases"
/>

## Threat Intelligence and MITRE ATT&CK

The MITRE ATT&CK framework catalogs adversary tactics and techniques based on real-world observations. Unlike the kill chain's linear model, ATT&CK provides a detailed matrix of hundreds of specific techniques organized by tactic.

Use ATT&CK to:
- Map your defenses against known techniques
- Identify coverage gaps
- Communicate precisely about adversary behavior
- Prioritize detection engineering efforts

<Callout type="tip" title="Start with your industry">MITRE publishes sector-specific threat profiles. Search for your industry vertical to see which techniques adversaries actually use against similar organizations.</Callout>

## Emerging Threat Trends

The threat landscape evolves constantly. Current trends include:

- **AI-powered attacks** — automated vulnerability discovery, deepfake social engineering, adaptive malware.
- **IoT/OT convergence** — operational technology connected to IT networks expands attack surface.
- **Cloud-native attacks** — misconfigured containers, serverless abuse, identity-based cloud compromises.
- **Living-off-the-land** — attackers using legitimate system tools (PowerShell, WMI, certutil) to avoid detection.

## Summary

Understanding attacker motivation, capabilities, and the attack lifecycle lets you place defenses where they matter most. Threat intelligence frameworks like MITRE ATT&CK turn abstract knowledge into actionable defense priorities. Stay current: the landscape shifts monthly.

<Quiz
  chapterSlug="102-threat-landscape"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Which kill chain phase involves establishing persistent access on a compromised system?",
      options: ["Delivery", "Exploitation", "Installation", "Command and Control"],
      correctIndex: 2,
      explanation: "Installation is the phase where malware establishes persistence through registry keys, scheduled tasks, or other mechanisms.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What makes insider threats particularly difficult to defend against?",
      options: ["They use advanced zero-day exploits", "They already have authorized access and bypass perimeter controls", "They always work with nation-states", "They only attack during business hours"],
      correctIndex: 1,
      explanation: "Insiders have legitimate credentials and access, making their malicious activity blend in with normal behavior.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the primary purpose of the MITRE ATT&CK framework?",
      options: ["To replace antivirus software", "To catalog adversary tactics and techniques based on real-world observations", "To automate incident response", "To calculate risk scores"],
      correctIndex: 1,
      explanation: "MITRE ATT&CK provides a comprehensive knowledge base of adversary tactics and techniques derived from actual incident data.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which attack category involves compromising a trusted vendor to reach downstream targets?",
      options: ["Phishing", "Denial of Service", "Supply chain attack", "Man-in-the-Middle"],
      correctIndex: 2,
      explanation: "Supply chain attacks compromise a trusted dependency or vendor to indirectly reach the ultimate target organization.",
      randomize: true,
    }
  ]}
/>
