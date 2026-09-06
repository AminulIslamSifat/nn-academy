---
title: "Zero Trust Architecture"
description: "Never trust, always verify — replacing the perimeter model with identity-centric, continuous access control."
read_time: 11
code_time: 6
---

## The Perimeter Problem

Traditional security assumed a hard outer shell (firewall) and a soft interior. Once inside, you were trusted. This fails when:

- Employees work remotely.
- Applications live in the cloud.
- An attacker phishes a valid credential.
- A compromised laptop is already "inside."

The perimeter is gone. **Zero Trust** starts from the assumption that no network location, device, or user is trusted by default.

## The Core Principle

> Never trust, always verify.

Every access request is evaluated **per-request**, based on:

1. **Who** is requesting (identity, authentication strength).
2. **What** they want (the specific resource).
3. **Where** they are (device health, network context, geolocation).
4. **When** they are requesting (time of day, anomaly score).

## The Zero Trust Pillars

| Pillar | Question It Answers |
|---|---|
| **Identity** | Is this really who they claim to be? |
| **Device** | Is this device managed, patched, and healthy? |
| **Network** | Is the connection encrypted and from an expected location? |
| **Application** | Is the user authorized for this specific app and action? |
| **Data** | Is the data classified and access policy enforced? |
| **Visibility** | Are we logging and monitoring every decision? |

<Callout type="info" title="Zero Trust is not a product">It is an architectural philosophy. You implement it with a combination of IAM, MDM, microsegmentation, encryption, and continuous monitoring. No single vendor delivers 'zero trust in a box.'</Callout>

## Identity as the New Perimeter

In a zero trust model, **identity replaces the network boundary**. The key components:

- **Strong authentication** — MFA required, preferably phishing-resistant (FIDO2/WebAuthn).
- **Conditional access policies** — access decisions based on real-time signals.
- **Session-bound tokens** — short-lived, revocable, scoped to specific resources.
- **Continuous verification** — re-evaluate trust during the session, not just at login.

## Microsegmentation

Instead of a flat internal network, divide it into small zones with strict access controls between them.

```
[Web Tier] --HTTPS--> [App Tier] --mTLS--> [DB Tier]
     |                    |                    |
   WAF              Service Mesh         Encrypted at rest
```

If an attacker compromises the web tier, microsegmentation prevents lateral movement to the database tier.

## The BeyondCorp Model

Google's **BeyondCorp** was one of the first large-scale zero trust deployments. Key ideas:

- No internal network privilege. Every request goes through the same access control, whether from the office or a coffee shop.
- Access decisions are based on device state, user identity, and context — not IP address.
- Every access is logged and auditable.

## Interactive: Simulate a Zero Trust Access Decision

<PyRunner
  cellId="123-zero-trust-architecture-cell-1"
  defaultCode={`# Simulate a zero trust policy engine
def evaluate_access(request):
    signals = []
    score = 0

    # Identity check
    if request.get("mfa_verified"):
        signals.append("MFA verified (+30)")
        score += 30
    else:
        signals.append("MFA missing (-20)")
        score -= 20

    # Device check
    if request.get("device_managed") and request.get("device_patched"):
        signals.append("Device managed and patched (+25)")
        score += 25
    elif request.get("device_managed"):
        signals.append("Device managed but unpatched (+10)")
        score += 10
    else:
        signals.append("Unmanaged device (-15)")
        score -= 15

    # Network check
    if request.get("encrypted"):
        signals.append("Encrypted connection (+15)")
        score += 15
    else:
        signals.append("Unencrypted connection (-10)")
        score -= 10

    # Context check
    if request.get("anomaly_score", 0) < 0.3:
        signals.append("Low anomaly score (+20)")
        score += 20
    else:
        signals.append("High anomaly score (-25)")
        score -= 25

    # Resource sensitivity
    if request.get("resource_sensitivity") == "high" and score < 60:
        signals.append("High-sensitivity resource requires score >= 60")
        return "DENY", score, signals

    decision = "ALLOW" if score >= 40 else "DENY"
    return decision, score, signals

# Test scenarios
scenarios = [
  {"name": "Corporate laptop, MFA, encrypted", "mfa_verified": True, "device_managed": True,
   "device_patched": True, "encrypted": True, "anomaly_score": 0.1, "resource_sensitivity": "medium"},
  {"name": "Personal phone, no MFA", "mfa_verified": False, "device_managed": False,
   "device_patched": False, "encrypted": True, "anomaly_score": 0.6, "resource_sensitivity": "high"},
  {"name": "Managed device, MFA, but high anomaly", "mfa_verified": True, "device_managed": True,
   "device_patched": True, "encrypted": True, "anomaly_score": 0.8, "resource_sensitivity": "medium"},
]

print("Zero Trust Policy Engine Simulation")
print("=" * 55)
for s in scenarios:
    name = s.pop("name")
    decision, score, signals = evaluate_access(s)
    print(f"\nScenario: {name}")
    print(f"  Decision: {decision} (score: {score})")
    for sig in signals:
        print(f"    {sig}")`}
  timeout={8}
  title="Zero Trust Policy Engine"
/>

## Migration Reality

Zero trust is a journey, not a switch. A practical migration path:

1. **Inventory** — know all users, devices, and applications.
2. **MFA everywhere** — the single highest-impact first step.
3. **Device management** — MDM/EDR on all endpoints.
4. **Microsegment critical assets** — start with the crown jewels.
5. **Continuous monitoring** — feed every access decision into the SIEM.

<Callout type="tip" title="Start with MFA">If you do nothing else, enforce phishing-resistant MFA. It eliminates the majority of credential-based attacks and is the foundation of zero trust identity.</Callout>

## Summary

Zero trust replaces the outdated perimeter model with continuous, context-aware verification. Identity is the new boundary, microsegmentation limits blast radius, and every decision is logged. It is not a product you buy but an architecture you build incrementally.

<Quiz
  chapterSlug="123-zero-trust-architecture"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the foundational assumption of zero trust?",
      options: ["Internal users are trusted by default", "No user, device, or network location is trusted by default", "Only external users need verification", "Firewalls are sufficient for internal security"],
      correctIndex: 1,
      explanation: "Zero trust assumes that threats exist both outside and inside the network. Every access request must be verified regardless of origin.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "In zero trust, what replaces the network perimeter as the security boundary?",
      options: ["A stronger firewall", "Identity and device context", "VLAN tagging", "IP whitelisting"],
      correctIndex: 1,
      explanation: "Zero trust shifts the boundary from network location to identity. Access decisions are based on who you are, your device state, and context — not which subnet you are on.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does microsegmentation primarily prevent?",
      options: ["DDoS attacks", "Lateral movement after initial compromise", "Phishing emails", "SQL injection"],
      correctIndex: 1,
      explanation: "Microsegmentation divides the network into isolated zones with strict access controls. If an attacker compromises one zone, they cannot freely move to others.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the recommended first step in a zero trust migration?",
      options: ["Replace all firewalls", "Enforce multi-factor authentication everywhere", "Migrate everything to the cloud", "Deploy a SIEM"],
      correctIndex: 1,
      explanation: "MFA is the highest-impact, lowest-friction starting point. It eliminates most credential-based attacks and establishes the identity foundation that zero trust depends on.",
      randomize: true,
    }
  ]}
/>
