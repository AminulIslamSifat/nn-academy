---
title: "Compliance Frameworks"
description: "GDPR, HIPAA, PCI-DSS, ISO 27001, SOC 2 — what each governs and how controls map to requirements."
read_time: 11
code_time: 5
---

## Why Compliance Matters

Compliance is not security, but it creates accountability. Frameworks force organizations to document controls, undergo audits, and face consequences for gaps. For security professionals, compliance is often the **budget justification** — "we need this tool because PCI-DSS requires it."

<Callout type="info" title="Compliance is the floor, not the ceiling">Meeting a framework's minimum requirements does not make you secure. It makes you auditable. Real security goes beyond the checklist.</Callout>

## The Major Frameworks

### GDPR (General Data Protection Regulation)

- **Scope:** Personal data of EU residents, regardless of where the company is located.
- **Key requirements:** Consent, right to access, right to erasure, breach notification within 72 hours, Data Protection Impact Assessments.
- **Enforcement:** Fines up to 4% of global annual revenue.

### HIPAA (Health Insurance Portability and Accountability Act)

- **Scope:** Protected Health Information (PHI) in the US healthcare system.
- **Key requirements:** Administrative, physical, and technical safeguards. Access controls, audit logs, encryption, Business Associate Agreements.
- **Enforcement:** Fines per violation, criminal penalties for willful neglect.

### PCI-DSS (Payment Card Industry Data Security Standard)

- **Scope:** Any organization that stores, processes, or transmits credit card data.
- **Key requirements:** 12 requirements including network segmentation, encryption of cardholder data, access control, regular vulnerability scanning, and penetration testing.
- **Enforcement:** Fines from card brands, loss of payment processing ability.

### ISO 27001

- **Scope:** Information Security Management System (ISMS) — a comprehensive management framework.
- **Key requirements:** Risk assessment, Statement of Applicability, 93 controls in Annex A, continuous improvement cycle.
- **Enforcement:** Certification audits by accredited bodies.

### SOC 2 (System and Organization Controls)

- **Scope:** Service organizations (SaaS, cloud providers, hosting).
- **Key requirements:** Trust Service Criteria — Security, Availability, Processing Integrity, Confidentiality, Privacy.
- **Enforcement:** Independent audit; the report is shared with customers.

## How Controls Map Across Frameworks

Most frameworks share common control themes:

| Control Theme | GDPR | HIPAA | PCI-DSS | ISO 27001 | SOC 2 |
|---|---|---|---|---|---|
| Access control | Art. 32 | §164.312(a) | Req. 7-8 | A.9 | CC6 |
| Encryption | Art. 32 | §164.312(e) | Req. 3-4 | A.10 | CC6.7 |
| Incident response | Art. 33 | §164.308(a)(6) | Req. 12.10 | A.16 | CC7.3 |
| Logging & monitoring | Art. 32 | §164.312(b) | Req. 10 | A.12.4 | CC7.2 |
| Risk assessment | Art. 35 | §164.308(a)(1) | Req. 12.2 | Clause 6 | CC3 |

<Callout type="tip" title="Map once, comply many">If you implement controls to the strictest framework (usually PCI-DSS or ISO 27001), you can map them to other frameworks. This avoids duplicating work.</Callout>

## Interactive: Compliance Gap Checker

<PyRunner
  cellId="128-compliance-frameworks-cell-1"
  defaultCode={`# Simulate a compliance gap assessment
required_controls = {
    "PCI-DSS": [
        "Firewall configuration",
        "Encrypt cardholder data at rest",
        "Encrypt cardholder data in transit",
        "Access control (need-to-know)",
        "Quarterly vulnerability scans",
        "Annual penetration test",
        "Audit log retention (1 year)",
    ],
    "SOC 2": [
        "Access control",
        "Encryption at rest",
        "Incident response plan",
        "Change management",
        "Vendor risk assessment",
    ],
}

# Current state: what the organization actually has
implemented = {
    "Firewall configuration",
    "Encrypt cardholder data at rest",
    "Encrypt cardholder data in transit",
    "Access control (need-to-know)",
    "Incident response plan",
    "Change management",
}

print("Compliance Gap Assessment")
print("=" * 55)

for framework, controls in required_controls.items():
    gaps = [c for c in controls if c not in implemented]
    met = [c for c in controls if c in implemented]
    pct = len(met) / len(controls) * 100

    print(f"\n{framework}: {len(met)}/{len(controls)} controls met ({pct:.0f}%)")
    if gaps:
        print(f"  GAPS ({len(gaps)}):")
        for g in gaps:
            print(f"    [ ] {g}")
    else:
        print("  All controls implemented!")

print("\n--- Cross-Framework Overlap ---")
all_controls = set()
for controls in required_controls.values():
    all_controls.update(controls)
print(f"Total unique controls across frameworks: {len(all_controls)}")
print(f"Currently implemented: {len(implemented & all_controls)}")
print(f"Remaining gaps: {len(all_controls - implemented)}")`}
  timeout={8}
  title="Compliance Gap Checker"
/>

## The Audit Process

1. **Scope definition** — which systems, data, and processes are in scope.
2. **Evidence collection** — policies, configs, screenshots, logs.
3. **Assessment** — auditor tests controls against requirements.
4. **Findings** — gaps documented with severity.
5. **Remediation** — fix gaps within agreed timelines.
6. **Report** — formal opinion (ISO certificate, SOC 2 report, attestation).

<Callout type="warning" title="Scope creep is real">If you store card data in a new system, PCI-DSS scope expands to include it. Track where regulated data flows and keep the scope as small as possible.</Callout>

## Summary

Compliance frameworks provide structure and accountability. Understand which ones apply to your organization, map controls across frameworks to avoid duplication, and treat compliance as a starting point — not the end goal — of your security program.

<Quiz
  chapterSlug="128-compliance-frameworks"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the maximum GDPR fine?",
      options: ["$1 million", "4% of global annual revenue", "10% of quarterly profit", "$500,000 per violation"],
      correctIndex: 1,
      explanation: "GDPR fines can reach up to 4% of global annual revenue or 20 million euros, whichever is higher. This makes GDPR one of the most financially significant compliance frameworks.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Which framework specifically governs credit card data?",
      options: ["HIPAA", "GDPR", "PCI-DSS", "SOC 2"],
      correctIndex: 2,
      explanation: "PCI-DSS (Payment Card Industry Data Security Standard) applies to any organization that stores, processes, or transmits credit card data. It has 12 specific requirements.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does 'compliance is the floor, not the ceiling' mean?",
      options: ["Compliance is the highest level of security achievable", "Meeting minimum requirements does not guarantee security; it is just the starting point", "Compliance is only for large organizations", "Compliance replaces the need for security teams"],
      correctIndex: 1,
      explanation: "Compliance frameworks set minimum requirements. Meeting them makes you auditable but not necessarily secure. Real security requires going beyond the checklist.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is the benefit of mapping controls across multiple compliance frameworks?",
      options: ["It reduces the number of controls needed", "It avoids duplicating work by implementing to the strictest standard and mapping to others", "It eliminates the need for audits", "It reduces the cost of security tools"],
      correctIndex: 1,
      explanation: "Most frameworks share common control themes. By implementing to the strictest framework and mapping controls to others, you comply with multiple frameworks without redundant work.",
      randomize: true,
    }
  ]}
/>
