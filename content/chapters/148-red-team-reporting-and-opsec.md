---
title: "Red Team Reporting & OpSec"
description: "Writing professional red team reports and maintaining operational security throughout engagements."
read_time: 11
code_time: 5
---

## The Report Is the Product

A red team engagement without a clear report is wasted effort. The report drives remediation and demonstrates ROI.

## Report Structure

### Executive Summary
Business impact in plain language. No jargon. Answer: What was found? What is the risk? What should leadership do?

### Technical Findings
For each finding:
1. Title and severity
2. Description of the vulnerability or attack path
3. Steps to reproduce (with screenshots/evidence)
4. Business impact
5. Remediation recommendation (specific, actionable)
6. MITRE ATT&CK mapping

### Attack Narrative
Chronological story of the engagement: initial access, lateral movement, objectives achieved. Helps the client understand the real-world risk.

### Recommendations
Prioritized list of improvements, mapped to frameworks (NIST, CIS).

## Operational Security

- Use dedicated infrastructure per engagement
- Never reuse domains or IPs across clients
- Use encrypted communications
- Sanitize tools (remove personal metadata)
- Clean up all artifacts post-engagement
- Follow rules of engagement strictly

<Callout type="tip" title="Evidence quality matters">Screenshots with timestamps, log excerpts, and proof-of-concept outputs make findings credible. A finding without evidence is just an opinion.</Callout>

## Summary

Professional reporting and strict OpSec distinguish red teamers from hobbyists. Write clearly, provide evidence, prioritize remediation, and protect your operational footprint.

<Quiz
  chapterSlug="148-red-team-reporting-and-opsec"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Who is the executive summary written for?",
      options: ["Developers", "Technical staff", "Leadership and non-technical stakeholders", "Other red teamers"],
      correctIndex: 2,
      explanation: "The executive summary communicates business impact and risk to leadership who make budget and strategy decisions. It avoids technical jargon and focuses on what matters to the business.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should red teams use dedicated infrastructure per engagement?",
      options: ["It is cheaper", "To prevent cross-contamination between clients and protect operational security", "It is required by law", "It improves performance"],
      correctIndex: 1,
      explanation: "Reusing infrastructure across clients risks cross-contamination (one client discovering another) and allows defenders to correlate engagements. Dedicated infrastructure maintains OpSec.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What should every technical finding include?",
      options: ["Only the vulnerability name", "Description, reproduction steps, evidence, impact, and remediation", "Only the CVSS score", "The source code of the exploit"],
      correctIndex: 1,
      explanation: "A complete finding includes enough detail for the client to understand, reproduce, and fix the issue. Evidence makes it credible, and specific remediation makes it actionable.",
      randomize: true,
    }
  ]}
/>
