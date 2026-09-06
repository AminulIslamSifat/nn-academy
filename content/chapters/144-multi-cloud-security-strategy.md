---
title: "Multi-Cloud Security Strategy"
description: "Managing security consistently across AWS, Azure, and GCP with unified policies and tooling."
read_time: 11
code_time: 5
---

## Why Multi-Cloud?

Organizations use multiple clouds for redundancy, avoiding vendor lock-in, regulatory requirements, or acquisitions. Security must be consistent across all of them.

## Challenges

- Different IAM models (AWS IAM vs Azure AD vs GCP IAM)
- Different networking constructs (VPC vs VNet vs VPC)
- Different logging formats and locations
- Different compliance tooling
- Increased attack surface

## Unified Security Controls

### Identity Federation
Use a single identity provider (Okta, Azure AD, Ping) federated to all clouds. Consistent MFA, conditional access, and provisioning.

### Infrastructure as Code
Define security controls in Terraform/Pulumi that deploy consistently across clouds. Review IaC like application code.

### Centralized Logging
Forward all cloud logs to a single SIEM. Normalize field names for consistent detection rules.

### CSPM (Cloud Security Posture Management)
Tools like Wiz, Orca, or Prowler scan all clouds with unified policies.

<Callout type="tip" title="Start with identity federation">Unified identity is the highest-leverage multi-cloud security investment. One IdP means consistent MFA, access reviews, and deprovisioning across all environments.</Callout>

## Summary

Multi-cloud security requires unified identity, consistent IaC, centralized logging, and cross-cloud posture management. Do not let each cloud evolve its own security silo.

<Quiz
  chapterSlug="144-multi-cloud-security-strategy"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the highest-leverage security investment for multi-cloud environments?",
      options: ["Buying separate security tools for each cloud", "Identity federation with a single IdP", "Using only one cloud provider", "Disabling cross-cloud networking"],
      correctIndex: 1,
      explanation: "Identity federation ensures consistent authentication, MFA, and access control across all clouds from a single source of truth. It is the foundation of unified multi-cloud security.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does CSPM stand for?",
      options: ["Cloud Service Provider Management", "Cloud Security Posture Management", "Central Security Policy Module", "Cloud Storage Protection Mechanism"],
      correctIndex: 1,
      explanation: "CSPM tools continuously scan cloud environments for misconfigurations, compliance violations, and security risks across one or more cloud providers.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why should cloud logs be centralized in a single SIEM?",
      options: ["To reduce storage costs", "To enable cross-cloud correlation and consistent detection rules", "Because individual cloud logging is unreliable", "To comply with GDPR"],
      correctIndex: 1,
      explanation: "Centralized logging enables correlation across clouds, consistent detection rules, and unified incident investigation. Without it, attacks spanning multiple clouds go undetected.",
      randomize: true,
    }
  ]}
/>
