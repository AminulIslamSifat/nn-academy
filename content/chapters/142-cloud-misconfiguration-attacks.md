---
title: "Cloud Misconfiguration Attacks"
description: "Common cloud misconfigurations and how attackers exploit them in real breaches."
read_time: 12
code_time: 6
---

## Why Cloud Misconfigs Dominate

Gartner estimates that through 2025, 99% of cloud security failures will be the customer fault. Misconfigurations are the leading cause of cloud breaches.

## Top Misconfigurations

### 1. Public S3 Buckets
Accidentally exposing sensitive data. Capital One breach (2019): 100 million records exposed via misconfigured WAF and S3.

### 2. Overprivileged IAM
Roles with wildcards (*), unused admin accounts, access keys in code repositories.

### 3. Open Security Groups
Port 22 (SSH), 3389 (RDP), or database ports open to 0.0.0.0/0.

### 4. Missing Encryption
Data at rest or in transit without encryption. Compliance violation and data exposure risk.

### 5. Disabled Logging
CloudTrail, VPC Flow Logs, or GuardDuty not enabled. Breaches go undetected.

<Callout type="warning" title="Automate configuration checks">Manual review does not scale. Use Prowler, CloudSploit, or AWS Config Rules to continuously audit your cloud environment against CIS benchmarks.</Callout>

## Attack Chains

```
Public S3 bucket -> download customer data
    OR
Leaked access key -> enumerate permissions -> escalate via iam:AttachUserPolicy
    OR
Open RDP -> brute force -> lateral movement via instance metadata
```

## Instance Metadata Service (IMDS)

Every EC2 instance can reach http://169.254.169.254 to retrieve its IAM role credentials. If an attacker achieves SSRF or code execution on the instance, they can steal these credentials.

**Defense:** Use IMDSv2 (requires session token) instead of IMDSv1.

## Interactive: Cloud Misconfiguration Scanner

<PyRunner
  cellId="142-cloud-misconfiguration-attacks-cell-1"
  defaultCode={`configs = [
    {"resource": "s3://customer-data-prod", "public": True, "encrypted": False, "logging": False, "versioning": False},
    {"resource": "sg-web-prod", "ports_open": ["22/tcp 0.0.0.0/0", "80/tcp 0.0.0.0/0", "443/tcp 0.0.0.0/0"], "issue": "SSH open to internet"},
    {"resource": "iam-role-dev-admin", "policy": "AdministratorAccess", "last_used": "never", "mfa": False},
    {"resource": "rds-prod-db", "public": False, "encrypted": True, "multi_az": False, "backup_retention": 0},
    {"resource": "ec2-api-server", "imds_version": "v1", "metadata_service": True},
]

print("Cloud Misconfiguration Scanner")
print("=" * 60)

for cfg in configs:
    issues = []
    res = cfg["resource"]
    
    if cfg.get("public") and "s3" in res:
        issues.append("CRITICAL: S3 bucket publicly accessible")
    if not cfg.get("encrypted", True) and "s3" in res:
        issues.append("HIGH: S3 bucket not encrypted at rest")
    if not cfg.get("logging", True) and "s3" in res:
        issues.append("MEDIUM: S3 access logging disabled")
    if "issue" in cfg:
        issues.append(f"HIGH: {cfg[issue]}")
    if cfg.get("policy") == "AdministratorAccess":
        issues.append("CRITICAL: Full admin privileges assigned")
    if cfg.get("last_used") == "never":
        issues.append("MEDIUM: Role never used - consider removing")
    if not cfg.get("mfa", True) and "iam" in res:
        issues.append("HIGH: MFA not enabled")
    if cfg.get("backup_retention", 7) == 0 and "rds" in res:
        issues.append("HIGH: No backup retention configured")
    if cfg.get("imds_version") == "v1":
        issues.append("HIGH: IMDSv1 enabled - vulnerable to SSRF credential theft")
    
    severity = "CRITICAL" if any("CRITICAL" in i for i in issues) else "WARNING" if issues else "OK"
    print(f"\n  [{severity:8s}] {res}")
    for issue in issues:
        print(f"    ! {issue}")
    if not issues:
        print("    Configuration looks good")`}
  timeout={8}
  title="Cloud Config Scanner"
/>

## Summary

Cloud misconfigurations are the dominant breach vector. Automate checks, enforce least privilege, enable logging everywhere, and use IMDSv2. Treat infrastructure-as-code reviews with the same rigor as application code reviews.

<Quiz
  chapterSlug="142-cloud-misconfiguration-attacks"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is IMDSv2 more secure than IMDSv1?",
      options: ["It encrypts the metadata", "It requires a session token, preventing simple SSRF-based credential theft", "It rotates credentials faster", "It disables the metadata service entirely"],
      correctIndex: 1,
      explanation: "IMDSv2 requires a PUT request to obtain a session token before accessing metadata. Simple SSRF attacks typically only support GET requests, making credential theft significantly harder.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "According to Gartner, what percentage of cloud security failures are the customer fault?",
      options: ["50%", "75%", "99%", "25%"],
      correctIndex: 2,
      explanation: "Gartner estimated that through 2025, 99% of cloud security failures would be the customer fault, primarily due to misconfigurations rather than cloud provider vulnerabilities.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What tool automates AWS security configuration checks against CIS benchmarks?",
      options: ["Nmap", "Prowler", "Burp Suite", "Metasploit"],
      correctIndex: 1,
      explanation: "Prowler is an open-source tool that audits AWS configurations against CIS benchmarks, best practices, and compliance frameworks. It identifies misconfigurations before attackers do.",
      randomize: true,
    }
  ]}
/>
