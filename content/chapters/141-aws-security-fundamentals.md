---
title: "AWS Security Fundamentals"
description: "IAM, S3, security groups, and the shared responsibility model in AWS."
read_time: 12
code_time: 6
---

## The Shared Responsibility Model

AWS secures the cloud (infrastructure). You secure what is IN the cloud (data, configurations, access). Most cloud breaches are customer misconfigurations, not AWS vulnerabilities.

## IAM: The Foundation

IAM controls who can do what in AWS. Key concepts:

- **Users** — long-lived identities (avoid for applications)
- **Roles** — temporary assumed identities (preferred)
- **Policies** — JSON documents defining allowed/denied actions
- **Principle of least privilege** — start with deny-all, add only what is needed

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::my-bucket/public/*"
  }]
}
```

<Callout type="danger" title="AdministratorAccess is almost never needed">The managed AdministratorAccess policy grants full access to everything. Use specific, scoped policies instead. Overprivileged IAM is the number one cloud security issue.</Callout>

## S3 Security

S3 buckets are the most commonly misconfigured AWS resource:

- **Bucket policies** — who can access what
- **ACLs** — legacy, prefer bucket policies
- **Block public access** — enable at account level
- **Encryption** — SSE-S3 or SSE-KMS at rest
- **Versioning** — protect against accidental deletion

## Security Groups vs NACLs

| Feature | Security Group | NACL |
|---|---|---|
| Level | Instance | Subnet |
| Rules | Allow only | Allow and deny |
| Stateful | Yes | No |
| Default | Deny all inbound | Deny all |

## Interactive: IAM Policy Analyzer

<PyRunner
  cellId="141-aws-security-fundamentals-cell-1"
  defaultCode={`policies = [
    {"name": "AdminAccess", "actions": ["*"], "resources": ["*"], "effect": "Allow"},
    {"name": "S3ReadOnly", "actions": ["s3:GetObject", "s3:ListBucket"], "resources": ["arn:aws:s3:::data-bucket/*"], "effect": "Allow"},
    {"name": "EC2Full", "actions": ["ec2:*"], "resources": ["*"], "effect": "Allow"},
    {"name": "LambdaInvoke", "actions": ["lambda:InvokeFunction"], "resources": ["arn:aws:lambda:us-east-1:123:function:api-handler"], "effect": "Allow"},
    {"name": "IAMCreateUser", "actions": ["iam:CreateUser", "iam:AttachUserPolicy"], "resources": ["*"], "effect": "Allow"},
]

def assess_policy(policy):
    risks = []
    if "*" in policy["actions"]:
        risks.append("CRITICAL: Wildcard actions (*) grants all permissions")
    elif any(a.endswith(":*") for a in policy["actions"]):
        risks.append("HIGH: Service wildcard (ec2:*) grants all actions in service")
    if "*" in policy["resources"]:
        risks.append("HIGH: Wildcard resource applies to all resources")
    if any("iam:" in a for a in policy["actions"]):
        risks.append("HIGH: IAM permissions can lead to privilege escalation")
    if not risks:
        risks.append("OK: Scoped appropriately")
    return risks

print("AWS IAM Policy Risk Assessment")
print("=" * 55)

for pol in policies:
    risks = assess_policy(pol)
    severity = "CRITICAL" if any("CRITICAL" in r for r in risks) else "WARNING" if any("HIGH" in r for r in risks) else "OK"
    print(f"\n  [{severity:8s}] {pol[name]}")
    print(f"    Actions: {, .join(pol[actions][:3])}{... if len(pol[actions])>3 else }")
    for r in risks:
        print(f"    -> {r}")`}
  timeout={8}
  title="IAM Policy Analyzer"
/>

## Summary

AWS security starts with IAM least privilege, proper S3 configuration, and network segmentation. The shared responsibility model means most security is your job. Automate compliance checks with tools like Prowler and CloudSploit.

<Quiz
  chapterSlug="141-aws-security-fundamentals"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "In the AWS shared responsibility model, who secures S3 bucket configurations?",
      options: ["AWS", "The customer", "Both equally", "Neither"],
      correctIndex: 1,
      explanation: "AWS secures the infrastructure. The customer secures their data, configurations, IAM policies, and access controls. S3 bucket misconfigurations are a customer responsibility.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should you avoid using IAM users with access keys for applications?",
      options: ["Access keys are slow", "Long-lived credentials are a persistent risk if leaked; IAM roles provide temporary credentials", "IAM users cannot access S3", "Access keys cost more"],
      correctIndex: 1,
      explanation: "IAM roles assume temporary credentials that auto-rotate. Access keys are long-lived and if committed to code or leaked, provide persistent access until manually rotated.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the difference between Security Groups and NACLs?",
      options: ["There is no difference", "Security Groups are stateful and instance-level; NACLs are stateless and subnet-level", "NACLs are stateful and Security Groups are stateless", "Security Groups support deny rules"],
      correctIndex: 1,
      explanation: "Security Groups operate at the instance level and are stateful (return traffic is automatically allowed). NACLs operate at the subnet level and are stateless (you must explicitly allow return traffic).",
      randomize: true,
    }
  ]}
/>
