---
title: "Identity and Access Management"
description: "Core IAM concepts including authentication, authorization, SSO, RBAC, ABAC, least privilege, and MFA."
read_time: 12
code_time: 6
---

## Authentication vs Authorization

Authentication verifies identity. Authorization verifies permission. Confusing these two concepts is the most common IAM mistake in security engineering.

- **Authentication (AuthN):** Proves you are who you claim to be. Answers "Who are you?"
- **Authorization (AuthZ):** Determines what you can do after identity is proven. Answers "What are you allowed to do?"

A user authenticates with a password and MFA token. The system then authorizes that user to read `/api/reports` but denies access to `/api/admin`. These are separate enforcement points with distinct failure modes.

## Authentication Methods Compared

Select authentication methods based on threat model, usability requirements, and compliance mandates.

| Method | Security Level | Phishing Resistant | User Friction | Best For |
|--------|---------------|-------------------|---------------|----------|
| Password Only | Low | No | Low | Legacy systems only |
| Password + SMS OTP | Medium | No | Medium | Consumer accounts |
| Password + TOTP | Medium-High | Partially | Medium | Internal tools |
| FIDO2/WebAuthn | High | Yes | Low | High-value targets |
| Certificate-Based | High | Yes | Medium | Machine-to-machine |
| Biometric + PIN | Medium-High | Partially | Low | Mobile/device unlock |

<Callout type="danger" title="Credential Stuffing Risk">Over 80% of account takeover attacks use credentials leaked from unrelated breaches. Password-only authentication is insufficient for any production system. Always enforce MFA and monitor for credential reuse against known breach databases like Have I Been Pwned.</Callout>

## Multi-Factor Authentication

MFA requires two or more independent evidence types from different categories:

1. **Something you know:** Password, PIN, security question answer
2. **Something you have:** Hardware token, smartphone, YubiKey, certificate
3. **Something you are:** Fingerprint, face recognition, iris scan
4. **Somewhere you are:** Geolocation, network origin, device posture

Effective MFA combines factors across categories. Two passwords do not constitute MFA. A password plus an SMS code sent to the same phone used for password recovery provides weaker assurance than password plus hardware key.

### MFA Implementation Priorities

- Prefer FIDO2/WebAuthn over TOTP over SMS OTP
- Enforce MFA for all privileged accounts without exception
- Allow users to register multiple second factors for account recovery
- Log and alert on MFA failures as potential attack indicators
- Support step-up authentication for sensitive operations

## Single Sign-On and Federation

SSO lets users authenticate once and access multiple applications without re-entering credentials. Federation extends this trust across organizational boundaries using standard protocols.

### SAML 2.0

Security Assertion Markup Language uses XML-based assertions exchanged between an Identity Provider (IdP) and Service Provider (SP).

- **Best for:** Enterprise web applications, legacy integrations
- **Flow:** Browser redirects carry signed XML assertions from IdP to SP
- **Strengths:** Mature ecosystem, broad vendor support
- **Weaknesses:** Complex XML parsing, larger payloads, harder to debug

### OpenID Connect (OIDC)

OIDC adds an identity layer on top of OAuth 2.0 using JSON Web Tokens (JWTs).

- **Best for:** Modern APIs, mobile apps, microservices
- **Flow:** Authorization code flow returns ID token (JWT) plus access token
- **Strengths:** Lightweight, REST-friendly, built-in discovery
- **Weaknesses:** Token lifecycle management, JWT validation pitfalls

<Callout type="tip" title="Protocol Selection">Choose OIDC for new projects. Reserve SAML for integrating with enterprise systems that require it. Both protocols solve federation; OIDC does so with less complexity for modern architectures.</Callout>

## Role-Based Access Control (RBAC)

RBAC assigns permissions to roles, then assigns roles to users. This decouples permissions from individual identities and simplifies administration.

### Core RBAC Components

- **Permissions:** Atomic actions on resources (e.g., `reports:read`, `users:delete`)
- **Roles:** Named collections of permissions (e.g., `analyst`, `admin`)
- **Users:** Identities assigned one or more roles
- **Sessions:** Active role sets during a login session

### RBAC Limitations

RBAC struggles with context-dependent decisions. You cannot express "allow access only during business hours" or "deny if risk score exceeds threshold" without triggering role explosion—the proliferation of narrowly scoped roles to capture context.

## Attribute-Based Access Control (ABAC)

ABAC evaluates policies against attributes of the subject, resource, action, and environment at decision time.

### ABAC Policy Example

```
PERMIT
  WHEN subject.department == resource.owner_department
  AND subject.clearance >= resource.classification
  AND environment.time BETWEEN 08:00 AND 18:00
  AND action IN [read, write]
```

### RBAC vs ABAC Comparison

| Dimension | RBAC | ABAC |
|-----------|------|------|
| Decision basis | Assigned roles | Evaluated attributes |
| Granularity | Coarse to medium | Fine-grained |
| Administration | Simple role assignment | Policy authoring required |
| Scalability | Degrades with complexity | Scales with attribute richness |
| Audit trail | Role membership changes | Policy evaluation logs |
| Best fit | Stable org structures | Dynamic, context-sensitive access |

Many organizations combine both: RBAC for baseline entitlements, ABAC for conditional overrides and sensitive resources.

## Least Privilege Principle

Grant the minimum permissions necessary to perform a task. Revoke them when the task ends. This limits blast radius from compromised accounts and insider threats.

### Practical Enforcement Steps

1. Start with zero permissions and add incrementally based on observed need
2. Use just-in-time access elevation instead of standing privileges
3. Conduct quarterly access reviews with mandatory justification
4. Automate deprovisioning tied to HR system events
5. Separate duties so no single identity controls critical workflows end-to-end

<Callout type="warning" title="Privilege Creep">Users accumulate permissions over time through role changes and temporary escalations that never get revoked. Automated access certification campaigns catch drift before attackers exploit it.</Callout>

## RBAC Permission Check Simulator

Experiment with role-to-permission mappings to understand how RBAC enforcement works at runtime.

<PyRunner
  cellId="117-identity-and-access-management-cell-1"
  defaultCode={`# RBAC Permission Check Simulator
roles = {
    "viewer": ["reports:read", "dashboards:read"],
    "analyst": ["reports:read", "reports:write", "dashboards:read", "data:export"],
    "admin": ["reports:read", "reports:write", "reports:delete",
              "dashboards:read", "dashboards:write", "users:manage"]
}

def check_permission(user_role, requested_permission):
    if user_role not in roles:
        return f"DENY: Unknown role '{user_role}'"
    if requested_permission in roles[user_role]:
        return f"ALLOW: '{user_role}' has '{requested_permission}'"
    return f"DENY: '{user_role}' lacks '{requested_permission}'"

# Test cases
print(check_permission("analyst", "reports:write"))
print(check_permission("viewer", "reports:delete"))
print(check_permission("admin", "users:manage"))
print(check_permission("guest", "reports:read"))`}
  timeout={8}
  title="RBAC Permission Check Simulator"
/>

Modify the `roles` dictionary and test cases to explore how adding or removing permissions affects authorization decisions. Notice that deny-by-default is the baseline.

## Key Takeaways

- Authentication and authorization are distinct enforcement points requiring separate controls
- MFA must combine truly independent factors; SMS OTP is the weakest acceptable second factor
- SSO reduces credential sprawl but creates high-value targets; protect your IdP accordingly
- RBAC scales well until context matters; supplement with ABAC for dynamic policies
- Least privilege is an ongoing process, not a one-time configuration

<Quiz
  questions={[
    {
      text: "Which statement correctly distinguishes authentication from authorization?",
      options: [
        "Authentication verifies identity; authorization verifies permission",
        "Authentication checks permissions; authorization verifies identity",
        "They are interchangeable terms for access control",
        "Authentication applies to users; authorization applies to machines"
      ],
      correctIndex: 0
    },
    {
      text: "Why is SMS OTP considered the weakest form of MFA?",
      options: [
        "SMS can be intercepted via SIM swapping and SS7 vulnerabilities",
        "SMS tokens expire too quickly for practical use",
        "SMS requires expensive hardware tokens",
        "SMS is incompatible with mobile devices"
      ],
      correctIndex: 0
    },
    {
      text: "When should you choose ABAC over pure RBAC?",
      options: [
        "When access decisions depend on contextual attributes like time or location",
        "When your organization has fewer than ten employees",
        "When all users need identical permissions",
        "When compliance requires simpler audit logs"
      ],
      correctIndex: 0
    },
    {
      text: "What is the primary risk of privilege creep?",
      options: [
        "Accumulated unused permissions expand the attack surface over time",
        "Users forget their original job responsibilities",
        "Role definitions become too simple to manage",
        "MFA tokens expire faster for privileged accounts"
      ],
      correctIndex: 0
    }
  ]}
/>
