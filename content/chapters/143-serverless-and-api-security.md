---
title: "Serverless & API Security"
description: "Securing Lambda functions, API Gateway, and serverless architectures against injection and authorization flaws."
read_time: 11
code_time: 6
---

## Serverless Changes the Threat Model

No servers to patch, but new attack surfaces emerge: function code, event sources, API Gateway configurations, and IAM permissions between services.

## Common Serverless Vulnerabilities

### 1. Injection in Event Data
Lambda functions process events from S3, SQS, API Gateway, DynamoDB streams. Unvalidated event data leads to injection.

### 2. Overprivileged Functions
Each Lambda should have minimal IAM permissions. A function that only reads from S3 should not have write access to DynamoDB.

### 3. API Gateway Misconfigurations
Missing authentication, overly permissive CORS, no rate limiting, verbose error responses.

### 4. Dependency Vulnerabilities
Serverless functions bundle dependencies. Outdated packages in the deployment package are exploitable.

<Callout type="info" title="Serverless does not mean security-less">Serverless removes infrastructure management but introduces new configuration and code-level risks. The OWASP Serverless Top 10 covers these specifically.</Callout>

## API Security Fundamentals

- Authenticate every request (JWT, API keys, OAuth)
- Authorize per-resource (not just per-endpoint)
- Rate limit to prevent abuse
- Validate and sanitize all input
- Use HTTPS only
- Implement proper error handling (no stack traces)

## Interactive: API Security Checklist

<PyRunner
  cellId="143-serverless-api-security-cell-1"
  defaultCode={`endpoints = [
    {"path": "/api/users", "method": "GET", "auth": True, "rate_limit": True, "input_validation": True, "cors": "specific"},
    {"path": "/api/admin/delete-user", "method": "POST", "auth": True, "rate_limit": False, "input_validation": True, "cors": "wildcard"},
    {"path": "/api/search", "method": "GET", "auth": False, "rate_limit": False, "input_validation": False, "cors": "wildcard"},
    {"path": "/api/upload", "method": "POST", "auth": True, "rate_limit": True, "input_validation": True, "cors": "specific"},
    {"path": "/health", "method": "GET", "auth": False, "rate_limit": True, "input_validation": True, "cors": "none"},
]

print("API Security Assessment")
print("=" * 60)

for ep in endpoints:
    issues = []
    if not ep["auth"] and ep["path"] != "/health":
        issues.append("No authentication")
    if not ep["rate_limit"]:
        issues.append("No rate limiting")
    if not ep["input_validation"]:
        issues.append("No input validation")
    if ep["cors"] == "wildcard":
        issues.append("Wildcard CORS (allows any origin)")
    if "admin" in ep["path"] and not ep["rate_limit"]:
        issues.append("Admin endpoint without rate limiting")
    
    status = "SECURE" if not issues else f"RISK ({len(issues)})"
    print(f"\n  [{status}] {ep[method]} {ep[path]}")
    for issue in issues:
        print(f"    ! {issue}")
    if not issues:
        print("    All checks passed")`}
  timeout={8}
  title="API Security Checker"
/>

## Summary

Serverless and APIs require shifting security focus from infrastructure to code, configuration, and IAM. Validate all event data, minimize function permissions, secure API Gateway, and keep dependencies updated.

<Quiz
  chapterSlug="143-serverless-and-api-security"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is input validation critical in serverless functions?",
      options: ["Serverless functions cannot use firewalls", "Event data from S3, SQS, and API Gateway is untrusted and can contain injection payloads", "Serverless functions are slower", "Input validation reduces cold start time"],
      correctIndex: 1,
      explanation: "Serverless functions receive events from various sources, all of which should be treated as untrusted. Without validation, injection attacks through event data are possible.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the recommended CORS configuration for production APIs?",
      options: ["Allow all origins (*)", "Specify exact allowed origins", "Disable CORS entirely", "Allow all subdomains"],
      correctIndex: 1,
      explanation: "Wildcard CORS (*) allows any website to make authenticated requests to your API. Specify exact origins to prevent cross-site attacks while allowing legitimate clients.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why should each Lambda function have minimal IAM permissions?",
      options: ["To reduce cost", "If compromised, the blast radius is limited to only what that function can access", "Lambda does not support broad permissions", "To improve performance"],
      correctIndex: 1,
      explanation: "Least privilege limits what an attacker can do after compromising a function. A function with read-only S3 access cannot modify databases or create new IAM users even if exploited.",
      randomize: true,
    }
  ]}
/>
