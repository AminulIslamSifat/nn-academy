---
title: "Web Application Testing with Proxies"
description: "Intercept, inspect, and modify HTTP traffic to find vulnerabilities in web applications."
read_time: 12
code_time: 7
---

## Why Intercept Traffic?

Browsers hide the raw HTTP conversation. A proxy sits between your browser and the server, letting you **see, pause, modify, and replay** every request and response. This is the core tool for manual web security testing.

The two dominant tools are **Burp Suite** (commercial + free Community edition) and **OWASP ZAP** (fully open source).

## How an Intercepting Proxy Works

```
Browser -> Proxy (localhost:8080) -> Target Server
Browser <- Proxy (inspect/modify) <- Target Server
```

You configure your browser to route traffic through the proxy. The proxy decrypts HTTPS using its own CA certificate (which you install in the browser).

<Callout type="info" title="The CA certificate is key">The proxy generates a unique certificate for each site, signed by its own CA. Without installing the proxy CA in your browser, HTTPS interception will show certificate warnings.</Callout>

## The Manual Testing Workflow

### 1. Spider / Crawl

Let the proxy map the application: discover pages, forms, API endpoints, and parameters. Burp's Spider and ZAP's Spider do this automatically.

### 2. Inspect Requests

Look for interesting patterns:

- Hidden form fields (`<input type="hidden" name="role" value="user">`)
- API endpoints not linked in the UI
- Cookies with sensitive data (base64-encoded JSON, JWTs)
- Verbose error messages
- Unusual headers (`X-Debug: true`)

### 3. Modify and Replay

This is where testing happens:

- **Change parameter values** — `role=user` to `role=admin`
- **Swap HTTP methods** — GET to POST, or add PUT/DELETE
- **Remove or duplicate parameters** — test server-side validation
- **Modify cookies and headers** — test session handling
- **Inject payloads** — XSS, SQLi, SSTI strings into parameters

### 4. Fuzzing

Send many variations of a request to find edge cases:

- Parameter values: numbers, strings, special characters
- HTTP headers: unusual values, missing headers
- Paths: directory traversal, hidden endpoints

## Common Findings

| Vulnerability | What to Look For |
|---|---|
| Broken access control | Accessing /admin as a regular user, IDOR via changing IDs |
| Injection | Error messages that change with special characters |
| XSS | Reflected input without encoding |
| CSRF | State-changing requests without anti-CSRF tokens |
| Information disclosure | Stack traces, debug endpoints, verbose errors |
| Session issues | Predictable tokens, missing Secure/HttpOnly flags |

<Callout type="warning" title="Test in a safe environment">Never test against production without explicit authorization. Use a staging environment that mirrors production. Unauthorized testing is illegal and can cause real damage.</Callout>

## Interactive: Simulate Request Tampering

<PyRunner
  cellId="125-web-app-testing-with-proxies-cell-1"
  defaultCode={`import base64
import json

# Simulated intercepted HTTP request
request = {
    "method": "GET",
    "path": "/api/user/profile",
    "headers": {
        "Cookie": "session=eyJ1c2VyIjogImpkb2UiLCAicm9sZSI6ICJ1c2VyIn0=",
        "User-Agent": "Mozilla/5.0",
    },
    "params": {"user_id": "1042"},
}

print("Intercepted Request:")
print(f"  {request['method']} {request['path']}?user_id={request['params']['user_id']}")
print(f"  Cookie: {request['headers']['Cookie']}")

# Decode the session cookie
cookie_value = request["headers"]["Cookie"].split("=")[1]
decoded = base64.b64decode(cookie_value).decode()
session_data = json.loads(decoded)
print(f"\nDecoded session: {session_data}")

# Tamper: escalate role
session_data["role"] = "admin"
tampered = base64.b64encode(json.dumps(session_data).encode()).decode()
print(f"\nTampered cookie: session={tampered}")
print(f"Tampered session: {session_data}")

# Tamper: IDOR - access another user
tampered_params = dict(request["params"])
tampered_params["user_id"] = "1001"
print(f"\nIDOR test: changing user_id from {request['params']['user_id']} to {tampered_params['user_id']}")

# Check responses
print("\n--- Simulated Server Responses ---")
tests = [
    ("Original request", 200, "Access granted for user jdoe (role: user)"),
    ("Tampered role=admin", 403, "Forbidden: role change detected in session token"),
    ("IDOR user_id=1001", 200, "Access granted - profile for user admin_user"),
]
for name, status, body in tests:
    flag = "VULN" if status == 200 and "admin" in name.lower() or "IDOR" in name else "OK"
    print(f"  [{flag:4s}] {status} - {name}: {body}")`}
  timeout={8}
  title="Request Tampering Simulator"
/>

## Burp vs ZAP

| Feature | Burp Suite | OWASP ZAP |
|---|---|---|
| Cost | Community free, Pro paid | Fully free |
| Intercept | Excellent manual workflow | Good |
| Scanner | Industry-leading (Pro) | Good active/passive |
| Extensibility | BApp Store (Java/Python) | Add-ons (Python) |
| Learning curve | Moderate | Moderate |

For learning, ZAP is a great starting point. For professional work, Burp Suite Pro is the industry standard.

## Summary

An intercepting proxy is the web tester's microscope. Crawl the app, inspect every request, tamper with parameters and headers, and observe how the server responds. The vulnerabilities you find this way — broken access control, injection, IDOR — are the ones automated scanners miss.

<Quiz
  chapterSlug="125-web-app-testing-with-proxies"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why must you install the proxy CA certificate in your browser?",
      options: ["To speed up HTTPS connections", "To allow the proxy to decrypt and re-encrypt HTTPS traffic without certificate warnings", "To enable the proxy to block ads", "To compress responses"],
      correctIndex: 1,
      explanation: "The proxy generates its own certificates for each site. Without trusting the proxy CA, the browser would show certificate errors for every HTTPS site.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is an IDOR vulnerability?",
      options: ["A type of SQL injection", "Accessing another user data by changing an identifier in the request", "A cross-site scripting attack", "A denial of service technique"],
      correctIndex: 1,
      explanation: "Insecure Direct Object Reference (IDOR) occurs when changing an ID parameter (like user_id) lets you access another user's data without proper authorization checks.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the purpose of fuzzing in web testing?",
      options: ["To crash the server", "To send many request variations and find edge cases or unexpected behavior", "To encrypt test data", "To bypass the proxy"],
      correctIndex: 1,
      explanation: "Fuzzing sends many variations of inputs (parameters, headers, paths) to discover edge cases, validation gaps, and unexpected error handling that could indicate vulnerabilities.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which finding is most likely to be caught by manual proxy testing but missed by automated scanners?",
      options: ["Missing Content-Security-Policy header", "Broken access control via role tampering", "Outdated server software version", "Missing X-Frame-Options header"],
      correctIndex: 1,
      explanation: "Broken access control requires understanding the application logic and testing specific role/permission changes. Automated scanners can detect missing headers and outdated versions, but logic flaws need human judgment.",
      randomize: true,
    }
  ]}
/>
