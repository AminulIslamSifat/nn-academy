---
title: "Authentication & Session Security"
description: "Passwords, MFA, session tokens, JWTs, and keeping users safe after login."
read_time: 12
code_time: 7
---

## Authentication Factors

Authentication proves identity. The three classic factors:

- **Something you know** — password, PIN, security question.
- **Something you have** — phone, hardware key, smart card.
- **Something you are** — fingerprint, face, iris.

Multi-factor authentication combines these so one stolen factor is not enough.

<Callout type="info" title="MFA blocks 99% of automated attacks">Microsoft's research shows that MFA blocks over 99.9% of account compromise attacks. Even a simple SMS-based second factor dramatically raises the bar.</Callout>

## Password Storage Done Right

Hash with a slow, salted, memory-hard function. Never store plaintext, never use fast hashes like MD5 for passwords.

```python
import hashlib
import os

# Correct password hashing
salt = os.urandom(16)
hash_value = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 600_000)

# Store: salt + hash_value
# Verify: recompute with the same salt and compare
```

Preferred algorithms in order: **Argon2id** > **scrypt** > **bcrypt** > **PBKDF2** (high iterations).

## Session Management

After login, the server issues a session token. Protect it:

- Set cookies **HttpOnly** (not readable by JavaScript) and **Secure** (HTTPS only).
- Set **SameSite** to `Strict` or `Lax` to block cross-site request forgery.
- Regenerate the session ID at privilege changes to stop session fixation.
- Set reasonable expiration times.

### Session Fixation Attack

An attacker sets a known session ID on the victim's browser before login. If the server does not regenerate the session ID after authentication, the attacker uses the same session ID to access the victim's authenticated session.

```python
# Vulnerable: session ID persists across login
@app.route("/login")
def login():
    if verify_credentials(username, password):
        session["user_id"] = user.id  # Same session ID!
        return redirect("/dashboard")

# Fixed: regenerate session on login
@app.route("/login")
def login():
    if verify_credentials(username, password):
        session.regenerate()  # New session ID
        session["user_id"] = user.id
        return redirect("/dashboard")
```

## JWT Security

JSON Web Tokens are popular for stateless authentication, but they have specific pitfalls:

| Pitfall | Fix |
|---|---|
| `alg: none` accepted | Always validate the algorithm against a whitelist |
| Secret key too weak | Use 256+ bit secrets or asymmetric keys |
| No expiry check | Always validate the `exp` claim |
| Sensitive data in payload | JWT payloads are base64, not encrypted |
| No token revocation | Implement a revocation list or short expiry |

```python
# Vulnerable: accepts any algorithm
token = jwt.decode(token, key)  # Could accept alg: none

# Safe: enforce algorithm
token = jwt.decode(token, key, algorithms=["HS256"])
```

<Callout type="warning" title="JWT payloads are not encrypted">A JWT payload is base64url-encoded, not encrypted. Anyone can decode it. Never put passwords, SSNs, or other sensitive data in JWT claims.</Callout>

## Common Attack Vectors

### Credential Stuffing

Attackers take username/password pairs leaked from one breach and try them on your site. Since people reuse passwords, this works at scale.

**Defense:** rate limiting, MFA, breach-list checking (HaveIBeenPwned API).

### Session Hijacking

Steal the session token via XSS, network sniffing, or physical access.

**Defense:** HttpOnly + Secure + SameSite cookies, short session timeouts.

### CSRF (Cross-Site Request Forgery)

Trick a logged-in user into performing an action on your site from a malicious page.

**Defense:** anti-CSRF tokens, SameSite cookies.

## Interactive: Password Hashing with Salt

<PyRunner
  cellId="111-authentication-sessions-cell-1"
  defaultCode={`import hashlib
import os

# Demonstrate password hashing with salt
password = "MyS3cur3Pass!"

# Generate a random salt
salt = os.urandom(16)

# Hash with PBKDF2 (600,000 iterations)
iterations = 600_000
dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, dklen=32)

print("Password Hashing Demonstration")
print("=" * 55)
print(f"Password: {password}")
print(f"Salt (hex): {salt.hex()}")
print(f"Iterations: {iterations:,}")
print(f"Hash (hex): {dk.hex()}")
print(f"Hash length: {len(dk)} bytes ({len(dk.hex())} hex chars)")

# Verify: same password + same salt = same hash
dk_verify = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, dklen=32)
print(f"\nVerification: {'MATCH' if dk == dk_verify else 'MISMATCH'}")

# Wrong password = different hash
dk_wrong = hashlib.pbkdf2_hmac('sha256', b"WrongPassword", salt, iterations, dklen=32)
print(f"Wrong password: {'MATCH' if dk == dk_wrong else 'MISMATCH (correct)'}")

# Different salt = different hash (prevents rainbow tables)
salt2 = os.urandom(16)
dk_salt2 = hashlib.pbkdf2_hmac('sha256', password.encode(), salt2, iterations, dklen=32)
print(f"Same password, different salt: {'MATCH' if dk == dk_salt2 else 'DIFFERENT (correct)'}")

print("\n--- Storage Format ---")
print(f"Store in DB: algorithm$iterations$salt_hex$hash_hex")
print(f"  pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()[:32]}...")`}
  timeout={8}
  title="Password Hashing Lab"
/>

## Interactive: Session Security Checker

<PyRunner
  cellId="111-authentication-sessions-cell-2"
  defaultCode={`# Evaluate session cookie security settings

def check_cookie_security(cookie_settings):
    issues = []
    score = 100

    if not cookie_settings.get("httponly"):
        issues.append("Missing HttpOnly: JavaScript can read the session token (XSS risk)")
        score -= 25

    if not cookie_settings.get("secure"):
        issues.append("Missing Secure flag: cookie sent over HTTP (sniffing risk)")
        score -= 25

    if cookie_settings.get("samesite") == "None":
        issues.append("SameSite=None: vulnerable to CSRF attacks")
        score -= 20
    elif cookie_settings.get("samesite") not in ("Strict", "Lax"):
        issues.append("SameSite not set: default behavior may allow CSRF")
        score -= 10

    if cookie_settings.get("max_age", 0) > 86400:
        issues.append(f"Session too long ({cookie_settings['max_age']//3600}h): increases hijack window")
        score -= 10

    return max(score, 0), issues

# Test scenarios
scenarios = [
    ("Insecure defaults", {"httponly": False, "secure": False, "samesite": "None", "max_age": 604800}),
    ("Partially secured", {"httponly": True, "secure": False, "samesite": "Lax", "max_age": 3600}),
    ("Fully secured", {"httponly": True, "secure": True, "samesite": "Strict", "max_age": 3600}),
]

print("Session Cookie Security Assessment")
print("=" * 55)

for name, settings in scenarios:
    score, issues = check_cookie_security(settings)
    grade = "A" if score >= 90 else "B" if score >= 70 else "C" if score >= 50 else "F"
    print(f"\n  {name}: {score}/100 (Grade {grade})")
    if issues:
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("    All security flags properly set!")`}
  timeout={8}
  title="Session Security Checker"
/>

## Summary

Strong authentication and hardened session handling together stop account takeover, which remains the most common real-world breach path. Use slow salted hashes, enforce MFA, protect session tokens with HttpOnly/Secure/SameSite, validate JWTs strictly, and regenerate sessions at privilege boundaries.

<Quiz
  chapterSlug="111-authentication-sessions"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does the HttpOnly flag on a cookie prevent?",
      options: ["The cookie from being sent over HTTP", "JavaScript from reading the cookie value", "The cookie from expiring", "The cookie from being deleted"],
      correctIndex: 1,
      explanation: "HttpOnly prevents client-side JavaScript from accessing the cookie. This protects the session token from being stolen via XSS attacks.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is a session fixation attack?",
      options: ["Brute forcing the session token", "Setting a known session ID on the victim before login and using it after they authenticate", "Stealing the session via network sniffing", "Guessing the session ID from the timestamp"],
      correctIndex: 1,
      explanation: "In session fixation, the attacker sets a known session ID on the victim's browser. If the server does not regenerate the session ID after login, the attacker uses the same ID to access the authenticated session.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why should you never accept 'alg: none' in JWT validation?",
      options: ["It makes the token too large", "It allows an attacker to forge tokens without any signature", "It is deprecated by the JWT standard", "It causes encoding errors"],
      correctIndex: 1,
      explanation: "The 'none' algorithm means no signature verification. If accepted, an attacker can create arbitrary tokens with any claims and the server will trust them. Always validate against a whitelist of expected algorithms.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What is credential stuffing?",
      options: ["Guessing passwords with a dictionary", "Using leaked username/password pairs from one breach to try logging into other services", "Injecting SQL into the login form", "Intercepting login traffic with a proxy"],
      correctIndex: 1,
      explanation: "Credential stuffing takes leaked credentials from one breach and tries them across many other services. Since people reuse passwords, this attack succeeds at scale.",
      randomize: true,
    },
    {
      id: "q5",
      type: "multiple-choice",
      prompt: "What is the primary defense against CSRF attacks?",
      options: ["Strong passwords", "Anti-CSRF tokens and SameSite cookie attribute", "HTTPS encryption", "Account lockout after failed attempts"],
      correctIndex: 1,
      explanation: "Anti-CSRF tokens ensure that state-changing requests include a unique token that an attacker's cross-site page cannot predict. SameSite cookies prevent the browser from sending cookies on cross-site requests.",
      randomize: true,
    }
  ]}
/>
