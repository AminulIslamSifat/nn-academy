---
title: "Windows Internals for Security"
description: "Registry, services, tokens, Active Directory integration, and Windows-specific attack surfaces."
read_time: 14
code_time: 7
---

## Why Windows Matters

Windows dominates enterprise desktops and is heavily present in server environments. Most corporate breaches involve Windows systems. Understanding Windows internals is non-negotiable for security professionals.

## The Windows Registry

The registry is a hierarchical database storing virtually all system and application configuration.

Key hives for security:

| Hive | What It Contains |
|---|---|
| HKLM\SYSTEM | Boot config, services, drivers |
| HKLM\SOFTWARE | Installed software, security policies |
| HKLM\SAM | Local user accounts and password hashes |
| HKCU | Per-user settings, autoruns |

Persistence via registry:
```powershell
# Common persistence locations
HKCU\Software\Microsoft\Windows\CurrentVersion\Run
HKLM\Software\Microsoft\Windows\CurrentVersion\Run
HKLM\System\CurrentControlSet\Services
```

<Callout type="info" title="Autoruns are the first persistence check">When investigating a compromised Windows system, always check autorun locations first. Tools like Sysinternals Autoruns enumerate all startup points.</Callout>

## Windows Services

Services run in the background with specific accounts (often SYSTEM). Misconfigured services are a major privilege escalation vector:

- **Unquoted service paths** — if the path has spaces and no quotes, Windows tries each segment as an executable.
- **Weak service permissions** — if a regular user can modify a service binary or configuration.
- **Service DLL hijacking** — placing a malicious DLL in a location the service searches before the legitimate one.

```powershell
# Check service permissions
sc qc ServiceName
icacls "C:\Path\To\Service.exe"

# Find unquoted service paths
wmic service get name,pathname | findstr /i /v "C:\Windows" | findstr /i /v '"'
```

## Access Tokens and Privileges

Every process has an **access token** containing:
- User SID and group SIDs
- Privileges (SeDebugPrivilege, SeImpersonatePrivilege, etc.)
- Integrity level (Low, Medium, High, System)

Key dangerous privileges:

| Privilege | Risk |
|---|---|
| SeDebugPrivilege | Attach to any process, dump credentials |
| SeImpersonatePrivilege | Impersonate other users (JuicyPotato) |
| SeBackupPrivilege | Read any file regardless of ACL |
| SeRestorePrivilege | Write any file regardless of ACL |

## Active Directory Integration

Windows domains use Kerberos authentication:

1. User authenticates to Domain Controller, receives TGT.
2. TGT used to request service tickets for specific resources.
3. Service ticket presented to the target service.

Common AD attacks exploit this flow:
- **Kerberoasting** — request service tickets for SPNs, crack offline.
- **AS-REP Roasting** — request TGTs for accounts without pre-auth.
- **Golden Ticket** — forge TGTs with KRBTGT hash.
- **Silver Ticket** — forge service tickets with service account hash.

<Callout type="warning" title="AD is the crown jewels">In most enterprises, compromising Active Directory means compromising everything. AD attack paths are the primary focus of modern red team engagements.</Callout>

## Interactive: Windows Privilege Escalation Checker

<PyRunner
  cellId="133-windows-internals-cell-1"
  defaultCode={`# Simulate checking Windows privesc vectors
system_info = {
    "services": [
        {"name": "VulnService", "path": "C:\Program Files\App\service.exe", "quoted": False, "user_writable": True},
        {"name": "SafeService", "path": '"C:\Program Files\Safe\service.exe"', "quoted": True, "user_writable": False},
        {"name": "AnotherVuln", "path": "C:\My App\bin\svc.exe", "quoted": False, "user_writable": False},
    ],
    "privileges": ["SeChangeNotifyPrivilege", "SeImpersonatePrivilege", "SeShutdownPrivilege"],
    "scheduled_tasks": [
        {"name": "BackupTask", "run_as": "SYSTEM", "binary": "C:\Scripts\backup.bat", "user_writable_dir": True},
        {"name": "UpdateTask", "run_as": "SYSTEM", "binary": "C:\Windows\update.exe", "user_writable_dir": False},
    ],
    "unpatched": ["CVE-2021-36934 (HiveNightmare)", "CVE-2021-1675 (PrintNightmare)"],
}

dangerous_privs = {"SeDebugPrivilege", "SeImpersonatePrivilege", "SeBackupPrivilege", "SeRestorePrivilege", "SeTakeOwnershipPrivilege"}

print("Windows Privilege Escalation Assessment")
print("=" * 60)

# Check services
print("\n--- Service Vulnerabilities ---")
for svc in system_info["services"]:
    issues = []
    if not svc["quoted"] and " " in svc["path"]:
        issues.append("UNQUOTED PATH with spaces")
    if svc["user_writable"]:
        issues.append("USER-WRITABLE BINARY")
    if issues:
        print(f"  [VULN] {svc['name']}: {', '.join(issues)}")
        print(f"         Path: {svc['path']}")
    else:
        print(f"  [OK]   {svc['name']}")

# Check privileges
print("\n--- Dangerous Privileges ---")
for priv in system_info["privileges"]:
    if priv in dangerous_privs:
        print(f"  [!] {priv} - enables credential theft or impersonation")
    else:
        print(f"  [ ] {priv} - low risk")

# Check scheduled tasks
print("\n--- Scheduled Task Issues ---")
for task in system_info["scheduled_tasks"]:
    if task["user_writable_dir"]:
        print(f"  [VULN] {task['name']}: writable directory, runs as {task['run_as']}")
    else:
        print(f"  [OK]   {task['name']}")

# Known vulns
print("\n--- Unpatched Vulnerabilities ---")
for v in system_info["unpatched"]:
    print(f"  [!] {v}")`}
  timeout={8}
  title="Windows Privesc Checker"
/>

## Windows Event Logs

Key logs for security monitoring:

| Event ID | What It Means |
|---|---|
| 4624 | Successful logon |
| 4625 | Failed logon |
| 4672 | Special privileges assigned (admin) |
| 4688 | Process creation |
| 4698 | Scheduled task created |
| 4720 | User account created |
| 4732 | Member added to privileged group |

## Summary

Windows internals — registry, services, tokens, AD, event logs — define the attack surface of most enterprise environments. Understanding these concepts enables both finding vulnerabilities and building effective detections.

<Quiz
  chapterSlug="133-windows-internals"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What makes an unquoted service path exploitable?",
      options: ["The service runs slower", "Windows tries each path segment as an executable, allowing an attacker to place a malicious binary earlier in the path", "The service crashes", "The registry entry becomes invalid"],
      correctIndex: 1,
      explanation: "If a service path like C:\Program Files\App\svc.exe is unquoted, Windows first tries C:\Program.exe, then C:\Program Files\App.exe. An attacker can place a malicious binary at one of these locations.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does SeImpersonatePrivilege allow?",
      options: ["Reading any file", "Impersonating other user tokens, enabling attacks like JuicyPotato", "Shutting down the system", "Creating new services"],
      correctIndex: 1,
      explanation: "SeImpersonatePrivilege allows a process to impersonate another user's security context. Attackers exploit this to escalate from a service account to SYSTEM.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is Kerberoasting?",
      options: ["Brute forcing domain admin passwords", "Requesting service tickets for SPNs and cracking them offline to recover service account passwords", "Exploiting Kerberos encryption weaknesses", "Stealing the KRBTGT hash"],
      correctIndex: 1,
      explanation: "Kerberoasting requests Kerberos service tickets for accounts with SPNs. The ticket is encrypted with the service account's NTLM hash, which can be cracked offline.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which Windows event ID indicates a new user account was created?",
      options: ["4624", "4625", "4720", "4688"],
      correctIndex: 2,
      explanation: "Event 4720 is logged when a new user account is created. Monitoring this event helps detect unauthorized account creation by attackers establishing persistence.",
      randomize: true,
    }
  ]}
/>
