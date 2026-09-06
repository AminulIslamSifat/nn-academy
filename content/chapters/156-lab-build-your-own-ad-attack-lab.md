---
title: "Lab: Build Your Own AD Attack Lab"
description: "Step-by-step guide to building a realistic Active Directory lab for practicing attacks safely."
read_time: 13
code_time: 8
---

## Why Build an AD Lab?

AD attacks cannot be learned from books alone. You need a live environment to practice enumeration, exploitation, and lateral movement. This lab gives you that.

## Lab Architecture

```
[DC01] Windows Server 2019 - Domain Controller
  |
[WS01] Windows 10 - Workstation (joined to domain)
  |
[WS02] Windows 10 - Workstation (joined to domain)
  |
[KALI] Kali Linux - Attack machine
```

All on a host-only virtual network (no internet).

## Step-by-Step Setup

### 1. Install Windows Server 2019 Evaluation
Free 180-day eval from Microsoft. Promote to Domain Controller, create domain corp.local.

### 2. Create Users and Groups
```powershell
New-ADUser -Name "John Smith" -SamAccountName jsmith -Enabled $true
New-ADUser -Name "SQL Service" -SamAccountName svc_sql -Enabled $true
Set-ADAccountPassword svc_sql -Reset -NewPassword (ConvertTo-SecureString "SQLpass123!" -AsPlainText -Force)
Set-ADUser svc_sql -ServicePrincipalNames @{Add="MSSQLSvc/db01.corp.local"}
```

### 3. Join Workstations
Join WS01 and WS02 to corp.local. Log in as different users to create session data.

### 4. Install Attack Tools on Kali
```bash
sudo apt install impacket-scripts crackmapexec bloodhound neo4j
```

### 5. Practice Attacks
- Enumerate with BloodHound
- Kerberoast svc_sql
- Lateral movement with CrackMapExec
- Dump credentials with Mimikatz

<Callout type="warning" title="Keep the lab isolated">Never connect your AD lab to the internet or your home network. Host-only networking prevents accidental exposure.</Callout>

## Interactive: Lab Setup Checklist

<PyRunner
  cellId="156-lab-build-ad-attack-lab-cell-1"
  defaultCode={`checklist = [
    {"step": "Download Windows Server 2019 Eval ISO", "done": True, "notes": "Free from Microsoft Evaluation Center"},
    {"step": "Create VM with 4GB RAM, 60GB disk", "done": True, "notes": "Minimum for DC"},
    {"step": "Promote to Domain Controller (corp.local)", "done": True, "notes": "dcpromo or Server Manager"},
    {"step": "Create 5+ user accounts with varied privileges", "done": False, "notes": "Include service accounts with SPNs"},
    {"step": "Create 2 Windows 10 workstation VMs", "done": False, "notes": "Free eval from Microsoft"},
    {"step": "Join workstations to domain", "done": False, "notes": "Log in as different users"},
    {"step": "Install Kali Linux attack VM", "done": False, "notes": "Pre-built from Offensive Security"},
    {"step": "Configure host-only network", "done": False, "notes": "NO internet access"},
    {"step": "Install BloodHound + Neo4j on Kali", "done": False, "notes": "bloodhound-python or SharpHound"},
    {"step": "Install Impacket + CrackMapExec", "done": False, "notes": "pip install impacket crackmapexec"},
    {"step": "Take VM snapshots", "done": False, "notes": "Snapshot before every attack practice"},
]

print("AD Attack Lab Setup Checklist")
print("=" * 55)

done_count = sum(1 for c in checklist if c["done"])
total = len(checklist)
pct = (done_count / total) * 100

for item in checklist:
    status = "DONE" if item["done"] else "TODO"
    marker = "x" if item["done"] else " "
    print(f"  [{marker}] {item[step]}")
    if item["notes"]:
        print(f"      Note: {item[notes]}")

print(f"\nProgress: {done_count}/{total} ({pct:.0f}%)")
if done_count == total:
    print("Lab is ready! Start with BloodHound enumeration.")
else:
    print(f"Remaining: {total - done_count} steps")`}
  timeout={8}
  title="Lab Setup Tracker"
/>

## Summary

Building an AD lab is the single best investment for learning enterprise security. Follow the setup steps, keep it isolated, and practice systematically through the ATT&CK matrix. Document your findings for your portfolio.

<Quiz
  chapterSlug="156-lab-build-your-own-ad-attack-lab"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why must an AD lab be on a host-only network?",
      options: ["To save bandwidth", "To prevent lab traffic and malware from reaching real networks", "Host-only is faster", "It is required by VMware"],
      correctIndex: 1,
      explanation: "AD labs run attack tools and malware. Host-only networking ensures nothing escapes to your home network or the internet, preventing accidental damage or legal issues.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why should you take VM snapshots before practicing attacks?",
      options: ["To save disk space", "To quickly revert to a clean state after breaking something during practice", "Snapshots improve performance", "It is required by the hypervisor"],
      correctIndex: 1,
      explanation: "Attack practice often breaks things. Snapshots let you instantly revert to a known-good state, saving hours of rebuild time and encouraging experimentation.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the first tool to run when enumerating an AD environment?",
      options: ["Mimikatz", "BloodHound", "Nmap", "Metasploit"],
      correctIndex: 1,
      explanation: "BloodHound maps AD relationships (users, groups, ACLs, sessions) and finds attack paths. It provides the strategic overview that guides all subsequent attacks.",
      randomize: true,
    }
  ]}
/>
