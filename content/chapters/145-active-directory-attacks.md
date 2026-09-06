---
title: "Active Directory Attacks"
description: "Kerberoasting, AS-REP roasting, DCSync, Golden Tickets — how AD gets compromised in practice."
read_time: 14
code_time: 8
---

## Why Active Directory?

AD is the identity backbone of most enterprises. Compromising AD means compromising the entire organization. Red teamers target AD because it is the highest-value target.

## Common AD Attack Paths

### Kerberoasting
Request service tickets for SPN-enabled accounts, crack offline:

```bash
# Request tickets
Rubeus kerberoast /outfile:hashes.txt

# Crack with hashcat
hashcat -m 13100 hashes.txt wordlist.txt
```

### AS-REP Roasting
Accounts without pre-authentication can be roasted:

```bash
Rubeus asreproast /outfile:asrep_hashes.txt
```

### DCSync
Replicate password hashes from domain controllers (requires Domain Admin or Replication rights):

```bash
mimikatz # lsadump::dcsync /domain:corp.local /user:krbtgt
```

### Golden Ticket
Forge a Kerberos TGT using the krbtgt hash. Grants unlimited access for 10 years.

<Callout type="danger" title="Golden Ticket is game over">Once an attacker has the krbtgt hash and creates a Golden Ticket, they have persistent, undetectable domain admin access. The only remediation is resetting krbtgt twice and rebuilding trust.</Callout>

## BloodHound

BloodHound maps AD relationships and finds attack paths automatically. Upload collected data and query for shortest paths to Domain Admin.

## Interactive: AD Attack Path Analyzer

<PyRunner
  cellId="145-active-directory-attacks-cell-1"
  defaultCode={`users = [
    {"name": "jsmith", "groups": ["Domain Users"], "spn": None, "preauth": True, "admin_count": 0},
    {"name": "svc_sql", "groups": ["Domain Users", "SQL Admins"], "spn": "MSSQLSvc/db01.corp.local", "preauth": True, "admin_count": 0},
    {"name": "svc_backup", "groups": ["Domain Users", "Backup Operators"], "spn": None, "preauth": False, "admin_count": 0},
    {"name": "admin_jones", "groups": ["Domain Admins"], "spn": None, "preauth": True, "admin_count": 1},
    {"name": "svc_web", "groups": ["Domain Users"], "spn": "HTTP/web01.corp.local", "preauth": True, "admin_count": 0},
]

print("Active Directory Attack Surface Analysis")
print("=" * 60)

kerberoastable = [u for u in users if u["spn"]]
asrep_roastable = [u for u in users if not u["preauth"]]
da_members = [u for u in users if "Domain Admins" in u["groups"]]
high_priv = [u for u in users if u["admin_count"] > 0 or "Backup Operators" in u["groups"]]

print(f"\nKerberoastable accounts ({len(kerberoastable)}):")
for u in kerberoastable:
    print(f"  {u[name]:15s} SPN: {u[spn]}")

print(f"\nAS-REP Roastable accounts ({len(asrep_roastable)}):")
for u in asrep_roastable:
    print(f"  {u[name]:15s} Pre-auth disabled")

print(f"\nDomain Admins ({len(da_members)}):")
for u in da_members:
    print(f"  {u[name]}")

print(f"\nHigh-privilege accounts ({len(high_priv)}):")
for u in high_priv:
    privs = [g for g in u["groups"] if g not in ["Domain Users"]]
    print(f"  {u[name]:15s} Groups: {, .join(privs)}")

print("\n--- Recommended Attack Path ---")
if kerberoastable:
    print(f"  1. Kerberoast {kerberoastable[0][name]} -> crack hash")
if asrep_roastable:
    print(f"  2. AS-REP roast {asrep_roastable[0][name]} (no pre-auth)")
print("  3. Enumerate local admin access with cracked creds")
print("  4. Pivot to DA via BloodHound path")
print("  5. DCSync krbtgt -> Golden Ticket")`}
  timeout={8}
  title="AD Attack Path Analyzer"
/>

## Summary

AD attacks follow predictable patterns: enumerate, kerberoast/AS-REP roast, lateral movement, privilege escalation to DA, persistence via Golden Ticket. Defenders must monitor for these specific techniques and harden service accounts.

<Quiz
  chapterSlug="145-active-directory-attacks"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is Kerberoasting?",
      options: ["Brute forcing AD passwords online", "Requesting service tickets for SPN-enabled accounts and cracking the hashes offline", "Exploiting Kerberos protocol vulnerabilities", "Stealing Kerberos tickets from memory"],
      correctIndex: 1,
      explanation: "Kerberoasting requests TGS tickets for service accounts (which are encrypted with the service account NTLM hash). The attacker cracks these hashes offline to recover the service account password.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What makes a Golden Ticket so dangerous?",
      options: ["It expires quickly", "It forges a Kerberos TGT using the krbtgt hash, granting unlimited domain access for years", "It only works for one hour", "It requires physical access"],
      correctIndex: 1,
      explanation: "A Golden Ticket is a forged TGT signed with the krbtgt hash. Since the DC trusts any ticket signed with this key, the attacker has persistent, nearly undetectable domain admin access.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What tool maps AD relationships and finds attack paths?",
      options: ["Nmap", "BloodHound", "Wireshark", "Metasploit"],
      correctIndex: 1,
      explanation: "BloodHound collects AD data (users, groups, ACLs, sessions) and uses graph theory to find the shortest attack paths from any compromised account to high-value targets like Domain Admin.",
      randomize: true,
    }
  ]}
/>
