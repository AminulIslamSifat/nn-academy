---
title: "Network Scanning with Nmap"
description: "Discover hosts, enumerate services, and map attack surfaces using the industry-standard network scanner."
read_time: 13
code_time: 8
---

## What Nmap Does

**Nmap** (Network Mapper) sends crafted packets to targets and interprets the responses to answer: What hosts are up? What ports are open? What services and versions are running? What OS is likely?

In a pentest, nmap is usually the first active step after passive reconnaissance.

<Callout type="warning" title="Scanning without authorization is illegal">Port scanning a network you do not own or have written permission to test violates computer fraud laws in most jurisdictions. Always have a signed scope before scanning.</Callout>

## Scan Types

| Scan | Flag | What It Does | Stealth |
|---|---|---|---|
| TCP SYN | `-sS` | Half-open; sends SYN, reads SYN/ACK or RST | Moderate |
| TCP Connect | `-sT` | Full 3-way handshake | Low |
| UDP | `-sU` | Sends UDP probes | Very slow |
| ACK | `-sA` | Maps firewall rules (stateful vs stateless) | Moderate |
| Ping sweep | `-sn` | Host discovery only, no port scan | High |
| Version detect | `-sV` | Probes open ports for service/version | Low |
| OS detect | `-O` | TCP/IP fingerprinting | Low |

```bash
# Typical pentest scan: SYN scan + version detection + top 1000 ports
nmap -sS -sV --top-ports 1000 -oN scan_results.txt 192.168.1.0/24
```

## Reading Nmap Output

```
PORT     STATE  SERVICE  VERSION
22/tcp   open   ssh      OpenSSH 8.9p1
80/tcp   open   http     Apache 2.4.54
443/tcp  open   https    nginx 1.24.0
3306/tcp closed mysql
8080/tcp open   http     Apache Tomcat 9.0.73
```

Key columns: **STATE** (open/closed/filtered), **SERVICE** (protocol guess), **VERSION** (fingerprinted version). A `filtered` state usually means a firewall dropped the packet.

## The Nmap Scripting Engine (NSE)

Nmap includes hundreds of Lua scripts for deeper enumeration:

```bash
# Run default scripts on open ports
nmap -sC -sV target.com

# Run specific vulnerability scripts
nmap --script vuln target.com

# Enumerate DNS
nmap --script dns-brute target.com
```

<Callout type="info" title="NSE categories">Scripts are grouped: `safe`, `intrusive`, `vuln`, `discovery`, `auth`, `exploit`. In a scoped pentest, start with `safe` and `default` categories before anything intrusive.</Callout>

## Scan Strategies

1. **Host discovery first** — `-sn` to find live hosts before port scanning.
2. **Top ports** — `--top-ports 1000` covers 90%+ of real services without scanning all 65535.
3. **Timing templates** — `-T3` is default; `-T4` is faster but noisier; `-T2` is stealthier.
4. **Output formats** — `-oN` (normal text), `-oX` (XML for tools), `-oA` (all formats).

## Interactive: Parse Nmap Output

<PyRunner
  cellId="124-network-scanning-with-nmap-cell-1"
  defaultCode={`import re

# Simulated nmap -sV output
nmap_output = """
Nmap scan report for 192.168.1.10
PORT     STATE  SERVICE      VERSION
22/tcp   open   ssh          OpenSSH 8.9p1 Ubuntu
80/tcp   open   http         Apache httpd 2.4.54
443/tcp  open   https        nginx 1.24.0
3306/tcp closed mysql
5432/tcp open   postgresql   PostgreSQL 15.3
8080/tcp open   http         Apache Tomcat 9.0.73
8443/tcp open   https-alt    Apache Tomcat 9.0.73

Nmap scan report for 192.168.1.20
PORT     STATE  SERVICE      VERSION
22/tcp   open   ssh          OpenSSH 9.0p1
53/tcp   open   domain       ISC BIND 9.18.12
80/tcp   open   http         Microsoft IIS 10.0
445/tcp  open   microsoft-ds Windows Server 2022
3389/tcp open   ms-wbt-server Microsoft Terminal Services
"""

# Parse into structured data
hosts = {}
current_host = None
for line in nmap_output.strip().split("\n"):
    host_match = re.match(r'Nmap scan report for (\S+)', line)
    if host_match:
        current_host = host_match.group(1)
        hosts[current_host] = []
        continue
    port_match = re.match(r'(\d+/\w+)\s+(\w+)\s+(\S+)\s*(.*)', line)
    if port_match and current_host:
        hosts[current_host].append({
            "port": port_match.group(1),
            "state": port_match.group(2),
            "service": port_match.group(3),
            "version": port_match.group(4).strip(),
        })

print("Nmap Output Parser")
print("=" * 60)
for host, ports in hosts.items():
    open_ports = [p for p in ports if p["state"] == "open"]
    print(f"\nHost: {host} ({len(open_ports)} open ports)")
    print(f"  {'PORT':12s} {'SERVICE':16s} {'VERSION'}")
    print(f"  {'-'*12} {'-'*16} {'-'*25}")
    for p in open_ports:
        print(f"  {p['port']:12s} {p['service']:16s} {p['version']}")

# Risk assessment
print("\n--- Quick Risk Assessment ---")
risky = {"mysql": "Database exposed", "postgresql": "Database exposed",
         "microsoft-ds": "SMB - potential lateral movement",
         "ms-wbt-server": "RDP - brute force target"}
for host, ports in hosts.items():
    for p in ports:
        if p["service"] in risky and p["state"] == "open":
            print(f"  [!] {host}:{p['port']} -> {risky[p['service']]}")`}
  timeout={8}
  title="Nmap Output Analyzer"
/>

## Defense: What Scanning Looks Like From the Other Side

Defenders see nmap as:

- A burst of SYN packets to many ports from one IP.
- Unusual TCP flag combinations (NULL, FIN, Xmas scans).
- Rapid connection attempts that do not complete the handshake.

Detection: alert on `>50 SYN packets to distinct ports from one IP within 60 seconds`.

## Summary

Nmap is the reconnaissance workhorse. Master the scan types, read the output critically, use NSE for depth, and always operate within a signed scope. From the defender's side, understanding nmap helps you write better detection rules.

<Quiz
  chapterSlug="124-network-scanning-with-nmap"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does a SYN scan (-sS) do differently from a connect scan (-sT)?",
      options: ["It scans UDP ports instead of TCP", "It sends a SYN and reads the response without completing the full handshake", "It only scans port 80", "It requires root privileges on Windows"],
      correctIndex: 1,
      explanation: "A SYN scan is a half-open scan. It sends a SYN packet and interprets the SYN/ACK (open) or RST (closed) without completing the TCP handshake, making it slightly stealthier.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What does a 'filtered' port state in nmap output typically indicate?",
      options: ["The service is running but encrypted", "A firewall or IDS is dropping the probe packets", "The port is open but the service crashed", "Nmap has a bug"],
      correctIndex: 1,
      explanation: "Filtered means nmap sent a probe but got no response (or an ICMP unreachable). This typically indicates a firewall is silently dropping the packets.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the Nmap Scripting Engine (NSE)?",
      options: ["A GUI for nmap", "A collection of Lua scripts that extend nmap with vulnerability checks and enumeration", "A replacement for nmap", "A Python API for nmap"],
      correctIndex: 1,
      explanation: "NSE allows running Lua scripts against scan targets for deeper enumeration, vulnerability detection, and even exploitation. Scripts are categorized by safety and purpose.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why should you use --top-ports 1000 instead of scanning all 65535 ports?",
      options: ["All ports cannot be scanned", "The top 1000 ports cover the vast majority of real services, saving time", "Scanning all ports is illegal", "Nmap cannot scan more than 1000 ports"],
      correctIndex: 1,
      explanation: "The top 1000 most commonly used ports cover over 90% of real-world services. Scanning all 65535 ports takes significantly longer with diminishing returns.",
      randomize: true,
    }
  ]}
/>
