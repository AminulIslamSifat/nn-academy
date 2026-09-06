---
title: "Network Security Fundamentals"
description: "How networks actually move packets, and where attackers intercept them."
read_time: 10
code_time: 5
---

## The Layers That Matter

- **IP** — addressing and routing; the source address can be forged.
- **TCP/UDP** — transport; TCP handshakes can be hijacked.
- **DNS** — name resolution; a prime target for redirection attacks.
- **HTTP/HTTPS** — application layer; where most web attacks land.

## Packet Anatomy

Each packet carries headers (who it is from, who it is to) and a payload. Attackers read or rewrite headers to impersonate, redirect, or drop traffic.

## Common Network Attacks

1. **IP spoofing** — forge the source address.
2. **ARP poisoning** — lie about MAC-to-IP mapping on a LAN.
3. **DNS hijacking** — return false IP addresses for a domain.
4. **Sniffing** — passively read unencrypted traffic.

<Callout type="warning" title="Trust nothing on the wire">Anything that travels unencrypted can be read, and anything that is not authenticated can be forged. This is why encryption and authentication are inseparable.</Callout>

## Ports & Services

Every open port is a door. Reducing the attack surface means closing ports you do not need and hardening the ones you keep.

## Summary

You now know the layers, the headers, and the classic interception techniques that later chapters defend against.
