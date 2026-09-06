---
title: "Wireless Security"
description: "Why radio is a broadcast medium, and how Wi-Fi protections have evolved."
read_time: 8
code_time: 4
---

## The Broadcast Problem

Radio waves travel through walls. Every wireless transmission can be received by anyone in range, so wireless **requires** encryption by default.

## The Evolution of Wi-Fi Security

| Protocol | Status | Weakness |
|---|---|---|
| WEP | Broken | Weak IVs, trivially cracked |
| WPA / TKIP | Deprecated | Reuse vulnerabilities |
| WPA2 / CCMP | Still used | KRACK, offline PSK guessing |
| WPA3 | Current | SAE handshake, stronger protection |

## Pre-Shared Key Risk

On a PSK network, anyone who captures the 4-way handshake can guess the password offline. Use a long, random passphrase or move to enterprise (WPA-Enterprise with a RADIUS server).

<Callout type="warning" title="Evil twin attacks">An attacker sets up a fake access point with a familiar name. Clients connect to the stronger signal and hand over credentials. Always verify network identity.</Callout>

## Hardening Checklist

- Use WPA3 where possible, else WPA2 with a strong passphrase.
- Disable WPS.
- Isolate guest traffic from the internal network.
- Monitor for rogue access points.

## Summary

Wireless is inherently exposed, so it demands strong encryption, enterprise authentication, and continuous monitoring for rogue devices.
