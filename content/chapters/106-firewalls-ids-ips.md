---
title: "Firewalls, IDS & IPS"
description: "The control plane of network defense: filtering, detecting, and preventing unwanted traffic."
read_time: 9
code_time: 5
---

## Firewalls

A firewall enforces an allow/deny policy between networks.

- **Packet-filter** — matches on IP/port, fast but shallow.
- **Stateful** — tracks connection state, understands handshakes.
- **Next-gen / WAF** — inspects application content (e.g. HTTP).

Default policy should be **deny-all, allow-by-exception**.

## IDS vs IPS

- **IDS (Intrusion Detection)** — watches and alerts, does not block.
- **IPS (Intrusion Prevention)** — sits inline and actively blocks.

Detection methods are either **signature-based** (known patterns) or **anomaly-based** (deviation from a learned baseline).

<Callout type="info" title="Anomaly detection connects to ML">Anomaly-based detection is fundamentally a classification problem: normal traffic vs. unusual traffic. The neural-network skills you learned apply directly here.</Callout>

## Segmentation

Divide the network into zones (DMZ, internal, management) so a breach in one zone cannot reach all others.

## Summary

Firewalls decide what may pass, IDS/IPS detect or stop what should not. Together with segmentation they form the network control plane.
