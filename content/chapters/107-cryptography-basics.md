---
title: "Applied Cryptography"
description: "Symmetric and asymmetric encryption, hashing, and how they combine to secure real systems."
read_time: 12
code_time: 8
---

## The Three Primitives

- **Symmetric encryption** — one shared key (AES). Fast, used for bulk data.
- **Asymmetric encryption** — public/private key pair (RSA, ECC). Slow, used for key exchange and signatures.
- **Hashing** — one-way digest (SHA-256). Integrity and password storage.

## Confidentiality in Practice

<InlineMath latex="C = E_k(P)" /> where <InlineMath latex="E" /> encrypts plaintext <InlineMath latex="P" /> with key <InlineMath latex="k" />. Only someone holding <InlineMath latex="k" /> can recover <InlineMath latex="P" />.

## Key Exchange

Asymmetric crypto lets two strangers agree on a symmetric key over an open channel (Diffie-Hellman). Then they use the fast symmetric cipher for the session. This is exactly how TLS works.

## Hashing Is Not Encryption

A hash cannot be reversed. You store password hashes, never plaintext. To resist brute force you add a **salt** and a slow function (bcrypt, scrypt, Argon2).

<Callout type="warning" title="Never store plaintext passwords">Storing plaintext passwords is an automatic breach the moment your database leaks. Hash with a strong, salted, slow algorithm.</Callout>

## Digital Signatures

Sign with your private key; anyone with your public key can verify. This gives authenticity and non-repudiation.

## Summary

Encryption provides confidentiality, hashing provides integrity, signatures provide authenticity. Real systems weave all three together.
