---
title: "Android Security Model"
description: "Sandboxing, permissions, intents, and common Android vulnerabilities."
read_time: 11
code_time: 6
---

## Android Architecture

Android uses a layered architecture with Linux kernel at the base, providing process isolation via UID-based sandboxing. Each app runs as a unique user.

## Key Security Mechanisms

### Application Sandboxing
Each app gets a unique UID. Apps cannot access each other data without explicit permission.

### Permission Model
Apps declare required permissions in AndroidManifest.xml. Users grant at install (pre-6.0) or runtime (6.0+).

### Intents
Inter-app communication mechanism. Exported components receiving intents are common vulnerability points.

### SELinux
Mandatory access control enforcing policies beyond UID sandboxing.

## Common Vulnerabilities

- Exported activities/services/receivers without permission checks
- Insecure data storage (SharedPreferences, SQLite without encryption)
- WebView JavaScript interface exposure
- Intent injection / interception
- Root detection bypass

<Callout type="info" title="ADB is your primary tool">Android Debug Bridge lets you install apps, pull files, view logs, and interact with the device. Enable USB debugging on test devices for security assessment.</Callout>

## Summary

Android security relies on sandboxing, permissions, and SELinux. Vulnerabilities arise from misconfigured exports, insecure storage, and intent handling. Mobile security testing requires understanding these mechanisms deeply.

<Quiz
  chapterSlug="153-android-security-model"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "How does Android isolate applications from each other?",
      options: ["Using virtual machines", "Each app runs as a unique Linux UID with sandboxed filesystem access", "Apps share memory but not storage", "Isolation is handled by the app store"],
      correctIndex: 1,
      explanation: "Android assigns each app a unique Linux UID. The kernel enforces that processes cannot access each other memory or files, providing strong isolation at the OS level.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why are exported Android components a security concern?",
      options: ["They use more battery", "They can be invoked by other apps, potentially without proper authorization checks", "They cannot be updated", "They require root access"],
      correctIndex: 1,
      explanation: "Exported activities, services, and broadcast receivers can be called by any app. Without proper permission checks or input validation, they become attack vectors for privilege escalation or data theft.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What tool is essential for Android security testing?",
      options: ["Nmap", "ADB (Android Debug Bridge)", "Wireshark", "Ghidra"],
      correctIndex: 1,
      explanation: "ADB provides command-line access to Android devices for installing apps, pulling data, viewing logs, and interacting with the system. It is the primary tool for mobile security assessments.",
      randomize: true,
    }
  ]}
/>
