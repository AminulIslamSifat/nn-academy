---
title: "Careers & Next Steps"
description: "Security roles, key certifications, building a portfolio, and your capstone project to tie it all together."
read_time: 10
code_time: 6
---

## The Security Job Landscape

Cybersecurity is not one job — it is a field with many specializations. Understanding the landscape helps you choose a direction.

### The Main Tracks

| Role | Focus | Entry Path |
|---|---|---|
| **SOC Analyst** | Monitor, triage, escalate alerts | Most common entry point |
| **Penetration Tester** | Authorized offensive testing | OSCP, bug bounties |
| **Security Engineer** | Build and maintain security tooling | Software engineering + security |
| **Incident Responder** | Lead breach containment and forensics | SOC or forensics background |
| **Security Architect** | Design secure systems | Senior engineering + architecture |
| **GRC Analyst** | Compliance, risk, policy | Audit, legal, or compliance background |
| **Malware Analyst** | Reverse engineer malicious code | Low-level programming, OS internals |
| **Threat Intelligence Analyst** | Track adversaries, produce intel | Research, writing, analysis |

<Callout type="info" title="The talent gap is real">The ISC2 Cybersecurity Workforce Study consistently reports millions of unfilled security positions globally. The barrier to entry is knowledge and demonstrated skill, not necessarily a degree.</Callout>

## Key Certifications

Certifications validate knowledge and get past HR filters. Choose based on your target role:

### Entry Level
- **CompTIA Security+** — broad foundational knowledge, widely recognized.
- **CompTIA Network+** — networking fundamentals (prerequisite knowledge).

### Intermediate
- **CySA+** (CompTIA) — security analytics, SOC-oriented.
- **BTL1** (Blue Team Level 1) — defensive operations, hands-on exam.
- **eJPT** — junior penetration testing, practical exam.

### Advanced
- **OSCP** (Offensive Security) — the gold standard for pentesting, 24-hour practical exam.
- **CISSP** — management/architecture, requires 5 years experience.
- **GCIA/GCIH/GCFE** (SANS) — specialized, expensive, highly respected.

<Callout type="tip" title="Certifications open doors, skills keep them">A certification gets you the interview. A home lab, a blog, or a bug bounty profile gets you the job. Invest in both.</Callout>

## Building a Portfolio

Demonstrable skill matters more than credentials in security. Build evidence:

### Home Lab

- Run a hypervisor (VirtualBox, Proxmox) with vulnerable VMs (Metasploitable, DVWA).
- Set up a SIEM (Wazuh, Elastic) and practice detection rules.
- Build an Active Directory lab and practice Kerberoasting, GPO attacks.

### CTF and Practice Platforms

- **TryHackMe** — guided learning paths, beginner-friendly.
- **HackTheBox** — unguided challenge machines, intermediate+.
- **OverTheWire** — Linux and security wargames.
- **PicoCTF** — free, beginner-oriented.

### Bug Bounties

Platforms like **HackerOne** and **Bugcrowd** let you legally test real applications. Even small findings demonstrate real-world skill.

### Writing and Sharing

- Write up your CTF solutions.
- Document your home lab projects.
- Contribute to open-source security tools.

## Interactive: Career Path Planner

<PyRunner
  cellId="131-careers-and-next-steps-cell-1"
  defaultCode={`# Simulated career path recommendation engine
interests = {
    "defensive": ["SOC Analyst", "Incident Responder", "Threat Intelligence Analyst"],
    "offensive": ["Penetration Tester", "Red Teamer", "Exploit Developer"],
    "engineering": ["Security Engineer", "Security Architect", "DevSecOps Engineer"],
    "governance": ["GRC Analyst", "Security Auditor", "CISO track"],
}

recommended_certs = {
    "SOC Analyst": ["Security+", "CySA+", "BTL1"],
    "Incident Responder": ["Security+", "GCIH", "GCFA"],
    "Penetration Tester": ["Security+", "eJPT", "OSCP"],
    "Security Engineer": ["Security+", "AWS Security Specialty", "CISSP (later)"],
    "GRC Analyst": ["Security+", "CISA", "CRISC"],
}

skills_map = {
    "SOC Analyst": ["SIEM (Splunk/Elastic)", "Log analysis", "Network fundamentals", "Scripting"],
    "Penetration Tester": ["Networking", "Linux", "Web security", "Scripting", "Report writing"],
    "Security Engineer": ["Cloud platforms", "CI/CD", "Infrastructure as Code", "Programming"],
    "Incident Responder": ["Forensics", "Memory analysis", "Malware triage", "Communication"],
}

print("Career Path Planner")
print("=" * 55)

for path, roles in interests.items():
    print(f"\n--- {path.upper()} Track ---")
    for role in roles:
        certs = recommended_certs.get(role, ["Security+"])
        skills = skills_map.get(role, ["Security fundamentals"])
        print(f"\n  {role}")
        print(f"    Certs: {' -> '.join(certs)}")
        print(f"    Skills: {', '.join(skills[:3])}...")

# Personalized recommendation
print("\n" + "=" * 55)
print("Recommended starting path for this course:")
print("  1. Complete all Academia modules (you are here!)")
print("  2. Build a home lab with Wazuh + vulnerable VMs")
print("  3. Do 10 TryHackMe rooms")
print("  4. Study for Security+")
print("  5. Write up your lab work on a blog")
print("  6. Apply for SOC Analyst or junior security roles")`}
  timeout={8}
  title="Career Path Planner"
/>

## The Capstone

You have covered the full stack: foundations, network security, web security, offensive and defensive operations, cloud, operations, tooling, and governance. Now tie it together.

### Capstone Project: Security Assessment Report

Choose a target (a vulnerable VM, a CTF machine, or a scope-approved application) and produce a professional security assessment:

1. **Threat model** the target using STRIDE.
2. **Enumerate** the attack surface (nmap, web crawling).
3. **Identify** at least 3 vulnerabilities.
4. **Demonstrate** exploitation in a safe environment.
5. **Recommend** remediation for each finding.
6. **Write** a professional report with executive summary, technical findings, and prioritized remediation plan.

This report becomes a portfolio piece that demonstrates you can do the actual job.

## Where to Go From Here

The field moves fast. Stay current:

- **Read** — Krebs on Security, The Hacker News, SANS NewsBites.
- **Listen** — Darknet Diaries, Risky Business, Security Now.
- **Practice** — weekly CTF rooms, bug bounty hunting.
- **Community** — local BSides conferences, DEF CON groups, Discord servers.

<Callout type="info" title="The learning never stops">Security is a field where the threat landscape changes monthly. The skills you built in this course are the foundation. The habit of continuous learning is what keeps you relevant.</Callout>

## Summary

You have built a comprehensive understanding of cybersecurity from first principles. Choose a specialization, get certified, build a portfolio, and start applying. The field needs people who understand both the theory and the practice — and you now have both.

<Quiz
  chapterSlug="131-careers-and-next-steps"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Which certification is considered the gold standard for penetration testing?",
      options: ["CompTIA Security+", "CISSP", "OSCP", "CEH"],
      correctIndex: 2,
      explanation: "The OSCP (Offensive Security Certified Professional) requires passing a 24-hour hands-on penetration test. It is the most respected certification for offensive security roles.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the most common entry-level role in cybersecurity?",
      options: ["Security Architect", "SOC Analyst", "CISO", "Malware Analyst"],
      correctIndex: 1,
      explanation: "SOC Analyst is the most common entry point. It provides exposure to real security events, log analysis, and incident triage, building the foundation for specialization.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "Why is a portfolio important alongside certifications?",
      options: ["Portfolios are required by law", "A portfolio demonstrates practical skills that certifications alone cannot prove", "Portfolios replace the need for certifications", "Portfolios are only needed for management roles"],
      correctIndex: 1,
      explanation: "Certifications validate knowledge, but a portfolio (home lab writeups, CTF solutions, bug bounty findings) proves you can apply that knowledge in practice. Together they make a strong case to employers.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "What should a capstone security assessment report include?",
      options: ["Only a list of vulnerabilities found", "Threat model, enumeration, findings with exploitation proof, remediation recommendations, and executive summary", "Just the nmap scan output", "A copy of the CISSP exam objectives"],
      correctIndex: 1,
      explanation: "A professional security assessment includes the full methodology: threat modeling, enumeration, vulnerability identification, proof of concept, and prioritized remediation. This mirrors real-world pentest deliverables.",
      randomize: true,
    }
  ]}
/>
