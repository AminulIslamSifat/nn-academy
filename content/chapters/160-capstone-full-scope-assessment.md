---
title: "Capstone: Full-Scope Security Assessment"
description: "Your final project: conduct a complete security assessment and produce a professional report."
read_time: 10
code_time: 10
---

## The Capstone

You have covered the full cybersecurity spectrum: foundations, network security, web security, offensive and defensive operations, cloud, OS internals, reverse engineering, red team, blue team, mobile/IoT, and real-world case studies. Now prove it.

## Project Requirements

Choose ONE of the following capstone options:

### Option A: Full Penetration Test
Select a scope-approved target (CTF machine, vulnerable VM, or authorized application) and produce a professional pentest report:

1. Reconnaissance and enumeration
2. Vulnerability identification (minimum 5 findings)
3. Exploitation with proof of concept
4. Post-exploitation and lateral movement (if applicable)
5. Professional report with executive summary, technical findings, and prioritized remediation

### Option B: Blue Team Detection Lab
Build a home SOC lab and demonstrate detection coverage:

1. Deploy SIEM (Wazuh/Elastic) with agents
2. Run 20+ Atomic Red Team tests across multiple ATT&CK tactics
3. Write Sigma/YARA rules for each detected technique
4. Build a dashboard showing detection coverage
5. Document gaps and improvement plan

### Option C: Security Research Paper
Conduct original research on a security topic:

1. Choose a topic (new vulnerability class, tool analysis, threat actor profiling)
2. Conduct thorough research with primary sources
3. Include practical demonstrations or proof of concept
4. Write a 3000+ word paper with citations
5. Present findings in a clear, structured format

## Deliverables

Regardless of option chosen, submit:

1. **Written report/paper** (professional formatting)
2. **Technical artifacts** (scripts, rules, screenshots, evidence)
3. **Portfolio-ready documentation** (blog post or GitHub repo)

<Callout type="info" title="This capstone IS your portfolio">Employers want to see what you can DO, not just what you know. A well-documented capstone project demonstrates practical skill more effectively than any certification.</Callout>

## Evaluation Criteria

| Criterion | Weight |
|---|---|
| Technical depth and accuracy | 30% |
| Professional report quality | 25% |
| Practical demonstration | 25% |
| Creativity and initiative | 10% |
| Documentation and reproducibility | 10% |

## Where to Go From Here

You now have the knowledge foundation of a professional cybersecurity specialist. Continue growing:

- **Certifications**: OSCP, CISSP, GCIH based on your specialization
- **Community**: BSides conferences, Discord servers, Twitter/X infosec community
- **Practice**: Weekly CTF rooms, bug bounties, open-source contributions
- **Writing**: Blog your learnings, contribute to security research

## Summary

Congratulations. You have completed a comprehensive cybersecurity curriculum from first principles to professional practice. The capstone proves your skill. The journey continues — stay curious, keep practicing, and never stop learning.

<Quiz
  chapterSlug="160-capstone-full-scope-assessment"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What is the most important deliverable of a penetration test?",
      options: ["The exploited vulnerabilities", "A professional report with findings, evidence, and prioritized remediation", "The tools used", "The time spent"],
      correctIndex: 1,
      explanation: "The report is the product. It communicates findings to stakeholders, drives remediation, and demonstrates the value of the engagement. Without a clear report, the pentest has no lasting impact.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "Why is a portfolio more valuable than certifications alone?",
      options: ["Portfolios are free", "A portfolio demonstrates practical, applied skill that certifications cannot prove", "Certifications expire", "Portfolios are required by law"],
      correctIndex: 1,
      explanation: "Certifications validate knowledge. Portfolios (lab writeups, CTF solutions, research papers, bug bounty findings) prove you can apply that knowledge in practice. Together they make the strongest case to employers.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What is the recommended next step after completing this curriculum?",
      options: ["Stop studying", "Choose a specialization, get certified, build a portfolio, and engage with the security community", "Only study theory", "Wait for a job offer"],
      correctIndex: 1,
      explanation: "The curriculum provides the foundation. Specialization, certification, portfolio building, and community engagement turn that foundation into a career. Security is a lifelong learning journey.",
      randomize: true,
    }
  ]}
/>
