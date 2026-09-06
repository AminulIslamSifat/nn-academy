

## Network Segmentation Patterns

### The Classic DMZ

Each boundary enforces specific rules. The web tier cannot directly query the database; it must go through the application tier.

### Microsegmentation (Modern)

In cloud and Kubernetes environments, segmentation is applied per-workload:

- **Service mesh** (Istio, Linkerd) enforces mTLS between services.
- **Network policies** restrict pod-to-pod communication.
- **Identity-based access** replaces IP-based rules.

## Secure Design Patterns

### 1. Fail Secure

When a system encounters an error, it should deny access rather than grant it.

### 2. Least Privilege

Every component gets only the permissions it needs.

### 3. Defense in Depth

No single control is sufficient. Layer them so if one fails, the next catches the attack.

### 4. Secure Defaults

New users and components should be secure without any configuration.

<Callout type="tip" title="Draw the data flow">Before implementing anything, draw how data flows through your system. Mark where it crosses trust boundaries, where it is encrypted, and where it is stored.</Callout>

## Cloud-Specific Architecture

- **VPC design** - public/private subnets, NAT gateways, VPC peering.
- **IAM** - roles and policies, not long-lived access keys.
- **Secrets** - use the cloud provider secret manager, not environment variables.
- **Logging** - enable CloudTrail/Activity Log and send to a separate, immutable account.

## Summary

Security architecture is about making deliberate design choices that reduce attack surface and limit blast radius. Threat model early, segment networks, apply least privilege, fail secure, and layer your defenses.


<Quiz
  chapterSlug="129-security-architecture-design"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "What does fail secure mean in security architecture?",
      options: ["The system should crash on error", "When a system encounters an error, it should deny access rather than grant it", "Errors should be logged but not acted upon", "The system should restart automatically"],
      correctIndex: 1,
      explanation: "Fail secure means that when an error occurs, the system defaults to denying access.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the primary purpose of network segmentation?",
      options: ["To improve network speed", "To limit lateral movement if one zone is compromised", "To reduce the number of firewalls needed", "To simplify DNS configuration"],
      correctIndex: 1,
      explanation: "Segmentation divides the network into zones with strict access controls between them.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "In STRIDE threat modeling, what does Elevation of Privilege ask?",
      options: ["Can the server be overwhelmed?", "Can an attacker gain higher access than they should have?", "Can data be modified in transit?", "Can actions be denied?"],
      correctIndex: 1,
      explanation: "Elevation of Privilege asks whether an attacker can escalate their access level beyond what is authorized.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Why is threat modeling most effective during the design phase?",
      options: ["Because developers are available", "Because fixing flaws on the whiteboard costs far less than in production", "Because compliance requires it", "Because it replaces penetration testing"],
      correctIndex: 1,
      explanation: "Design-phase changes are cheap. Retrofitting security into a built system requires rework and potential downtime.",
      randomize: true,
    }
  ]}
/>
