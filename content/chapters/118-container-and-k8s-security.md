---
title: "Container & Kubernetes Security"
description: "How containers work, where they break, and how to harden Kubernetes clusters."
read_time: 13
code_time: 7
---

## Why Containers Change the Threat Model

A container is a process with its own filesystem view, network namespace, and cgroup limits — all enforced by the Linux kernel. Unlike a VM, containers **share the host kernel**. A kernel exploit inside one container can potentially reach every other container on that host.

This means container security is about **isolation boundaries**: image, runtime, orchestration, and network.

## The Container Image Supply Chain

An image is a stack of read-only layers. If any layer contains malware, every container started from that image is compromised.

### Common Image Vulnerabilities

- **Base image bloat** — pulling `ubuntu:latest` ships 70+ packages you never use, each with its own CVE history.
- **Outdated packages** — images are built once and never patched unless you rebuild.
- **Embedded secrets** — a `COPY .env /app/.env` in the Dockerfile bakes credentials into the image forever.
- **Unpinned tags** — `FROM node:latest` can silently change under you.

```dockerfile
# Bad: unpinned, bloated, secrets baked in
FROM node:latest
COPY . /app
RUN npm install

# Better: pinned slim image, no secrets
FROM node:20.11-alpine@sha256:abc123
COPY --from=builder /app/dist /app
```

<Callout type="warning" title="Pin by digest, not tag">A tag like `node:20` can be republished with different content. Pinning by sha256 digest guarantees you get exactly the image you audited.</Callout>

## Image Scanning

Tools like **Trivy**, **Grype**, and **Snyk Container** compare each package in your image against CVE databases. Run them in CI before the image is pushed.

```bash
# Example Trivy scan
trivy image --severity HIGH,CRITICAL myapp:1.2.3
```

## Runtime Security

Once a container is running, the attack surface is the process itself:

- **Run as non-root** — `USER 1000` in the Dockerfile. A root container that gets exploited gives the attacker root on the host.
- **Read-only filesystem** — `--read-only` flag forces attackers to find writable paths.
- **Drop capabilities** — containers get a reduced set of Linux capabilities by default, but you can drop more: `--cap-drop=ALL --cap-add=NET_BIND_SERVICE`.
- **No privileged mode** — `--privileged` disables almost all isolation. Never use it in production.
- **Seccomp & AppArmor** — restrict which syscalls the container process can make.

## Secrets Management

Never store secrets in images, environment variables committed to git, or Kubernetes ConfigMaps.

| Method | Security Level |
|---|---|
| Hardcoded in Dockerfile | ❌ Worst |
| Kubernetes ConfigMap | ❌ Plain text in etcd |
| Kubernetes Secret (base64) | ⚠️ Encoding, not encryption |
| Sealed Secrets / SOPS | ✅ Encrypted at rest |
| External Vault (HashiCorp Vault) | ✅ Best — dynamic, audited |

## Kubernetes-Specific Hardening

### RBAC

Kubernetes RBAC controls who can do what via `Role` and `ClusterRole` objects. Apply least privilege: a CI service account that only needs to create Deployments should not have `cluster-admin`.

### Network Policies

By default, every pod can talk to every other pod. A `NetworkPolicy` restricts this:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-ingress
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  # no ingress rules = deny all inbound traffic
```

### Pod Security Standards

Kubernetes 1.25+ replaced PodSecurityPolicy with **Pod Security Admission** labels on namespaces: `restricted`, `baseline`, or `privileged`. Use `restricted` for workloads that do not need special privileges.

<Callout type="tip" title="Defense in depth in K8s">Combine image scanning, non-root runtime, network policies, and RBAC. No single control is sufficient; together they make a compromised pod much less useful to an attacker.</Callout>

## Interactive: Simulate an RBAC Permission Check

<PyRunner
  cellId="118-container-k8s-security-cell-1"
  defaultCode={`# Simulate Kubernetes RBAC permission evaluation
rules = [
  {"verbs": ["get", "list"], "resources": ["pods"], "namespace": "default"},
  {"verbs": ["create"], "resources": ["deployments"], "namespace": "default"},
  {"verbs": ["get"], "resources": ["secrets"], "namespace": "kube-system"},
]

def can_i(rules, verb, resource, namespace):
    for r in rules:
        if verb in r["verbs"] and resource in r["resources"] and namespace == r["namespace"]:
            return True
    return False

tests = [
    ("get", "pods", "default"),
    ("delete", "pods", "default"),
    ("get", "secrets", "default"),
    ("create", "deployments", "default"),
    ("list", "secrets", "kube-system"),
]

print("RBAC Permission Check Simulator")
print("=" * 55)
for verb, resource, ns in tests:
    result = "ALLOWED" if can_i(rules, verb, resource, ns) else "DENIED"
    print(f"  {verb:8s} {resource:12s} in {ns:12s} -> {result}`}
  timeout={8}
  title="K8s RBAC Permission Simulator"
/>

## Summary

Container security spans the entire lifecycle: build a minimal, scanned image; run it unprivileged with a locked-down runtime; and enforce network policies, RBAC, and pod security standards at the orchestration layer. Each layer reduces what an attacker can do after compromising one component.

<Quiz
  chapterSlug="118-container-and-k8s-security"
  questions={[
    {
      id: "q1",
      type: "multiple-choice",
      prompt: "Why is sharing the host kernel a security concern for containers?",
      options: ["Containers use more memory than VMs", "A kernel exploit in one container can affect all containers on the host", "Containers cannot run on different operating systems", "The kernel cannot enforce cgroup limits"],
      correctIndex: 1,
      explanation: "Unlike VMs which have their own kernel, containers share the host kernel. A vulnerability in the kernel can be exploited from any container to compromise the host and all sibling containers.",
      randomize: true,
    },
    {
      id: "q2",
      type: "multiple-choice",
      prompt: "What is the most secure way to handle secrets in a container image?",
      options: ["Copy .env into the image at build time", "Use Kubernetes ConfigMaps", "Use an external secret manager like HashiCorp Vault", "Store them in environment variables in the pod spec"],
      correctIndex: 2,
      explanation: "An external secret manager provides encryption at rest, dynamic rotation, and audit logging. ConfigMaps and baked-in files are plaintext or easily decoded.",
      randomize: true,
    },
    {
      id: "q3",
      type: "multiple-choice",
      prompt: "What does a NetworkPolicy with an empty ingress rule and policyTypes [Ingress] do?",
      options: ["Allows all inbound traffic", "Denies all inbound traffic to matching pods", "Redirects traffic to a firewall pod", "Logs all inbound connections"],
      correctIndex: 1,
      explanation: "A NetworkPolicy that selects pods but defines no ingress rules acts as a default-deny for inbound traffic to those pods.",
      randomize: true,
    },
    {
      id: "q4",
      type: "multiple-choice",
      prompt: "Which flag should you avoid in production containers?",
      options: ["--cap-drop=ALL", "--read-only", "--privileged", "USER 1000"],
      correctIndex: 2,
      explanation: "--privileged disables most kernel-level isolation, giving the container near-full access to the host. It should never be used in production.",
      randomize: true,
    }
  ]}
/>
