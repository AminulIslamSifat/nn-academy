---
title: "The Shared Responsibility Model"
description: "Understanding security boundaries between cloud providers and customers across IaaS, PaaS, and SaaS."
read_time: 10
code_time: 5
---

## Security Is a Partnership

Cloud security is never solely the provider's job. The Shared Responsibility Model defines exactly which security tasks belong to the cloud provider and which belong to you. Misunderstanding this boundary is the leading cause of cloud breaches. You cannot secure what you do not own, but you also cannot assume protection for assets that remain under your control.

## The Core Division

Responsibility splits into two distinct categories regardless of service model:

-   **Security OF the Cloud:** The provider protects the infrastructure running the services. This includes hardware, global infrastructure, hypervisors, and physical data centers.
-   **Security IN the Cloud:** The customer secures their specific usage. This includes guest operating systems, application code, identity management, firewall configurations, and data encryption.

The line between these two shifts depending on whether you use IaaS, PaaS, or SaaS.

## Service Model Breakdown

### Infrastructure as a Service (IaaS)

You retain the most responsibility in IaaS environments like Amazon EC2 or Azure Virtual Machines. The provider manages the physical host and network fabric. You manage everything above the hypervisor.

-   **Customer:** OS patching, application security, network traffic filtering, IAM policies, data-at-rest encryption.
-   **Provider:** Physical security, compute hardware, storage hardware, virtualization layer.

### Platform as a Service (PaaS)

In PaaS offerings like AWS RDS or Azure App Service, the provider absorbs operating system and runtime maintenance. Your focus narrows to application logic and data.

-   **Customer:** Application code, database configuration, identity integration, client-side encryption.
-   **Provider:** OS patching, middleware updates, runtime environment, database engine maintenance.

### Software as a Service (SaaS)

SaaS solutions like Salesforce or Microsoft 365 shift nearly all infrastructure responsibility to the provider. Your primary duty becomes access governance and data classification.

-   **Customer:** User accounts, MFA enforcement, data sharing policies, compliance monitoring.
-   **Provider:** Application security, infrastructure, OS, middleware, physical assets.

## Comparison Table

| Responsibility Area | IaaS (e.g., EC2) | PaaS (e.g., RDS) | SaaS (e.g., Office 365) |
| :--- | :--- | :--- | :--- |
| Physical Infrastructure | Provider | Provider | Provider |
| Network Controls | Shared | Provider | Provider |
| Operating System | **Customer** | Provider | Provider |
| Middleware/Runtime | **Customer** | Provider | Provider |
| Application Code | **Customer** | **Customer** | Provider |
| Data Encryption | **Customer** | Shared | Shared |
| Identity & Access | **Customer** | **Customer** | **Customer** |
| Configuration Mgmt | **Customer** | **Customer** | Limited |

<Callout type="danger" title="The Configuration Trap">Most cloud breaches result from customer misconfiguration, not provider failure. Leaving an S3 bucket public or disabling MFA on administrative accounts falls entirely on the customer, even in managed services.</Callout>

## Practical Examples: Patching and Encryption

Understanding abstract models requires concrete mapping. Consider how patching differs across AWS services:

-   **EC2 (IaaS):** You must schedule, test, and apply all OS and application patches. The provider only patches the underlying host during scheduled maintenance windows.
-   **RDS (PaaS):** AWS applies database engine patches automatically during your maintenance window. You still patch any custom stored procedures or connected applications.
-   **S3 (Managed Storage):** There is no OS to patch. AWS handles all backend maintenance. You configure bucket policies and lifecycle rules.

Encryption follows a similar pattern. In IaaS, you manage keys and implement volume encryption. In PaaS, the provider often encrypts data at rest by default, but you may need to enable customer-managed keys (CMK) for regulatory compliance. In SaaS, encryption is typically transparent, though you control who accesses the decrypted content.

## Interactive: Check Patch Responsibility

Use the simulator below to determine who handles patching for different service types. The script parses a dictionary of common cloud services and outputs the responsible party.

<PyRunner
  cellId="116-shared-responsibility-model-cell-1"
  defaultCode={`services = {
    "EC2": "IaaS",
    "RDS": "PaaS",
    "Lambda": "Serverless",
    "Office365": "SaaS",
    "AppService": "PaaS"
}

responsibilities = {
    "IaaS": "Customer patches OS and Apps",
    "PaaS": "Provider patches OS/Runtime; Customer patches App",
    "Serverless": "Provider patches everything except Code",
    "SaaS": "Provider patches all software layers"
}

print("=== Patch Responsibility Report ===")
for service, model in services.items():
    owner = responsibilities.get(model, "Unknown Model")
    print(f"{service} ({model}): {owner}")`}
  timeout={8}
  title="Check Patch Responsibility"
/>

## Key Takeaways

1.  **Identify your service model first.** Never assume security controls transfer automatically between IaaS, PaaS, and SaaS.
2.  **Automate configuration checks.** Use tools like AWS Config or Azure Policy to detect drift from secure baselines.
3.  **Review provider documentation.** Responsibility matrices change frequently as new managed features launch.
4.  **Assume breach.** Design architectures where customer-side failures do not cascade into total compromise.
5.  **Audit shared controls.** Network segmentation and IAM require coordination between provider capabilities and customer implementation.

## Knowledge Check

<Quiz
  questions={[
    {
      text: "In an IaaS environment like EC2, who is responsible for patching the guest operating system?",
      options: [
        "The customer",
        "The cloud provider",
        "Shared equally",
        "The hypervisor vendor"
      ],
      correctIndex: 0,
      explanation: "In IaaS, the customer manages everything above the hypervisor, including the guest OS, applications, and data."
    },
    {
      text: "Which security task remains the customer's responsibility across ALL cloud service models?",
      options: [
        "Identity and Access Management",
        "Physical data center security",
        "Hypervisor patching",
        "Hardware replacement"
      ],
      correctIndex: 0,
      explanation: "Customers always control who accesses their resources, regardless of whether they use IaaS, PaaS, or SaaS."
    },
    {
      text: "What is the most common cause of cloud security incidents according to industry reports?",
      options: [
        "Customer misconfiguration",
        "Provider infrastructure failure",
        "Zero-day hypervisor exploits",
        "Physical data center breaches"
      ],
      correctIndex: 0,
      explanation: "Misconfigurations such as open storage buckets, excessive permissions, and disabled logging account for the vast majority of cloud breaches."
    },
    {
      text: "In a PaaS offering like AWS RDS, who applies database engine patches?",
      options: [
        "The cloud provider",
        "The customer DBA",
        "Third-party vendor",
        "No patching required"
      ],
      correctIndex: 0,
      explanation: "PaaS providers handle runtime and engine maintenance. Customers focus on schema design, queries, and application integration."
    }
  ]}
/>
