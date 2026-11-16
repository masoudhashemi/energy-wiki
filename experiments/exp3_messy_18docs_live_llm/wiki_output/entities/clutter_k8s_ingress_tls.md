# Runbook: Kubernetes NGINX Ingress TLS Certificate Rotation via Cert-Manager

**Source Type:** `ops_clutter` | **Authors:** Cloud Infrastructure Team (2026) | **ID:** `runbook-k8s-19`

**Formula / Snippet:** `ops_clutter`

**Status:** *Filtered out / Unattended across all concepts ($\sigma(a_i^K) < 0.5$ — Distractor or Out-of-Domain Clutter)*

## Compiled Content & Relational Bridges
Step-by-step runbook for diagnosing Let's Encrypt ACME HTTP-01 challenge failures on Kubernetes ingress-nginx pods. Inspect ClusterIssuer webhook logs, verify Route53 DNS CAA records, and delete stale CertificateRequest secrets.
