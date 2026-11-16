---
entity_id: clutter_k8s_ingress_tls
title: "Runbook: Kubernetes NGINX Ingress TLS Certificate Rotation via Cert-Manager"
source_type: "ops_clutter"
authors: "Cloud Infrastructure Team"
year: 2026
arxiv_id: "runbook-k8s-19"
---

# Runbook: Kubernetes NGINX Ingress TLS Certificate Rotation via Cert-Manager

**Source Type:** `ops_clutter` | **Authors:** Cloud Infrastructure Team (2026) — `runbook-k8s-19`

**Formula / Snippet:** `kubectl rollout restart deployment/ingress-nginx-controller`

## Raw Unstructured Note
Step-by-step runbook for diagnosing Let's Encrypt ACME HTTP-01 challenge failures on Kubernetes ingress-nginx pods. Inspect ClusterIssuer webhook logs, verify Route53 DNS CAA records, and delete stale CertificateRequest secrets.
