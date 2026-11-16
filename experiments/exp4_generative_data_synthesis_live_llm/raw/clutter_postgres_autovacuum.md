---
entity_id: clutter_postgres_autovacuum
title: "Postmortem: PostgreSQL Transaction ID Wraparound & Autovacuum Tuning"
source_type: "ops_clutter"
authors: "SRE Database Reliability Team"
year: 2026
arxiv_id: "inc-2026-041"
---

# Postmortem: PostgreSQL Transaction ID Wraparound & Autovacuum Tuning

**Source Type:** `ops_clutter` | **Authors:** SRE Database Reliability Team (2026) — `inc-2026-041`

**Formula / Snippet:** `autovacuum_freeze_max_age = 200000000`

## Raw Unstructured Note
Root cause analysis of replica lag spike on primary billing database shard-02. Long-running analytics transactions blocked autovacuum from freezing dead tuples in theWAL segment directory. Tuned vacuum_cost_limit and added pg_stat_activity alert thresholds.
