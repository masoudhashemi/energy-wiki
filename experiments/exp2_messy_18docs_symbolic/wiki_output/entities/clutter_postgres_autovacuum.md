# Postmortem: PostgreSQL Transaction ID Wraparound & Autovacuum Tuning

**Source Type:** `ops_clutter` | **Authors:** SRE Database Reliability Team (2026) | **ID:** `inc-2026-041`

**Formula / Snippet:** `autovacuum_freeze_max_age = 200000000`

**Status:** *Filtered out / Unattended across all concepts ($\sigma(a_i^K) < 0.5$ — Distractor or Out-of-Domain Clutter)*

## Compiled Content & Relational Bridges
Root cause analysis of replica lag spike on primary billing database shard-02. Long-running analytics transactions blocked autovacuum from freezing dead tuples in theWAL segment directory. Tuned vacuum_cost_limit and added pg_stat_activity alert thresholds.
