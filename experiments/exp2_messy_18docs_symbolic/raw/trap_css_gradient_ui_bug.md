---
entity_id: trap_css_gradient_ui_bug
title: "Bug #884: Safari CSS Linear-Gradient Flicker & DOM Attention Focus"
source_type: "polysemous_trap"
authors: "Frontend Design Systems Team"
year: 2026
arxiv_id: "jira-ui-884"
---

# Bug #884: Safari CSS Linear-Gradient Flicker & DOM Attention Focus

**Source Type:** `polysemous_trap` | **Authors:** Frontend Design Systems Team (2026) — `jira-ui-884`

**Formula / Snippet:** `background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%)`

## Raw Unstructured Note
Fixes GPU compositing flicker when animating CSS linear-gradient and radial-gradient backgrounds inside scrollable modal overlays. Adjusts keyboard tab-index attention focus rings and pairwise flexbox sibling margins on retina viewports.
