"""
Runner Script for Experiment Suite 1: Clean 10-Paper KB (Symbolic Relational Critic)
====================================================================================
Executes K=10 Textual SGLD concept inference, attention identification, wiki state
compilation, and Experiments 1-5 on the 10 clean research papers in `raw/`,
exporting `wiki_output/` and `wiki_viewer.html`.
"""

from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import setup_exp1_and_exp2

if __name__ == "__main__":
    setup_exp1_and_exp2.setup_exp1_clean(ROOT_DIR)
