#!/usr/bin/env python3
"""Launch the repo's CLM System One server (repo/src/clm/server.py) for a non-8B encoder.

The repo hard-codes the encoder width (HIDDEN = 4096, Qwen3-8B) for the raw ablation
and the vector-cache pools; this sets it to ``--hidden`` and then runs the unchanged
server. All other arguments go to clm.server.

    python serve_clm.py --hidden 1024 -- --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings \
        --emb-model qwen3-0.6b --ckpt runs/choice_qwen3-0.6b_p20_s1234/best_head.pt --no-download
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo", "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, required=True)
ours, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]

import clm.heads
import clm.engine
clm.heads.HIDDEN = clm.engine.HIDDEN = ours.hidden
from clm import server  # noqa: E402

sys.argv = ["clm-serve", *rest]
server.main()
