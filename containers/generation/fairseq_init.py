"""Inference-only Fairseq 0.10.2 initialization for PGMG."""

import sys

__all__ = ["pdb"]
__version__ = "0.10.2"

from fairseq.logging import meters, metrics, progress_bar

sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

# Fairseq utilities must initialize before the transformer module exports:
# utils and multihead_attention intentionally reference each other.
from fairseq import utils
