"""Inference-only Fairseq 0.10.2 module exports required by PGMG."""

from .gelu import gelu, gelu_accurate
from .fairseq_dropout import FairseqDropout
from .layer_norm import Fp32LayerNorm, LayerNorm
from .multihead_attention import MultiheadAttention
from .quant_noise import quant_noise
from .transformer_layer import TransformerDecoderLayer, TransformerEncoderLayer

__all__ = [
    "FairseqDropout",
    "Fp32LayerNorm",
    "LayerNorm",
    "MultiheadAttention",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
    "gelu",
    "gelu_accurate",
    "quant_noise",
]
