from dataclasses import dataclass
from typing import Union, Tuple, Optional, List


@dataclass
class VisualConfig:
    image_size: Union[Tuple[int, int], int] = 224
    patch_size: int = 16
    dim: int = 768
    layers: Union[List[int], int] = 12
    head_width: int = 64
    fc_ratio: float = 4.0
    scale_factor: float = None

    patch_dropout: float = 0.
    ln_pre: bool = True
    ls_init_value: Optional[float] = None
    attn_pool: str = None
    attn_pool_heads: int = 8
    attn_pool_queries: int = 256
    global_pool: str = "token"
    return_tokens: bool = False


@dataclass
class TextConfig:
    seq_len: int = 77
    vocab_size: int = 49408
    dim: int = 512
    heads: int = 8
    layers: int = 12
    fc_ratio: float = 4.0

    ls_init_value: Optional[float] = None
    pad_id: int = 0
    eos_id: int = 2
    pool_type: str = "argmax"
    proj_type: str = "linear"
    proj_bias: bool = False
    embed_cls: bool = False
    casual_mask: bool = True
    use_pad_mask: bool = False
    return_tokens: bool = False


