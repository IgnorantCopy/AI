import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Callable, OrderedDict, Optional, Tuple


class PatchDropout(nn.Module):
    def __init__(self, prob: float = 0.5, exclude_cls_token: bool = True):
        super().__init__()
        assert 0. <= prob < 1., f"{prob} not in [0, 1)"
        self.prob = prob
        self.exclude_cls_token = exclude_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.prob == 0.:
            return x
        if self.exclude_cls_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])     # tell torchscript that cls_tokens is a tensor

        batch_size, seq_len, _ = x.shape
        batch_indices = torch.arange(batch_size).unsqueeze(-1)
        num_patches_keep = max(1, int(seq_len * (1 - self.prob)))
        rand = torch.randn((batch_size, seq_len))
        patch_keep_indices = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_keep_indices]
        if self.exclude_cls_token:
            x = torch.cat((cls_tokens, x), dim=1)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.scale = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.scale) if self.inplace else x * self.scale


class ResidualAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, fc_ratio: float = 4.0, ls_init_value: float = None,
                 activation: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
                 is_cross_attn: bool = False, batch_first: bool = True):
        super().__init__()
        self.ln1 = norm_layer(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=batch_first)
        self.ls1 = LayerScale(dim, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attn:
            self.ln_cross = norm_layer(dim)
        self.ln2 = norm_layer(dim)
        fc_dim = int(fc_ratio * dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("fc_in", nn.Linear(dim, fc_dim)),
            ("gelu", activation()),
            ("fc_out", nn.Linear(fc_dim, dim))
        ]))
        self.ls2 = LayerScale(dim, ls_init_value) if ls_init_value is not None else nn.Identity()
    
    def get_weight_type(self) -> torch.dtype:
        if hasattr(self.mlp.fc_in, "int8_original_dtype"):
            return self.mlp.fc_in.int8_original_dtype
        return self.mlp.fc_in.weight.dtype
    
    def _attention(self,
                   q: torch.Tensor,
                   k: Optional[torch.Tensor] = None,
                   v: Optional[torch.Tensor] = None,
                   attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = q if k is None else k
        v = q if v is None else v
        attn_mask = attn_mask.to(q.dtype) if attn_mask is not None else None
        return self.attention(q, k, v, attn_mask=attn_mask, need_weights=False)[0]

    def forward(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor] = None,
            v: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        k = self.ln_cross(k) if hasattr(self, "ln_cross") and k is not None else None
        v = self.ln_cross(v) if hasattr(self, "ln_cross") and v is not None else None
        x = q + self.ls1(self._attention(self.ln1(q), k, v, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.ln2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, fc_ratio: float,
                 activation: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
                 ls_init_value: float = None, batch_first: bool = True):
        """
        Transformer Block
        :param dim: base dimension of the network
        :param layers: number of layers
        :param heads: number of heads
        :param fc_ratio: the factor of the hidden dimension of the MLP compared to the input dimension
        :param activation: activation function
        :param norm_layer: normalization layer
        :param ls_init_value: initialization value for the layer scale
        :param batch_first: whether the input tensor has the batch dimension first or not
        """
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.heads = heads
        self.batch_first = batch_first

        self.res_blocks = nn.ModuleList([
            ResidualAttentionBlock(
                dim, heads,
                fc_ratio=fc_ratio,
                ls_init_value=ls_init_value,
                activation=activation,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])
    
    def get_cast_type(self) -> torch.dtype:
        return self.res_blocks[0].get_weight_type()
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()
        for block in self.res_blocks:
            x = block(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim: int, context_dim: int, heads: int,
                 n_queries: int = 256, norm_layer: Callable = nn.LayerNorm,):
        """
        Attention Pooling
        :param dim: base dimension of the network
        :param context_dim: dimension of the context
        :param heads: number of heads
        :param n_queries: number of queries
        :param norm_layer: normalization layer
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, dim))
        self.attention = nn.MultiheadAttention(dim, heads, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(dim)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        q = self.ln_q(self.query).unsqueeze(0).expand(batch_size, -1, -1)
        k = self.ln_k(x)
        out = self.attention(q, k, k, need_weights=False)
        return out[0]


class ViT(nn.Module):
    def __init__(self, image_size: int, patch_size: int, dim: int, layers: int, heads: int,
                 fc_ratio: float, scale: float = None, output_dim: int = 512, patch_dropout: float = 0.,
                 activation: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm, ln_pre: bool = True,
                 ls_init_value: float = None, attn_pool: str = None, attn_pool_heads: int = 8,
                 attn_pool_queries: int = 256, global_pool: str = None, return_tokens: bool = False):
        """
        Vision Transformer Encoder
        :param image_size: size of the input image (assuming square image)
        :param patch_size: size of the patch (assuming square patch)
        :param dim: base dim of the network
        :param layers: number of layers
        :param heads: number of heads
        :param fc_ratio: the factor of the hidden dimension of the MLP compared to the input dimension
        :param scale: scale factor for the positional encoding
        :param output_dim: output dimension of the network
        :param patch_dropout: 
        :param activation: activation function
        :param norm_layer: normalization layer
        :param ln_pre: whether to normalize before the transformer block
        :param ls_init_value: initialization value for the layer scale
        :param attn_pool: type of attention pooling to use (if any)
        :param attn_pool_heads: number of heads for attention pooling
        :param attn_pool_queries: number of queries for attention pooling
        :param global_pool: type of global pooling to use (if any)
        :param return_tokens: whether to return the tokens as well as the pooled representation
        """
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        image_height, image_width = self.image_size
        patch_height, patch_width = self.patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.output_dim = output_dim
        self.scale = dim ** -0.5 if scale is None else scale

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_embedding = nn.Parameter(self.scale * torch.randn(dim))
        self.pos_embedding = nn.Parameter(self.scale * torch.randn(self.num_patches + 1, dim))

        self.patch_dropout = PatchDropout(patch_dropout)
        self.ln_pre = norm_layer(dim) if ln_pre else nn.Identity()
        self.transformer = Transformer(
            dim, layers, heads, fc_ratio,
            activation, norm_layer,
            ls_init_value=ls_init_value,
            batch_first=True
        )

        self.global_pool_type = global_pool
        self.return_tokens = return_tokens
        self.attn_pool_type = attn_pool
        if attn_pool:
            pool_dim = output_dim
            self.attn_pool = AttentionPool(pool_dim, dim, attn_pool_heads, attn_pool_queries, norm_layer)
            if attn_pool in ["parallel", "cascade"]:
                self.attn_pool_contrastive = AttentionPool(pool_dim, dim, attn_pool_heads, 1, norm_layer)
            else:
                self.attn_pool_contrastive = None
        else:
            pool_dim = dim
            self.attn_pool = None
            self.attn_pool_contrastive = None

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(self.scale * torch.randn((pool_dim, output_dim)))

    def _embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = rearrange(x, "b c g_h g_w -> b (g_h g_w) c")
        cls_token = repeat(self.cls_embedding, "d -> b 1 d", b=x.shape[0])
        x = torch.cat([cls_token.to(x.dtype), x], dim=1)
        x = x + self.pos_embedding.to(x.dtype)  # [batch_size, num_patches + 1, dim]
        x = self.ln_pre(self.patch_dropout(x))
        return x

    def _global_pooling(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_pool_type == "mean":
            x_pool, tokens = x[:, 1:].mean(1), x[:, 1:]
        elif self.global_pool_type == "token":
            x_pool, tokens = x[:, 0], x[:, 1:]
        else:
            x_pool, tokens = x
        return x_pool, tokens

    def _pooling(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                x = self.ln_post(x)
                tokens = self.attn_pool(x)
                if self.attn_pool_type == "parallel":
                    x_pool = self.attn_pool_contrastive(x)
                elif self.attn_pool_type == "cascade":
                    x_pool = self.attn_pool(tokens)
                else:
                    raise ValueError(f"Unknown attention pooling type: {self.attn_pool_type}")
            else:
                x = self.ln_post(self.attn_pool(x))
                x_pool, tokens = self._global_pooling(x)
        else:
            x = self.ln_post(x)
            x_pool, tokens = self._global_pooling(x)

        return x_pool, tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self._embedding(x)
        x = self.transformer(x)
        x_pool, tokens = self._pooling(x)

        if self.proj is not None:
            x_pool = x_pool @ self.proj

        if self.return_tokens:
            return x_pool, tokens
        return x_pool


class TextTransformer(nn.Module):
    def __init__(self, seq_len: int = 77, vocab_size: int = 49408, dim: int = 512, heads: int = 8, layers: int = 12,
                 fc_ratio: float = 4.0, output_dim: Optional[int] = 512, ls_init_value: float = None,
                 pad_id: int = 0, eos_id: int = 2, pool_type: str = "argmax", proj_type: str = "linear",
                 proj_bias: bool = False, activation: Callable = nn.GELU, norm_layer: Callable = nn.LayerNorm,
                 embed_cls: bool = False, casual_mask: bool = True, use_pad_mask: bool = False,
                 return_tokens: bool = False):
        """
        Text Transformer Encoder
        :param seq_len: sequence length
        :param vocab_size: vocabulary size
        :param dim: embedding dimension
        :param heads: number of heads
        :param layers: number of layers
        :param fc_ratio: the factor of the hidden dimension of the MLP compared to the input dimension
        :param output_dim: output dimension of the network
        :param ls_init_value: initialization value for the layer scale
        :param pad_id: padding token id
        :param eos_id: end-of-sentence token id
        :param pool_type: method to pool the output of the transformer block
        :param proj_type: projection type for the output of the transformer block
        :param proj_bias: whether to use bias in the projection layer
        :param activation: activation function
        :param norm_layer: normalization layer
        :param embed_cls: whether to use CLS embedding
        :param casual_mask: whether to use casual mask
        :param use_pad_mask: whether to use padding mask
        :param return_tokens: whether to return tokens
        """
        super().__init__()
        assert pool_type in ["first", "last", "argmax", "eos", "none", None], f"Unknown pooling type: {pool_type}"

        self.num_pos = self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.fc_ratio = fc_ratio
        self.output_dim = output_dim
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.pool_type = pool_type
        self.proj_type = proj_type
        self.use_pad_mask = use_pad_mask and not casual_mask
        self.return_tokens = return_tokens

        self.token_embedding = nn.Embedding(vocab_size, dim)
        if embed_cls:
            self.cls_embedding = nn.Parameter(torch.empty(dim))
            self.num_pos += 1
        else:
            self.cls_embedding = None
        self.pos_embedding = nn.Parameter(torch.empty(self.num_pos, dim))
        self.transformer = Transformer(
            dim, layers, heads, fc_ratio,
            activation, norm_layer,
            ls_init_value=ls_init_value,
            batch_first=True
        )
        self.ln_post = norm_layer(dim)

        if casual_mask:
            self.register_buffer("attn_mask", self._build_casual_mask(), persistent=False)
        else:
            self.attn_mask = None

        if not output_dim or proj_type == "none":
            self.text_proj = None
        elif proj_bias:
            self.text_proj = nn.Linear(dim, output_dim, bias=True)
        else:
            self.text_proj = nn.Parameter(torch.empty(dim, output_dim))

        self._init_params()

    def _build_casual_mask(self):
        mask = torch.empty((self.num_pos, self.num_pos))
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _build_additive_mask(self, text: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        :param text: original text ids without cls token, in shape [batch_size, seq_len]
        :param dtype:
        :return: an additive mask (-inf) in shape [batch_size * heads, seq_len, seq_len]
        """
        batch_size = text.shape[0]
        seq_len = text.shape[1] + 1 if self.cls_embedding is None else text.shape[1]

        valid = text != self.pad_id

        if self.cls_embedding is not None:
            cls_valid = valid.new_ones((batch_size, 1))
            valid = torch.cat([valid, cls_valid])

        # broadcast over query dimension
        key_mask = repeat(valid, "b s -> b q s", q=seq_len)
        additive = torch.zeros_like(key_mask, dtype=dtype)
        additive.masked_fill_(~key_mask, float("-inf"))
        additive = additive.repeat_interleave(self.heads, 0)
        return additive

    def _init_params(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.01)
        if self.cls_embedding is not None:
            nn.init.normal_(self.cls_embedding, std=0.01)

        attn_std = self.transformer.dim ** -0.5
        proj_std = attn_std * ((2 * self.transformer.layers) ** -0.5)
        fc_std = (2 * self.transformer.dim) ** -0.5
        for block in self.transformer.res_blocks:
            nn.init.normal_(block.attention.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attention.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.fc_in.weight, std=fc_std)
            nn.init.normal_(block.mlp.fc_out.weight, std=proj_std)

        if self.text_proj is not None:
            if isinstance(self.text_proj, nn.Linear):
                nn.init.normal_(self.text_proj.weight, std=attn_std)
                nn.init.zeros_(self.text_proj.bias)
            else:
                nn.init.normal_(self.text_proj, std=attn_std)

    def _embedding(self, text: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cast_type = self.transformer.get_cast_type()
        batch_size, seq_len = text.shape
        
        x = self.token_embedding(text).to(cast_type)

        if self.cls_embedding is not None:
            cls_token = repeat(self.cls_embedding, "d -> b 1 d", b=batch_size)
            x = torch.cat([cls_token.to(x.dtype), x], dim=1)
            seq_len += 1

        attn_mask = self.attn_mask

        if self.use_pad_mask or self.cls_embedding is not None:
            add_mask = self._build_additive_mask(text, x.dtype)
            if attn_mask is None:
                attn_mask = add_mask
            else:
                attn_mask = attn_mask[:seq_len, :seq_len].unsqueeze(0) + add_mask

        x = x + self.pos_embedding[:seq_len].to(cast_type)
        return x, attn_mask

    def _global_pooling(self, x: torch.Tensor, text: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pool_type == "first":
            x_pool = x[:, 0]
        elif self.pool_type == "last":
            x_pool = x[:, -1]
        elif self.pool_type == "argmax":
            assert text is not None
            x_pool = x[torch.arange(x.shape[0], device=x.device), text.argmax(-1)]
        elif self.pool_type == "eos":
            assert text is not None
            index = (text == self.eos_id).int().argmax(-1)
            x_pool = x[torch.arange(x.shape[0], device=x.device), index]
        else:
            x_pool = x

        return x_pool

    def forward(self, text: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, attn_mask = self._embedding(text)
        x = self.transformer(x, attn_mask=self.attn_mask)

        if self.cls_embedding is not None:
            self.pool_type = "last"
            x_pool = self._global_pooling(x, text)
            x_pool = self.ln_post(x_pool)
            tokens = x[:, :-1]
        else:
            x = self.ln_post(x)
            x_pool = self._global_pooling(x, text)
            tokens = x

        if self.text_proj is not None:
            if isinstance(self.text_proj, nn.Linear):
                x_pool = self.text_proj(x_pool)
            else:
                x_pool = x_pool @ self.text_proj

        if self.return_tokens:
            return x_pool, tokens
        return x_pool