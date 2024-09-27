import os
from contextlib import contextmanager
from omegaconf import DictConfig
from functools import partial
from einops import rearrange, repeat
import numpy as np

import torch
from torch import nn
import torch.distributed
import torch.nn.functional as F

from sat.model.base_model import BaseModel
from sat.model.mixins import BaseMixin
from sat.ops.layernorm import LayerNorm
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from sat.mpu.utils import split_tensor_along_last_dim

from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    disabled_train,
    instantiate_from_config,
)

from sgm.modules.diffusionmodules.openaimodel import Timestep
from sgm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    timestep_embedding,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, channels, patch_size, height, width):
    x = rearrange(
        x,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=height // patch_size,
        w=width // patch_size,
        p1=patch_size,
        p2=patch_size,
    )
    return x


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        text_hidden_size=None,
        do_rearrange=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.text_hidden_size = text_hidden_size
        self.do_rearrange = do_rearrange

        self.proj = nn.Linear(in_channels * patch_size**2, hidden_size)
        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)

    def word_embedding_forward(self, input_ids, images, encoder_outputs, **kwargs):
        # images: B x C x H x W
        if self.do_rearrange:
            patches_images = rearrange(
                images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size
            )
        else:
            patches_images = images
        emb = self.proj(patches_images)

        if self.text_hidden_size is not None:
            text_emb = self.text_proj(encoder_outputs)
            emb = torch.cat([text_emb, emb], dim=1)

        return emb

    def reinit(self, parent_model=None):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
        del self.transformer.word_embeddings


def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        max_height,
        max_width,
        hidden_size,
        text_length=0,
        block_size=16,
        **kwargs,
    ):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.hidden_size = hidden_size
        self.text_length = text_length
        self.block_size = block_size
        self.image_pos_embedding = nn.Parameter(
            torch.zeros(self.max_height, self.max_width, hidden_size), requires_grad=False
        )

    def position_embedding_forward(self, position_ids, target_size, **kwargs):
        ret = []
        for h, w in target_size:
            h, w = h // self.block_size, w // self.block_size
            image_pos_embed = self.image_pos_embedding[:h, :w].reshape(h * w, -1)
            pos_embed = torch.cat(
                [
                    torch.zeros(
                        (self.text_length, self.hidden_size),
                        dtype=image_pos_embed.dtype,
                        device=image_pos_embed.device,
                    ),
                    image_pos_embed,
                ],
                dim=0,
            )
            ret.append(pos_embed[None, ...])
        return torch.cat(ret, dim=0)

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_2d_sincos_pos_embed(self.image_pos_embedding.shape[-1], self.max_height, self.max_width)
        pos_embed = pos_embed.reshape(self.max_height, self.max_width, -1)

        self.image_pos_embedding.data.copy_(torch.from_numpy(pos_embed).float())


class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        block_size,
        out_channels,
        elementwise_affine=False,
        eps=1e-6,
        do_unpatchify=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.block_size = block_size
        self.out_channels = out_channels
        self.do_unpatchify = do_unpatchify

        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, out_channels * patch_size**2)

    def final_forward(self, logits, emb, text_length, target_size=None, **kwargs):
        x = logits[:, text_length:]
        shift, scale = self.adaln(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        if self.do_unpatchify:
            target_height, target_width = target_size[0]
            assert (
                target_height % self.block_size == 0 and target_width % self.block_size == 0
            ), "target size must be divisible by block size"
            out_height, out_width = (
                target_height // self.block_size * self.patch_size,
                target_width // self.block_size * self.patch_size,
            )
            x = unpatchify(
                x, channels=self.out_channels, patch_size=self.patch_size, height=out_height, width=out_width
            )
        return x

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class AdalnAttentionMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        num_layers,
        time_embed_dim,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=False,
        eps=1e-6,
    ):
        super().__init__()

        self.adaln_modules = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorms = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, elementwise_affine=elementwise_affine, eps=eps)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorms = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, elementwise_affine=elementwise_affine, eps=eps)
                    for _ in range(num_layers)
                ]
            )

    def layer_forward(
        self,
        hidden_states,
        mask,
        text_length,
        layer_id,
        emb,
        *args,
        **kwargs,
    ):
        layer = self.transformer.layers[layer_id]
        adaln_module = self.adaln_modules[layer_id]

        (
            shift_msa_img,
            scale_msa_img,
            gate_msa_img,
            shift_mlp_img,
            scale_mlp_img,
            gate_mlp_img,
            shift_msa_txt,
            scale_msa_txt,
            gate_msa_txt,
            shift_mlp_txt,
            scale_mlp_txt,
            gate_mlp_txt,
        ) = adaln_module(emb).chunk(12, dim=1)
        gate_msa_img, gate_mlp_img, gate_msa_txt, gate_mlp_txt = (
            gate_msa_img.unsqueeze(1),
            gate_mlp_img.unsqueeze(1),
            gate_msa_txt.unsqueeze(1),
            gate_mlp_txt.unsqueeze(1),
        )

        attention_input = layer.input_layernorm(hidden_states)

        text_attention_input = modulate(attention_input[:, :text_length], shift_msa_txt, scale_msa_txt)
        image_attention_input = modulate(attention_input[:, text_length:], shift_msa_img, scale_msa_img)
        attention_input = torch.cat((text_attention_input, image_attention_input), dim=1)

        attention_output = layer.attention(attention_input, mask, layer_id=layer_id, **kwargs)
        if self.transformer.layernorm_order == "sandwich":
            attention_output = layer.third_layernorm(attention_output)

        text_hidden_states, image_hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]
        text_attention_output, image_attention_output = (
            attention_output[:, :text_length],
            attention_output[:, text_length:],
        )
        text_hidden_states = text_hidden_states + gate_msa_txt * text_attention_output
        image_hidden_states = image_hidden_states + gate_msa_img * image_attention_output
        hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)

        mlp_input = layer.post_attention_layernorm(hidden_states)

        text_mlp_input = modulate(mlp_input[:, :text_length], shift_mlp_txt, scale_mlp_txt)
        image_mlp_input = modulate(mlp_input[:, text_length:], shift_mlp_img, scale_mlp_img)
        mlp_input = torch.cat((text_mlp_input, image_mlp_input), dim=1)

        mlp_output = layer.mlp(mlp_input, layer_id=layer_id, **kwargs)
        if self.transformer.layernorm_order == "sandwich":
            mlp_output = layer.fourth_layernorm(mlp_output)

        text_hidden_states, image_hidden_states = hidden_states[:, :text_length], hidden_states[:, text_length:]
        text_mlp_output, image_mlp_output = mlp_output[:, :text_length], mlp_output[:, text_length:]
        text_hidden_states = text_hidden_states + gate_mlp_txt * text_mlp_output
        image_hidden_states = image_hidden_states + gate_mlp_img * image_mlp_output
        hidden_states = torch.cat((text_hidden_states, image_hidden_states), dim=1)

        return hidden_states

    def attention_forward(self, hidden_states, mask, layer_id, **kwargs):
        attention = self.transformer.layers[layer_id].attention

        attention_fn = attention_fn_default
        if "attention_fn" in attention.hooks:
            attention_fn = attention.hooks["attention_fn"]

        qkv = attention.query_key_value(hidden_states)
        mixed_query_layer, mixed_key_layer, mixed_value_layer = split_tensor_along_last_dim(qkv, 3)

        dropout_fn = attention.attention_dropout if self.training else None

        query_layer = attention._transpose_for_scores(mixed_query_layer)
        key_layer = attention._transpose_for_scores(mixed_key_layer)
        value_layer = attention._transpose_for_scores(mixed_value_layer)

        if self.qk_ln:
            query_layernorm = self.query_layernorms[layer_id]
            key_layernorm = self.key_layernorms[layer_id]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kwargs)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attention.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = attention.dense(context_layer)
        if self.training:
            output = attention.output_dropout(output)

        return output

    def mlp_forward(self, hidden_states, layer_id, **kwargs):
        mlp = self.transformer.layers[layer_id].mlp

        intermediate_parallel = mlp.dense_h_to_4h(hidden_states)
        intermediate_parallel = mlp.activation_func(intermediate_parallel)
        output = mlp.dense_4h_to_h(intermediate_parallel)

        if self.training:
            output = mlp.dropout(output)

        return output

    def reinit(self, parent_model=None):
        for layer in self.adaln_modules:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)


str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class DiffusionTransformer(BaseModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_size,
        patch_size,
        num_layers,
        num_attention_heads,
        text_length,
        time_embed_dim=None,
        num_classes=None,
        adm_in_channels=None,
        modules={},
        dtype="fp32",
        layernorm_order="pre",
        activation_func=None,
        elementwise_affine=False,
        parallel_output=True,
        block_size=16,
        **kwargs,
    ):
        self.model_channels = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.text_length = text_length
        self.block_size = block_size
        self.dtype = str_to_dtype[dtype]

        hidden_size_head = hidden_size // num_attention_heads

        if activation_func is None:
            approx_gelu = nn.GELU(approximate="tanh")
            activation_func = approx_gelu

        transformer_args = {
            "vocab_size": 1,
            "max_sequence_length": 64,
            "skip_init": False,
            "model_parallel_size": 1,
            "is_decoder": False,
            "layernorm_order": layernorm_order,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "parallel_output": parallel_output,
        }
        transformer_args = DictConfig(transformer_args)
        super().__init__(
            args=transformer_args,
            transformer=None,
            layernorm=partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6),
            activation_func=activation_func,
            **kwargs,
        )

        model_channels = hidden_size
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert self.adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(self.adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        pos_embed_config = modules.get(
            "pos_embed_config",
            {
                "target": "sgm.modules.diffusionmodules.dit.PositionEmbeddingMixin",
            },
        )
        pos_embed_extra_kwargs = {
            "hidden_size": hidden_size,
            "text_length": text_length,
        }
        pos_embed_mixin = instantiate_from_config(pos_embed_config, **pos_embed_extra_kwargs)
        self.add_mixin("pos_embed", pos_embed_mixin, reinit=True)

        patch_embed_config = modules.get(
            "patch_embed_config",
            {
                "target": "sgm.modules.diffusionmodules.dit.ImagePatchEmbeddingMixin",
            },
        )
        patch_embed_extra_kwargs = {
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "patch_size": patch_size,
        }
        patch_embed_mixin = instantiate_from_config(patch_embed_config, **patch_embed_extra_kwargs)
        self.add_mixin("patch_embed", patch_embed_mixin, reinit=True)

        attention_config = modules.get(
            "attention_config",
            {
                "target": "sgm.modules.diffusionmodules.dit.AdalnAttentionMixin",
            },
        )
        attention_extra_kwargs = {
            "hidden_size": hidden_size,
            "hidden_size_head": hidden_size_head,
            "num_layers": num_layers,
            "time_embed_dim": time_embed_dim,
            "elementwise_affine": elementwise_affine,
        }
        attention_mixin = instantiate_from_config(attention_config, **attention_extra_kwargs)
        self.add_mixin("adaln", attention_mixin, reinit=True)

        final_layer_config = modules.get(
            "final_layer_config",
            {
                "target": "sgm.modules.diffusionmodules.dit.FinalLayerMixin",
            },
        )
        final_layer_extra_kwargs = {
            "hidden_size": hidden_size,
            "time_embed_dim": time_embed_dim,
            "patch_size": patch_size,
            "block_size": block_size,
            "out_channels": out_channels,
            "elementwise_affine": elementwise_affine,
        }
        final_layer_mixin = instantiate_from_config(final_layer_config, **final_layer_extra_kwargs)
        self.add_mixin("final_layer", final_layer_mixin, reinit=True)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        x = x.to(self.dtype)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        input_ids = position_ids = attention_mask = torch.ones((1, 1)).to(x.dtype)

        output = super().forward(
            images=x,
            emb=emb,
            encoder_outputs=context,
            text_length=self.text_length,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )[0]

        return output
