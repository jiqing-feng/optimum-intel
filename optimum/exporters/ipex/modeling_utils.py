#  Copyright 2024 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaModel, repeat_kv

from optimum.intel.utils.import_utils import is_ipex_version
from optimum.intel.utils.modeling_utils import _setattr_from_module

from .cache_utils import IPEXPagedCache


logger = logging.getLogger(__name__)

_IPEX_MINIMUM_VERSION_FOR_PATCHING = "2.3.0"


if is_ipex_version("<", _IPEX_MINIMUM_VERSION_FOR_PATCHING):
    logger.warning(
        f"Please upgrade the IPEX version to at least {_IPEX_MINIMUM_VERSION_FOR_PATCHING} if you want to patch the model."
    )
else:
    from intel_extension_for_pytorch.llm.functional import rms_norm, rotary_embedding
    from intel_extension_for_pytorch.llm.modules import (
        Linear2SiluMul,
        LinearAdd,
        LinearAddAdd,
        LinearGelu,
        PagedAttention,
    )


# Adapted from https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L918
def _llama_model_forward(
    self,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    input_lens = kwargs.get("attention_mask").cumsum(-1)[:, -1].int()
    setattr(kwargs.get("past_key_values"), "input_lens", input_lens)
    setattr(kwargs.get("past_key_values"), "original_attn_mask", kwargs.get("attention_mask"))
    return LlamaModel.forward(self, **kwargs)


class XPULinear2SiluMul(torch.nn.Module):
    def __init__(
        self,
        gate_proj: torch.nn.Module,
        up_proj: torch.nn.Module,
    ):
        super().__init__()
        self.gate_proj_weight = gate_proj.weight.transpose(0, 1).contiguous()
        self.up_proj_weight = up_proj.weight.transpose(0, 1).contiguous()
        self.gate_proj_bias = gate_proj.bias
        self.up_proj_bias = up_proj.bias

    def forward(
        self,
        hidden_states,
    ):
        up = torch.ops.torch_ipex.mm_silu(hidden_states, self.gate_proj_weight)
        if self.gate_proj_bias is not None:
            up += self.gate_proj_bias
        hidden_states = torch.ops.torch_ipex.mm_resmul(hidden_states, self.up_proj_weight, up)
        if self.up_proj_bias is not None:
            hidden_states += self.up_proj_bias
        return hidden_states


class XPULinearAdd(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
    ):
        super().__init__()
        self.weight = module.weight.transpose(0, 1).contiguous()
        self.bias = module.bias

    def forward(
        self,
        hidden_states,
        residual,
    ):
        token_len, _ = hidden_states.size()
        if residual is None:
            hidden_states = torch.matmul(hidden_states, self.weight)
            if self.bias is not None:
                hidden_states += self.bias
        else:
            if self.bias is not None:
                hidden_states = torch.ops.torch_ipex.mm_bias_resadd(
                    hidden_states, self.weight, self.bias, 1.0, residual, 1.0
                )
            else:
                hidden_states = torch.addmm(
                    residual.flatten(0, -2),
                    hidden_states.flatten(0, -2),
                    self.weight,
                    beta=1.0,
                )
        hidden_states = hidden_states.view(token_len, -1)
        return hidden_states


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L83
def _ipex_rms_layer_norm_forward(self, hidden_states):
    return rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _gpt2_block_forward(self, hidden_states, *args, **kwargs):
    attention_mask = kwargs.get("attention_mask", None)
    if attention_mask is not None:
        bsz, seq_len, _ = hidden_states.size()
        layer_past = kwargs.get("layer_past", None)
        past_len = layer_past[0].size(-2) if layer_past is not None else 0
        attention_mask = (1 - attention_mask / torch.finfo(attention_mask.dtype).min).squeeze(1, 2)
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (bsz, seq_len), hidden_states, past_len)
        kwargs["attention_mask"] = attention_mask

    return GPT2Block.forward(self, hidden_states, *args, **kwargs)


class _IPEXAttention(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device.type
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=self.module_device
        ).repeat_interleave(self.num_groups)

    def qkv_gemm(self, hidden_states):
        raise NotImplementedError("Need to implement in specific model class")

    def rope(self, *args, **kwargs):
        raise NotImplementedError("Need to implement in specific model class")

    def postprocess_attention_output(self, attn_output, bsz, seq_len):
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[IPEXPagedCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and kwargs.get("layer_past", None) is not None:
            past_key_value = kwargs.pop("layer_past", None)
        bsz, q_len = position_ids.size()
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.get_seq_length()
        qkv_out = self.qkv_gemm(hidden_states)
        if isinstance(qkv_out, tuple) and len(qkv_out) == 3:
            query, key, value = qkv_out[0], qkv_out[1], qkv_out[2]
            query, key = self.rope(query, key, **kwargs)
        else:
            query, key, value = self.rope(qkv_out, **kwargs)

        if past_key_value is not None:
            key_cache = key.view(-1, self.num_key_value_heads, self.head_dim)
            value_cache = value.view(-1, self.num_key_value_heads, self.head_dim)
            if q_len > 1:
                index = past_key_value.original_attn_mask.view(-1) != 0
                key_cache = key_cache[index]
                value_cache = value_cache[index]
            key_cache, value_cache = past_key_value.update(
                key_cache, value_cache, self.layer_idx, position_ids, past_key_value.input_lens
            )

        if past_len == 0:
            # # prefill, remove padding
            query = query.transpose(1, 2)
            key = repeat_kv(key.transpose(1, 2), self.num_key_value_groups)
            value = repeat_kv(value.transpose(1, 2), self.num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query.device.type == "xpu" and causal_mask is not None:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
        else:
            # decode
            query = query.view(-1, self.num_heads, self.head_dim)
            attn_output = torch.empty_like(query)
            PagedAttention.single_query_cached_kv_attention(
                attn_output,
                query,
                key_cache,
                value_cache,
                self.kv_head_mapping,
                1.0 / math.sqrt(self.head_dim),
                past_key_value.block_tables,
                past_key_value.input_lens,
                past_key_value.block_size,
                past_key_value.input_lens.max(),
                None,
            )
            attn_output = attn_output.reshape(-1, 1, attn_output.shape[-2] * attn_output.shape[-1])

        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights


class _IPEXLlamaAttention(_IPEXAttention):
    def __init__(self, module, config) -> None:
        super().__init__(module, config)
        concat_weight = torch.concat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]).contiguous()
        bias_list = [bias for bias in [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias] if bias]
        use_bias = bias_list != []
        self.concat_qkv = nn.Linear(concat_weight.shape[1], concat_weight.shape[0], bias=use_bias)
        self.concat_qkv.weight = nn.Parameter(concat_weight)
        if use_bias:
            concat_bias = torch.concat(bias_list, 0).contiguous()
            self.concat_linear.bias = nn.Parameter(concat_bias)
        self.q_slice = self.q_proj.out_features
        self.k_slice = self.q_slice + self.k_proj.out_features
        self.v_slice = self.k_slice + self.v_proj.out_features
        del self.__dict__["_modules"]["q_proj"]
        del self.__dict__["_modules"]["k_proj"]
        del self.__dict__["_modules"]["v_proj"]
        if self.module_device == "cpu":
            if module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mha_linear_add = LinearAdd(module.o_proj)
                del self.__dict__["_modules"]["o_proj"]
        elif self.module_device == "xpu":
            if module.o_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mha_linear_add = XPULinearAdd(module.o_proj)
                del self.__dict__["_modules"]["o_proj"]

    def qkv_gemm(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        qkv_out = self.concat_qkv(hidden_states)
        query = qkv_out[:, :, : self.q_slice].view(bsz, seq_len, self.num_heads, self.head_dim)
        key = qkv_out[:, :, self.q_slice : self.k_slice].view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value = qkv_out[:, :, self.k_slice :].view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        return query, key, value

    def rope(self, query, key, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        rotary_embedding(query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True)
        return query, key


class _IPEXFalconAttention(_IPEXAttention):
    def qkv_gemm(self, hidden_states):
        return self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

    def rope(self, fused_qkv, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)
        (query, key, value) = self._split_heads(fused_qkv)
        rotary_embedding(query, key, position_embeddings[1], position_embeddings[0], query.size(-1), True)
        return query, key, value


class _IPEXGPT2Attention(_IPEXAttention):
    def __init__(self, module, config) -> None:
        super().__init__(module, config)

    def _split_heads_ipex(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        return tensor.view(new_shape)  # (batch, seq_length, head, head_features)

    def qkv_gemm(self, hidden_states):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads_ipex(query, self.num_heads, self.head_dim)
        key = self._split_heads_ipex(key, self.num_heads, self.head_dim)
        value = self._split_heads_ipex(value, self.num_heads, self.head_dim)
        return query, key, value

    def rope(self, query, key, *args, **kwargs):
        return query, key

    def postprocess_attention_output(self, attn_output, bsz, seq_len):
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


# Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L186
class _IPEXLlamaMLP(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        self.module_device = next(module.parameters()).device.type
        if self.module_device == "cpu":
            # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = LinearAdd(module.down_proj)
                del self.__dict__["_modules"]["down_proj"]
            self.linear_silu_mul = Linear2SiluMul(module.gate_proj, module.up_proj)
            del self.__dict__["_modules"]["gate_proj"]
            del self.__dict__["_modules"]["up_proj"]
        elif self.module_device == "xpu":
            # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
            if module.down_proj.__class__.__name__ not in ["LinearAllreduce"]:
                self.mlp_linear_add = XPULinearAdd(module.down_proj)
                del self.__dict__["_modules"]["down_proj"]
            self.linear_silu_mul = XPULinear2SiluMul(module.gate_proj, module.up_proj)
            del self.__dict__["_modules"]["gate_proj"]
            del self.__dict__["_modules"]["up_proj"]

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None, **kwargs):
        if hasattr(self, "linear_silu_mul"):
            mlp_gate = self.linear_silu_mul(hidden_states)
            if hasattr(self, "mlp_linear_add"):
                hidden_states = self.mlp_linear_add(mlp_gate, residual)
            else:
                hidden_states = self.down_proj(mlp_gate)
                hidden_states = residual + hidden_states
        else:
            hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
            hidden_states = residual + hidden_states

        return hidden_states


class _IPEXFalconMLP(nn.Module):
    def __init__(self, module, config) -> None:
        super().__init__()
        _setattr_from_module(self, module)
        self.config = config
        # LinearAllreduce and LinearLayer cannot use fused op LinearAdd
        self.linear_gelu = LinearGelu(module.dense_h_to_4h)
        del self.__dict__["_modules"]["dense_h_to_4h"]
        if module.dense_4h_to_h.__class__.__name__ not in ["LinearAllreduce"]:
            self.linear_add_add = LinearAddAdd(module.dense_4h_to_h)
            del self.__dict__["_modules"]["dense_4h_to_h"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_output: torch.Tensor = None,
        residual: torch.Tensor = None,
        **kwargs,
    ):
        mlp_hidden_states = self.linear_gelu(hidden_states)
        if hasattr(self, "linear_add_add"):
            output = self.linear_add_add(mlp_hidden_states, attention_output, residual)
        else:
            mlp_output = self.mlp.dense_4h_to_h(mlp_hidden_states)
            output = mlp_output + attention_output + residual

        return output


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/llama/modeling_llama.py#L694
class _IPEXLlamaDecoderLayer(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attn = _IPEXLlamaAttention(module.self_attn, config)
        self.mlp = _IPEXLlamaMLP(module.mlp, config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Please see the original model's forward to check the parameter
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, present, attn_weights = self.self_attn(hidden_states=hidden_states, **kwargs)

        if hasattr(self.self_attn, "mha_linear_add"):
            hidden_states = self.self_attn.mha_linear_add(hidden_states, residual)
        else:
            hidden_states = self.self_attn.o_proj(hidden_states)
            hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, residual, **kwargs)

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (attn_weights,)
        if kwargs.get("use_cache", False):
            outputs += (present,)

        return outputs


class _IPEXFalconDecoderLayer(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.self_attention = _IPEXFalconAttention(module.self_attention, config)
        self.mlp = _IPEXFalconMLP(module.mlp, config)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Please see the original model's forward to check the parameter
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output, present, attn_weights = self.self_attention(hidden_states, **kwargs)
        attn_output = self.self_attention.dense(attn_output)
        hidden_states = self.mlp(hidden_states, attn_output, residual)

        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False):
            outputs += (attn_weights,)
        if kwargs.get("use_cache", False):
            outputs += (present,)

        return outputs


# Adapted from https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/bert/modeling_bert.py#L524
class _IPEXIntermediate(nn.Module):
    def __init__(self, module, config):
        super().__init__()
        _setattr_from_module(self, module)
        self.linear_gelu = LinearGelu(module.dense)
        del self.__dict__["_modules"]["dense"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_gelu(hidden_states)
        return hidden_states
