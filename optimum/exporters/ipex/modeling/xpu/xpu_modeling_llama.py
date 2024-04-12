from typing import Tuple
import torch
import torch.nn as nn
from typing import Optional


import intel_extension_for_pytorch
from intel_extension_for_pytorch.transformers.models.xpu.optimize_transformers.modules.llama import NewIPEXLLAMABlock

from ..modeling_llama import _IPEXLlamaDecoderLayer, _IPEXLlamaAttention, _IPEXLlamaMLP

class _IPEXLlamaAttentionXPU(_IPEXLlamaAttention):
    def __init__(self, module, config, distributed=False, optimized_module=None) -> None:
        super().__init__(module, config, distributed)
        self.attn_impl = None
        if optimized_module is not None:
            self.attn_impl = optimized_module

    def preprocess_for_optimize(self, hidden_states, layer_past, **kwargs):
        if self.attn_impl is not None:
            return self.attn_impl.preprocess_for_optimize(hidden_states, layer_past, **kwargs)
        else:
            return super().preprocess_for_optimize(hidden_states, layer_past, **kwargs)

    def qkv_gemm(self, hidden_states, **kwargs):
        if self.attn_impl is not None:
            return self.attn_impl.qkv_gemm(hidden_states, **kwargs)
        else:
            return super().qkv_gemm(hidden_states, **kwargs)

    def rope(self, query, key, value, position_ids, layer_past, **kwargs):
        if self.attn_impl is not None:
            return self.attn_impl.rope(query, key, value, position_ids, layer_past, **kwargs)
        else:
            return super().rope(query, key, value, position_ids, layer_past, **kwargs)

    def get_present(self, query, key, value, use_cache, **kwargs):
        if self.attn_impl is not None:
            return self.attn_impl.get_present(query, key, value, use_cache, kwargs)
        else:
            return super().get_present(query, key, value, use_cache, **kwargs)

    def sdpa(self, query, key, value, attention_mask, past_key_value, **kwargs):
        if self.attn_impl is not None:
            return self.attn_impl.sdpa(query, key, value, attention_mask, past_key_value, **kwargs)
        else:
            return super().sdpa(query, key, value, attention_mask, past_key_value, **kwargs)

    def out_proj(self, hidden_states, residual, **kwargs):
        if self.attn_impl is not None:
            return self.attn_impl.out_proj(hidden_states, residual, **kwargs)
        else:
            return super().out_proj(hidden_states, residual, **kwargs)

    def post_process_for_optimize(self):
        if self.attn_impl is not None:
            return self.attn_impl.post_process_for_optimize()
        else:
            return super().post_process_for_optimize()




class _IPEXLlamaMLPXPU(_IPEXLlamaMLP):
    def __init__(self, module, config, distributed=False, optimized_module=None) -> None:
        super().__init__(module, config, distributed)
        self.mlp_impl = None
        if optimized_module is not None:
            self.mlp_impl = optimized_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        if self.mlp_impl is not None:
            self.mlp_impl(hidden_states, residual, **kwargs)
        else:
            super().forward(hidden_states, residual, **kwargs)




class _IPEXLlamaDecoderLayerXPU(_IPEXLlamaDecoderLayer):
    def __init__(self, module, config, distributed=False) -> None:
        super().__init__(module, config, distributed)
        self.block_impl = NewIPEXLLAMABlock(module, config)
        self.attn = _IPEXLlamaAttentionXPU(module.self_attn, config, self.block_impl.attn)
        self.mlp = _IPEXLlamaMLPXPU(module.mlp, config, self.block_impl.mlp)

    def preprocess_for_optimize(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attention: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        **kwargs
    ):
        return self.block_impl.preprocess_for_optimize(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attention,
            use_cache,
            **kwargs
        )



    def postprocess_for_optimize(self, hidden_states, output_attention, use_cache, self_attn_weight, present_key_value, **kwargs):
        return self.block_impl.postprocess_for_optimize(
            hidden_states,
            output_attention,
            use_cache,
            self_attn_weight,
            present_key_value,
            **kwargs
        )

