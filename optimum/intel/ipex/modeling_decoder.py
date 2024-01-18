import logging
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union, Tuple

from transformers import AutoModelForCausalLM, PretrainedConfig

from ..generation.modeling import BaseModelForCausalLM, jit_trace
from .modeling_base import IPEXModel

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.transformers.optimize import get_dummy_input
from intel_extension_for_pytorch.cpu._auto_kernel_selection import _enable_tpp, _using_tpp, _disable_tpp

logger = logging.getLogger(__name__)


class IPEXModelForCausalLM(IPEXModel, BaseModelForCausalLM):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    forward = BaseModelForCausalLM.forward
    generate = BaseModelForCausalLM.generate
    can_generate = BaseModelForCausalLM.can_generate

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        IPEXModel.__init__(self, model, config)
        BaseModelForCausalLM.__init__(self, model, config, model_save_dir, use_cache, **kwargs)

    @classmethod
    def apply_jit_optimize(cls, model, task, use_cache, support_ipex_transformers):
        if not support_ipex_transformers:
            return jit_trace(model, task, use_cache)
        else:
            sample_inputs = get_dummy_input(model, return_dict=True)
            with torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True if model.dtype is torch.bfloat16 else False
            ):
                _enable_tpp()
                model = ipex.optimize(model.eval(), dtype=model.dtype, inplace=True)
                trace_model = torch.jit.trace(
                    model,
                    example_kwarg_inputs=sample_inputs,
                    strict=False,
                    check_trace=False,
                )
                trace_model = torch.jit.freeze(trace_model)
                trace_model(**sample_inputs)
                trace_model(**sample_inputs)

            return trace_model

    def prepare_past_key_values(self, input_ids):
        num_layers = self.normalized_config.num_layers
        beam_idx_tmp = torch.zeros(
            (2048, input_ids.shape[0]), dtype=torch.long
        ).contiguous()
        past_key_values = tuple(
            [
                (
                    torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    torch.zeros([1, 1, 1, 1]).contiguous(),
                    beam_idx_tmp,
                )
                for i in range(num_layers)
            ]
        )
        return past_key_values
