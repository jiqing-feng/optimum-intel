import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import intel_extension_for_pytorch as ipex
import torch
from transformers import AutoModelForCausalLM, PretrainedConfig
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager

from ..generation.modeling import BaseModelForCausalLM, jit_trace
from ..utils.import_utils import is_torch_version
from ..utils.modeling_utils import patch_decoder_attention_mask
from .modeling_base import IPEXModel


logger = logging.getLogger(__name__)

SUPPORT_MODEL_LIST_FOR_CAUSAL_LM = {
    # "llama": LlamaForCausalLM
}

SUPPORT_TASK_LIST = {
    # "text-generation": SUPPORT_MODEL_LIST_FOR_CAUSAL_LM
}


class IPEXModelForCausalLM(IPEXModel, BaseModelForCausalLM):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    forward = BaseModelForCausalLM.forward
    generate = BaseModelForCausalLM.generate
    can_generate = BaseModelForCausalLM.can_generate

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        jit: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        if is_torch_version("<", "2.1.0"):
            raise ImportError("`torch>=2.0.0` is needed to trace your model")

        task = cls.export_feature
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "use_cache": use_cache,
            "torch_dtype": torch_dtype,
        }

        model_type = None
        support_ipex_transformers = False
        for name in SUPPORT_MODEL_LIST_FOR_CAUSAL_LM.keys():
            if name in model_id:
                support_ipex_transformers = True
                model_type = name
                break

        if support_ipex_transformers and task in SUPPORT_TASK_LIST:
            # model = SUPPORT_TASK_LIST[task][model_type].from_pretrained(model_id, **model_kwargs)
            pass
        else:
            model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
            model = patch_decoder_attention_mask(model)

        model = ipex.optimize(model, dtype=torch_dtype, level="O1", auto_kernel_selection=True)

        if jit:
            traced_model = ipex_jit_trace(model, task, use_cache, support_ipex_transformers)
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)
            torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
            config.torchscript = True

            return cls._from_pretrained(
                model_id=save_dir_path,
                config=config,
                use_cache=use_cache,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                model_dtype=torch_dtype,
                **kwargs,
            )

        return cls(
            model,
            config=config,
            use_cache=use_cache,
            model_dtype=torch_dtype,
            **kwargs,
        )


def ipex_jit_trace(model, task, use_cache, support_ipex_transformers):
    if not support_ipex_transformers:
        return jit_trace(model, task, use_cache)
    else:
        # from intel_extension_for_pytorch.transformers.optimize import get_dummy_input
        # dummy_jit_inputs = get_dummy_input(task, model) # From ipex
        # model = torch.jit.trace(model, example_input_kwargs=dummy_jit_inputs)
        return model
