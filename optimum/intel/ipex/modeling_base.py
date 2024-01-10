import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModel,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import WEIGHTS_NAME

from optimum.modeling_base import OptimizedModel


logger = logging.getLogger(__name__)


class IPEXModel(OptimizedModel):
    auto_model_class = AutoModel
    export_feature = "feature-extraction"
    base_model_prefix = "ipex_model"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super(IPEXModel, self).__init__(
            model=model, config=config, model_save_dir=model_save_dir, use_cache=use_cache, **kwargs
        )
        self.model.to(self._device)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        use_cache: bool = True,
        jit: bool = True,
        **kwargs,
    ):
        # Load the model from local directory
        if os.path.isdir(model_id):
            model_cache_path = os.path.join(model_id, file_name)
            model_save_dir = model_id
        # Download the model from the hub
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            model_save_dir = Path(model_cache_path).parent

        if getattr(config, "torchscript", False):
            model = torch.jit.load(model_cache_path)
            torch.jit.freeze(model.eval())
        else:
            model_class = _get_model_class(config, cls.auto_model_class._model_mapping)
            model = model_class.from_pretrained(model_save_dir)

        return cls(
            model,
            config=config,
            model_save_dir=model_save_dir,
            use_cache=use_cache,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        if getattr(self.config, "torchscript", False):
            torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))
        else:
            torch.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def can_generate(self):
        return isinstance(self.model, GenerationMixin)

    def generate(self, *args, **kwargs):
        if not self.can_generate():
            raise TypeError(
                f"The current model class {self.model.__class__} is not compatible with `.generate()`, as it doesn't have a language model head."
            )
        return self.model.generate(*args, **kwargs)
