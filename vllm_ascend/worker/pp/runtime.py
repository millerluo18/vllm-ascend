# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-send storage for fixed-partition pipeline parallelism."""

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch

from vllm_ascend.worker.pp.buffer import PPSendBuffer
from vllm_ascend.worker.pp.config import get_pp_send_buffer_size


class PPSendRuntime:
    def __init__(self, config, model, device, device_module, pp_group, tp_group, model_runner=None):
        self.pp = pp_group
        self.tp = tp_group
        self.model = model
        self.model_runner = model_runner
        self._persistent_attributes = ()
        self.buffer = None
        size = get_pp_send_buffer_size(config)
        if size and pp_group.world_size > 1 and not pp_group.is_last_rank:
            self.buffer = PPSendBuffer(size, device_module, metadata_handles=1)
            tensor_factory = model.make_empty_intermediate_tensors
            send_factory = getattr(tensor_factory, "for_send", tensor_factory)
            template = send_factory(
                config.scheduler_config.max_num_batched_tokens,
                dtype=config.model_config.dtype,
                device=device,
            )
            self.buffer.reserve(template.tensors, allow_dynamic=self._allow_dynamic(config))
            if self.buffer.dynamic_enabled:
                self._persistent_attributes = tuple(
                    (module, name)
                    for module in model.modules()
                    for name in set(module._buffers)
                    | {name for name, value in vars(module).items() if isinstance(value, torch.Tensor)}
                )

    @staticmethod
    def _allow_dynamic(config) -> bool:
        return (
            config.model_config.enforce_eager
            and config.compilation_config.mode == 0
            and config.speculative_config is None
            and config.lora_config is None
        )

    def _reused_storages(self):
        persistent = [getattr(module, name, None) for module, name in self._persistent_attributes]
        inputs = getattr(self.model_runner, "intermediate_tensors", None)
        if inputs is not None:
            persistent.extend(inputs.tensors.values())
        return {value.untyped_storage()._cdata for value in persistent if isinstance(value, torch.Tensor)}

    def protect_outputs(self):
        return self.buffer.protect_outputs() if self.buffer is not None else nullcontext()

    def prepare(self, output) -> None:
        if output.total_num_scheduled_tokens and self.buffer is not None:
            self.buffer.acquire()

    def send(self, tensors: Mapping[str, Any], *, all_gather_group) -> None:
        assert self.buffer is not None
        self.buffer.send(
            tensors,
            lambda value: self.pp.isend_tensor_dict(value, all_gather_group=all_gather_group),
            reused_storages=self._reused_storages() if self.buffer.dynamic_enabled else (),
        )

    def cancel(self) -> None:
        if self.buffer is not None:
            self.buffer.cancel()

    def drain(self) -> None:
        if self.buffer is not None:
            self.buffer.drain()
