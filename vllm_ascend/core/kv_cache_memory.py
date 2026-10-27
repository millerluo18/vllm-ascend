# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from vllm import envs
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


def uses_minimax_pp_group_compensation(config: VllmConfig) -> bool:
    """Select the device-resident MiniMax V2 PP allocation path."""
    model = config.model_config
    parallel = config.parallel_config
    return (
        model is not None
        and model.hf_text_config.model_type == "minimax_m3_text"
        and envs.VLLM_USE_V2_MODEL_RUNNER
        and parallel.pipeline_parallel_size > 1
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
        and config.cache_config.num_gpu_blocks_override is None
        and not (config.additional_config or {}).get("enable_kvpp", False)
        and config.kv_transfer_config is None
        and config.speculative_config is None
        and config.attention_config.hisparse_config is None
        and not (config.additional_config or {}).get("sparse_kv_offload_config", {}).get("enabled", False)
    )


def minimax_pp_bytes_per_block(groups: list[KVCacheGroupSpec]) -> int:
    """Count the private per-layer pages the V2 allocator materializes."""
    total = 0
    seen: set[str] = set()
    for group in groups:
        if group.host_resident:
            raise ValueError("MiniMax PP physical budgeting requires device-resident caches")
        for name in group.layer_names:
            if name in seen:
                raise ValueError(f"Duplicate MiniMax KV cache layer: {name}")
            seen.add(name)
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if type(spec) is not FullAttentionSpec and not isinstance(spec, AscendSFAIndexerCacheSpec):
                raise ValueError(f"Unsupported MiniMax PP KV cache spec: {type(spec).__name__}")
            total += spec.page_size_bytes
    return total
