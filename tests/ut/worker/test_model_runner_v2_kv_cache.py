# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.core.kv_cache_memory import minimax_pp_bytes_per_block
from vllm_ascend.patch.platform import patch_kv_cache_utils as planner_patch
from vllm_ascend.patch.worker.patch_v2 import patch_model_runner  # noqa: F401


@pytest.mark.parametrize("is_profiling", [False, True])
def test_initialize_preserves_connector_containers_and_flattens_runner_cache(is_profiling):
    k, v, conv, ssm, single = [torch.empty(3, 4) for _ in range(5)]
    other = torch.empty(3, 4, device="meta")
    caches = {"attention": (k, None, v), "mamba": [conv, ssm], "single": single, "other": other}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    # MagicMock makes aux_output_config.enabled truthy by default; disable it so
    # the test exercises the KV-connector path only (main-only branch).
    runner.vllm_config.aux_output_config.enabled = False
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])
    registration_order = []
    runner._register_sparse_kv_caches.side_effect = lambda _: registration_order.append("offload")

    def register_connector(*_args):
        registration_order.append("connector")

    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", return_value=caches),
        patch.object(upstream, "get_kv_connector", side_effect=register_connector) as connector,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=is_profiling)

    assert [id(tensor) for tensor in runner.kv_caches] == [id(tensor) for tensor in (k, v, conv, ssm, single)]
    runner._register_sparse_kv_caches.assert_called_once_with(caches)
    if is_profiling:
        assert registration_order == ["offload"]
        connector.assert_not_called()
        assert runner.kv_connector is upstream.NO_OP_KV_CONNECTOR
    else:
        assert registration_order == ["offload", "connector"]
        assert connector.call_args.args[1] is caches
        assert type(caches["attention"]) is tuple
        assert type(caches["mamba"]) is list
        assert caches["attention"][2] is v
        assert caches["mamba"][1] is ssm
        assert caches["other"] is other


def test_initialize_builds_aux_output_connector_when_enabled():
    """The vendored initializer keeps upstream's AuxOutput connector wiring."""
    caches = {"attention": torch.empty(3, 4)}
    runner = MagicMock()
    runner.device = torch.device("cpu")
    runner.is_encoder_decoder = False
    runner.speculator = None
    runner.vocab_size = 16
    runner.max_model_len = 128
    runner.cache_config.kv_sharing_fast_prefill = False
    # Mirror of the case above: exercise the AuxOutput branch only.
    runner.vllm_config.aux_output_config.enabled = True
    runner.jit_warmup_registry.activate.side_effect = nullcontext
    config = KVCacheConfig(num_blocks=3, kv_cache_tensors=[], kv_cache_groups=[])

    with (
        patch.object(upstream, "init_attn_backend", return_value=([], MagicMock(), [])),
        patch.object(upstream, "maybe_create_adaptive_verification_manager", return_value=None),
        patch.object(upstream, "BlockTables"),
        patch.object(upstream.pcp, "maybe_build_pcp_manager", return_value=None),
        patch.object(upstream, "maybe_build_ubatch_runner", return_value=None),
        patch.object(upstream, "initialize_mamba_ssu_backend"),
        patch.object(upstream, "has_compiled_submodule", return_value=False),
        patch.object(upstream, "ModelCudaGraphManager"),
        patch.object(upstream, "check_attention_cp_compatibility"),
        patch.object(upstream, "init_kv_cache", return_value=caches),
        patch.object(upstream, "get_kv_connector"),
        patch.object(upstream, "get_aux_output_connector", return_value="aux") as get_aux,
    ):
        upstream.GPUModelRunner.initialize_kv_cache(runner, config, is_profiling=False)

    # The initializer deep-copies the config, so assert against the resolved one
    # (identity shortcut) instead of comparing the original object by value.
    get_aux.assert_called_once_with(runner.model, runner.vllm_config, runner.kv_cache_config)
    assert runner.aux_output_connector == "aux"


def test_mrv2_block_copy_preserves_segmented_mamba_storage():
    storage = torch.arange(36, dtype=torch.float32)
    conv = storage[:12].view(3, 4)
    ssm = storage[12:].view(3, 8)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    upstream.copy_kv_cache_blocks_inplace([conv, ssm, conv, ssm], 3, [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)])
    torch.testing.assert_close(conv[2], before_conv[0])
    torch.testing.assert_close(ssm[2], before_ssm[0])
    torch.testing.assert_close(conv[:2], before_conv[:2])
    torch.testing.assert_close(ssm[:2], before_ssm[:2])


def test_minimax_pp_cost_counts_final_pages_and_ignores_empty_groups():
    """Final projected pages, not local pre-grouping pages, bound allocation."""
    main = FullAttentionSpec(
        block_size=256, num_kv_heads=4, head_size=128, dtype=torch.float8_e4m3fn
    )
    index = AscendSFAIndexerCacheSpec(
        block_size=2048, num_kv_heads=1, head_size=128,
        dtype=torch.float8_e4m3fn, cache_dtype_str="fp8",
    )
    groups = [KVCacheGroupSpec([], main)]
    for layer in range(8):
        groups.append(KVCacheGroupSpec([f"layer.{layer}.main"], main))
        groups.append(KVCacheGroupSpec([f"layer.{layer}.index"], index))
    physical_bytes = minimax_pp_bytes_per_block(groups)
    assert physical_bytes == 4194304
    available = 13700639641
    blocks = available // physical_bytes
    assert blocks == 3266
    assert blocks * physical_bytes <= available < (blocks + 1) * physical_bytes


@pytest.mark.parametrize("budget", [13700639641, 21821084876])
@pytest.mark.parametrize("legacy", [False, True])
def test_minimax_pp_compensates_budget_using_global_projected_specs(monkeypatch, budget, legacy):
    """Shrink only the budget; preserve upstream inputs and planning entry."""
    main = FullAttentionSpec(
        block_size=256, num_kv_heads=4, head_size=128, dtype=torch.float8_e4m3fn
    )
    index = AscendSFAIndexerCacheSpec(
        block_size=2048, num_kv_heads=1, head_size=128,
        dtype=torch.float8_e4m3fn, cache_dtype_str="fp8",
    )
    groups = []
    worker_specs = {}
    for layer in range(8):
        for kind, spec in (("main", main), ("index", index)):
            name = f"layer.{layer}.{kind}"
            groups.append(KVCacheGroupSpec([name], spec))
            worker_specs[name] = replace(spec, block_size=256)
    specs = [worker_specs]
    budgets = [budget]
    expected = budget // 16
    sentinel = object()
    monkeypatch.setattr(planner_patch, "uses_minimax_pp_group_compensation", lambda _: True)
    monkeypatch.setattr(planner_patch, "vllm_version_is", lambda _: legacy)
    monkeypatch.setattr(planner_patch, "_orig_pool_bytes_per_block", lambda _: main.page_size_bytes)
    planning = planner_patch._kv_cache_planning
    monkeypatch.setattr(planning, "get_kv_cache_groups", lambda config, merged: groups)
    monkeypatch.setattr(
        planner_patch, "get_kv_cache_config_builder", lambda _: sentinel, raising=False
    )
    monkeypatch.setattr(
        planning, "get_kv_cache_groups_from_builder",
        lambda builder, config, merged: groups, raising=False,
    )
    monkeypatch.setattr(
        planning, "_pool_bytes_per_block_from_builder",
        lambda builder, config, projected: main.page_size_bytes, raising=False,
    )
    calls = []

    def upstream_plan(config, actual_specs, actual_budgets):
        calls.append((actual_specs, actual_budgets))
        return sentinel

    monkeypatch.setattr(planner_patch, "_orig_get_kv_cache_configs", upstream_plan)
    result = planner_patch._ascend_get_kv_cache_configs(MagicMock(), specs, budgets)
    assert result is sentinel
    assert calls == [(specs, [expected])]
    assert calls[0][0] is specs
    assert budgets == [budget]
    assert expected // main.page_size_bytes == budget // (16 * main.page_size_bytes)
    assert worker_specs["layer.0.index"].block_size == 256
