# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_ascend.worker.pp.buffer import PPSendBuffer
from vllm_ascend.worker.pp.runtime import PPSendRuntime


def _gloo_worker(rank, rendezvous, first_two_posted, dynamic):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(7)
        layers = [(torch.randn(8, 8) / 4, torch.randn(8)) for _ in range(12)]

        def execute(value, start, end):
            for weight, bias in layers[start:end]:
                value = torch.tanh(value @ weight + bias)
            return value

        if rank == 0:
            buffer = PPSendBuffer(2)
            output = torch.empty(7, 4, 8)
            buffer.reserve({"hidden_states": output}, allow_dynamic=dynamic)
            for batch in range(12):
                rows = 3 if batch % 2 else 7
                boundary = 6
                buffer.acquire()
                produced = execute(torch.full_like(output[:rows], batch), 0, boundary)
                if not dynamic:
                    output[:rows].copy_(produced)
                    produced = output[:rows]

                def send(payload):
                    header = torch.tensor(payload["header"], dtype=torch.int64)
                    # Gloo Work retains the header allocation until completion.
                    return [
                        dist.isend(header, 1),
                        dist.isend(payload["hidden_states"], 1),
                    ]

                buffer.send(
                    {
                        "hidden_states": produced,
                        "header": (batch, boundary, batch, rows),
                    },
                    send,
                )
                output.fill_(-100)
                if batch == 1:
                    first_two_posted.set()
            buffer.drain()
            assert buffer.num_sends == 12
            assert buffer.num_snapshots == (0 if dynamic else 12)
        else:
            assert first_two_posted.wait(30), "producer waited for receive before filling two slots"
            for batch in range(12):
                wire = torch.empty(4, dtype=torch.int64)
                dist.recv(wire, 0)
                batch_id, cut, version, rows = wire.tolist()
                assert (batch_id, cut, version) == (batch, 6, batch)
                value = torch.empty(rows, 4, 8)
                dist.recv(value, 0)
                actual = execute(value, cut, 12)
                expected = execute(torch.full_like(value, batch), 0, 12)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dynamic", [False, True])
def test_delayed_receiver_preserves_fixed_pp_outputs(tmp_path, dynamic):
    context = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=((tmp_path / "rendezvous").as_uri(), context.Event(), dynamic),
        nprocs=2,
        join=True,
    )


def test_reserved_storage_rejects_an_unbudgeted_shape():
    buffer = PPSendBuffer(2)
    buffer.reserve({"hidden_states": torch.empty(7, 4, 8)})
    buffer.acquire()
    with pytest.raises(ValueError, match="exceeds its reserved"):
        buffer.send({"hidden_states": torch.empty(8, 4, 8)}, lambda tensors: ())
    buffer.cancel()
    assert buffer.reserved_bytes == 2 * 7 * 4 * 8 * 4


def test_snapshot_separates_output_and_metadata_from_the_next_forward():
    buffer = PPSendBuffer(1)
    output = torch.ones(2, 4, 8)
    header = [0, 4, 0]
    received = []
    buffer.acquire()

    def send(tensors):
        received.append(tensors)
        return ()

    buffer.send({"hidden_states": output, "header": header}, send)
    output.zero_()
    header[1] = 5
    assert received[0]["header"] == [0, 4, 0]
    torch.testing.assert_close(received[0]["hidden_states"], torch.ones_like(output))
    buffer.drain()


def test_dynamic_send_retains_storage_without_snapshot_until_send_completion():
    class Work:
        completed = False

        def is_completed(self):
            return self.completed

        def wait(self):
            self.completed = True

    pool = PPSendBuffer(1)
    pool.reserve({"hidden_states": torch.empty(7, 4)}, allow_dynamic=True)
    assert pool.reserved_bytes == 0
    assert pool.budget_bytes == 7 * 4 * 4
    output = torch.ones(3, 4)
    storage = weakref.ref(output.untyped_storage())
    address = output.data_ptr()
    work = Work()

    def send(payload):
        assert payload["hidden_states"].data_ptr() == address
        return [work]

    pool.acquire()
    pool.send({"hidden_states": output}, send)
    del output
    assert storage() is not None
    assert pool.num_snapshots == 0
    assert pool.num_retained_tensors == 1
    assert pool.retained_bytes == pool.live_bytes == 3 * 4 * 4
    pool.drain()
    assert work.completed
    assert storage() is None
    assert pool.retained_bytes == pool.live_bytes == 0


def test_mixed_send_copies_reused_storage_and_retains_independent_output():
    pool = PPSendBuffer(1)
    pool.reserve(
        {"hidden_states": torch.empty(7, 4), "topk": torch.empty(7, 2, dtype=torch.int32)},
        allow_dynamic=True,
    )
    output = torch.ones(3, 4)
    topk = torch.ones(3, 2, dtype=torch.int32)
    sent = []
    pool.acquire()
    pool.send(
        {"hidden_states": output, "topk": topk},
        lambda payload: sent.append(payload) or (),
        reused_storages={topk.untyped_storage()._cdata},
    )
    topk.zero_()
    assert sent[0]["hidden_states"].data_ptr() == output.data_ptr()
    assert torch.all(sent[0]["topk"] == 1)
    assert pool.num_snapshots == pool.num_retained_tensors == 1
    assert pool.reserved_bytes == 3 * 2 * 4
    assert pool.retained_bytes == 3 * 4 * 4
    assert pool.live_bytes == pool.reserved_bytes + pool.retained_bytes
    pool.drain()
    assert pool.retained_bytes == 0
    assert pool.live_bytes == pool.reserved_bytes


def test_noncontiguous_output_is_copied_to_contiguous_send_storage():
    pool = PPSendBuffer(1)
    pool.reserve({"hidden_states": torch.empty(7, 4)}, allow_dynamic=True)
    output = torch.ones(4, 3).t()
    sent = []
    pool.acquire()
    pool.send(
        {"hidden_states": output},
        lambda payload: sent.append(payload) or (),
    )
    output.zero_()
    assert torch.all(sent[0]["hidden_states"] == 1)
    assert sent[0]["hidden_states"].is_contiguous()
    assert pool.num_snapshots == 1
    pool.drain()


def test_dynamic_send_rejects_a_payload_exceeding_its_memory_budget():
    pool = PPSendBuffer(1)
    pool.reserve({"hidden_states": torch.empty(7, 4)}, allow_dynamic=True)
    pool.acquire()
    with pytest.raises(ValueError, match="exceeds its reserved"):
        pool.send({"hidden_states": torch.empty(8, 4)}, lambda payload: ())
    pool.cancel()


def test_small_view_of_oversized_storage_is_snapshotted_to_keep_retention_bounded():
    pool = PPSendBuffer(1)
    pool.reserve({"hidden_states": torch.empty(7, 4)}, allow_dynamic=True)
    oversized = torch.ones(12, 4)
    output = oversized[:3]
    sent = []
    pool.acquire()
    pool.send({"hidden_states": output}, lambda payload: sent.append(payload) or ())
    assert sent[0]["hidden_states"].data_ptr() != output.data_ptr()
    assert pool.reserved_bytes <= pool.budget_bytes
    assert pool.num_snapshots == 1
    pool.drain()


@pytest.mark.parametrize("eager,alias_kind", [(True, "model"), (True, "input"), (False, "model")])
def test_runtime_retains_fresh_eager_outputs_and_snapshots_persistent_aliases(eager, alias_kind):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("workspace", torch.ones(7, 4))

        def make_empty_intermediate_tensors(self, batch, dtype, device):
            return SimpleNamespace(
                tensors={
                    key: torch.empty(batch, 4, dtype=dtype, device=device) for key in ("hidden_states", "residual")
                }
            )

    model = Model()
    config = SimpleNamespace(
        additional_config={"pp_send_buffer_size": 1},
        use_v2_model_runner=True,
        model_config=SimpleNamespace(enforce_eager=eager, dtype=torch.float32, quantization=None),
        compilation_config=SimpleNamespace(mode=0),
        speculative_config=None,
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=7),
    )
    sent = []
    group = SimpleNamespace(
        world_size=2,
        is_last_rank=False,
        isend_tensor_dict=lambda payload, **kwargs: sent.append(payload) or (),
    )
    input_buffer = torch.ones(7, 4)
    runner = SimpleNamespace(intermediate_tensors=SimpleNamespace(tensors={"residual": input_buffer}))
    runtime = PPSendRuntime(config, model, "cpu", None, group, None, model_runner=runner)
    if eager:
        assert runtime.buffer.reserved_bytes == 0
    assert runtime.buffer.budget_bytes == 2 * 7 * 4 * 4
    output = torch.ones(3, 4)
    reused = model.workspace if alias_kind == "model" else input_buffer
    runtime.prepare(SimpleNamespace(total_num_scheduled_tokens=3))
    runtime.send({"hidden_states": output, "residual": reused[1:4]}, all_gather_group=None)
    assert (sent[0]["hidden_states"].data_ptr() == output.data_ptr()) is eager
    reused.zero_()
    assert torch.all(sent[0]["residual"] == 1)
    runtime.drain()


@pytest.mark.parametrize("mutation", ["view", "foreach", "out"])
def test_first_overwrite_waits_before_mutation_and_future_sends_snapshot(mutation):
    pool = PPSendBuffer(2)
    pool.reserve({"hidden_states": torch.empty(3, 4)}, allow_dynamic=True)
    output = torch.ones(3, 4)
    observed = []

    class Work:
        def is_completed(self):
            return False

        def wait(self):
            observed.append(output.clone())

    pool.acquire()
    pool.send({"hidden_states": output}, lambda payload: [Work()])
    with pool.protect_outputs():
        if mutation == "view":
            output.view(-1).add_(1)
        elif mutation == "foreach":
            torch._foreach_add_([output], 1)
        else:
            torch.add(output, 1, out=output)
    assert torch.all(observed[0] == 1)
    assert pool.num_reuse_waits == 1
    pool.acquire()
    sent = []
    pool.send({"hidden_states": output}, lambda payload: sent.append(payload) or ())
    output.zero_()
    assert torch.all(sent[0]["hidden_states"] == 2)
    assert pool.num_snapshots == 1
    pool.drain()


def test_guard_does_not_wait_for_unrelated_fresh_output_writes():
    pool = PPSendBuffer(2)
    pool.reserve({"hidden_states": torch.empty(3, 4)}, allow_dynamic=True)

    class Work:
        def is_completed(self):
            return False

        def wait(self):
            pass

    pool.acquire()
    pool.send({"hidden_states": torch.ones(3, 4)}, lambda payload: [Work()])
    with pool.protect_outputs():
        fresh = torch.zeros(3, 4)
        fresh.add_(1)
    assert pool.num_reuse_waits == pool.num_snapshots == 0
    pool.drain()


def test_retained_memory_accounts_for_backing_storage_and_deduplicates_aliases():
    pool = PPSendBuffer(1)
    pool.reserve({"hidden_states": torch.empty(7, 4), "residual": torch.empty(7, 4)}, allow_dynamic=True)
    output = torch.ones(7, 4)
    pool.acquire()
    pool.send({"hidden_states": output[:3], "residual": output[1:4]}, lambda payload: ())
    assert pool.reserved_bytes == 0
    assert pool.retained_bytes == pool.live_bytes == 7 * 4 * 4
    pool.drain()
    assert pool.live_bytes == 0
