# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.pp.buffer import PPRecvBuffer
from vllm_ascend.worker.pp.config import get_pp_recv_buffer_size


class Device:
    def __init__(self):
        self.current = "compute"
        self.events = []

    def Stream(self):
        return "receive"

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def Event(self):
        device = self

        class Event:
            complete = False

            def record(self, stream):
                device.events.append(("record_done", stream))

            def query(self):
                return self.complete

            def synchronize(self):
                device.events.append(("wait_done", device.current))
                self.complete = True

        return Event()

    def synchronize(self):
        self.events.append(("synchronize", self.current))


def receiver(device):
    class Input:
        device = SimpleNamespace(type="npu")

        def record_stream(self, stream):
            device.events.append(("retain", stream))

    class Work:
        def __init__(self):
            self.waited = False

        def wait(self):
            assert not self.waited, "Transport must be waited exactly once"
            self.waited = True
            device.events.append(("wait", device.current))

    class Group:
        def irecv_tensor_dict(self, all_gather_group):
            device.events.append(("allocate_and_submit", device.current))
            assert all_gather_group == "tp"
            tensors = {"hidden_states": Input(), "batch": 7}

            def gather():
                device.events.append(("gather", device.current))
                tensors["gathered"] = True

            return tensors, [Work()], [gather]

    return Group()


def test_receive_submission_does_not_inherit_compute_stream_and_wait_is_lazy():
    device = Device()
    runtime = PPRecvBuffer(2, device, receiver(device))
    tensors, handles, callbacks = runtime.receive("tp")
    assert device.current == "compute"
    assert device.events == [("allocate_and_submit", "receive")]
    assert callbacks == []
    assert "gathered" not in tensors
    handles[0].wait()
    handles[0].wait()
    assert tensors["gathered"]
    assert device.events == [
        ("allocate_and_submit", "receive"),
        ("wait", "compute"),
        ("retain", "compute"),
        ("gather", "compute"),
    ]
    runtime.drain()
    assert device.events[-1] == ("synchronize", "compute")


def test_drain_unconsumed_receive_waits_transport_without_starting_collectives():
    device = Device()
    runtime = PPRecvBuffer(2, device, receiver(device))
    runtime.receive("tp")
    runtime.drain()
    assert device.events == [
        ("allocate_and_submit", "receive"),
        ("wait", "compute"),
        ("synchronize", "compute"),
    ]


def test_receive_cannot_replace_an_unconsumed_input():
    device = Device()
    runtime = PPRecvBuffer(2, device, receiver(device))
    runtime.receive("tp")
    with pytest.raises(RuntimeError, match="Release the preceding"):
        runtime.receive("tp")


def test_consumed_receive_can_be_followed_by_another_batch():
    device = Device()
    runtime = PPRecvBuffer(2, device, receiver(device))
    for _ in range(2):
        tensors, handles, _ = runtime.receive("tp")
        handles[0].wait()
        assert tensors["gathered"]
        runtime.release()
    assert device.events.count(("allocate_and_submit", "receive")) == 2
    assert device.events.count(("wait", "compute")) == 2


def test_capacity_waits_for_oldest_consumer_before_allocating_next_input():
    device = Device()
    runtime = PPRecvBuffer(2, device, receiver(device))
    for _ in range(2):
        _, handles, _ = runtime.receive("tp")
        handles[0].wait()
        runtime.release()
    assert runtime.pending_count == 2
    assert runtime.num_backpressure_waits == 0
    runtime.receive("tp")
    assert runtime.pending_count == 2
    assert runtime.num_backpressure_waits == 1
    assert device.events[-2:] == [("wait_done", "compute"), ("allocate_and_submit", "receive")]
    runtime.drain()


def test_completed_input_is_reaped_without_backpressure():
    device = Device()
    runtime = PPRecvBuffer(1, device, receiver(device))
    _, handles, _ = runtime.receive("tp")
    handles[0].wait()
    runtime.release()
    runtime._pending[0].done.complete = True
    runtime.receive("tp")
    assert runtime.num_backpressure_waits == 0
    assert runtime.pending_count == 1
    runtime.drain()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_receive_capacity_is_rejected(capacity):
    device = Device()
    with pytest.raises(ValueError, match="positive integer"):
        PPRecvBuffer(capacity, device, receiver(device))


@pytest.mark.parametrize("size", [0, 1, 3])
def test_receive_size_alone_controls_enablement(size):
    config = SimpleNamespace(additional_config={"pp_recv_buffer_size": size}, use_v2_model_runner=True)
    assert get_pp_recv_buffer_size(config) == size


@pytest.mark.parametrize("size", [-1, True, 1.5, "2"])
def test_invalid_receive_size_is_rejected(size):
    config = SimpleNamespace(additional_config={"pp_recv_buffer_size": size}, use_v2_model_runner=True)
    with pytest.raises(ValueError, match="pp_recv_buffer_size"):
        get_pp_recv_buffer_size(config)


@pytest.mark.parametrize("gather", [False, True])
def test_receive_budget_includes_all_slots_and_optional_reconstruction_workspace(gather):
    device = Device()
    runtime = PPRecvBuffer(5, device, receiver(device))
    runtime.reserve(
        {"hidden_states": torch.empty(7, 4), "topk": torch.empty(7, 2, dtype=torch.int32)}, all_gather=gather
    )
    assert runtime.max_batch_bytes == 7 * (4 + 2) * 4
    assert runtime.budget_bytes == (5 + int(gather)) * runtime.max_batch_bytes


@pytest.mark.parametrize("invalid", ["shape", "dtype", "key", "duplicate"])
def test_receive_rejects_unbudgeted_metadata_before_tensor_allocation(invalid):
    device = Device()

    class Group:
        def recv_object(self):
            key = "other" if invalid == "key" else "hidden_states"
            metadata = [
                (
                    key,
                    SimpleNamespace(
                        size=(8 if invalid == "shape" else 3, 4),
                        dtype=torch.float64 if invalid == "dtype" else torch.float32,
                        device="cpu",
                    ),
                )
            ]
            return metadata * (2 if invalid == "duplicate" else 1)

        def irecv_tensor_dict(self, all_gather_group):
            self.recv_object()
            raise AssertionError("Invalid metadata must fail before allocation")

    group = Group()
    runtime = PPRecvBuffer(2, device, group)
    runtime.reserve({"hidden_states": torch.empty(7, 4)})
    with pytest.raises(ValueError, match="budget|Duplicate"):
        runtime.receive(None)


def test_budgeted_receive_accepts_smaller_batch_and_preserves_original_group():
    device = Device()

    class Group:
        def recv_object(self):
            return [("hidden_states", SimpleNamespace(size=(3, 4), dtype=torch.float32, device="cpu"))]

        def irecv_tensor_dict(self, all_gather_group):
            metadata = self.recv_object()
            key, value = metadata[0]
            return {key: torch.empty(value.size, dtype=value.dtype)}, [], []

    group = Group()
    original = group.recv_object
    runtime = PPRecvBuffer(2, device, group)
    runtime.reserve({"hidden_states": torch.empty(7, 4)})
    tensors, handles, _ = runtime.receive(None)
    assert tensors["hidden_states"].shape == (3, 4)
    assert group.recv_object == original
    handles[0].wait()
    runtime.release()
    runtime.drain()
