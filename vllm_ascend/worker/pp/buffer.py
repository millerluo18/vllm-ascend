# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded storage for PP sends and dynamically allocated receives."""

from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode


class SendWork(Protocol):
    def is_completed(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass
class _SendSlot:
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    works: Sequence[SendWork] = ()
    retained_storages: set[int] = field(default_factory=set)


class _PPOutputWriteGuard(TorchDispatchMode):
    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.buffer._pending:
            writable = []
            for index, argument in enumerate(func._schema.arguments):
                if argument.alias_info is not None and argument.alias_info.is_write:
                    writable.append(args[index] if index < len(args) else kwargs.get(argument.name))
            self.buffer._before_write(writable)
        return func(*args, **kwargs)


class PPSendBuffer:
    """Keep independent output storage until the matching send has completed.

    ``acquire`` runs before forward and only waits when all slots are busy.
    ``send`` snapshots reused outputs and retains independently owned outputs,
    then posts communication on a separate stream. Keeping a reference to a
    graph output alone is unsafe: replay may overwrite the same allocation.
    """

    def __init__(self, capacity: int, device_module: Any = None, *, metadata_handles: int = 0):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("PP send buffer capacity must be a positive integer")
        self._metadata_handles = metadata_handles
        self.capacity = capacity
        self._device_module = device_module
        self._comm_stream = device_module.Stream() if device_module is not None else None
        self._slots = [_SendSlot() for _ in range(capacity)]
        self._free = deque(range(capacity))
        self._pending: deque[int] = deque()
        self._active: int | None = None
        self._reserved = False
        self._allow_dynamic = False
        self._dynamic_specs = {}
        self._reused_storages: dict[int, StorageWeakRef] = {}
        self._budget_bytes = 0
        self._failure: Exception | None = None
        self.num_backpressure_waits = 0
        self.num_sends = 0
        self.num_snapshots = 0
        self.num_retained_tensors = 0
        self.num_reuse_waits = 0

    @staticmethod
    def _storage_bytes(tensors) -> int:
        storages = {
            tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
            for tensor in tensors
            if isinstance(tensor, torch.Tensor)
        }
        return sum(storages.values())

    @property
    def reserved_bytes(self) -> int:
        """Live snapshot storage, including currently unused slot buffers."""
        return self._storage_bytes(t for slot in self._slots for t in slot.buffers.values())

    @property
    def retained_bytes(self) -> int:
        """Output backing storage retained without copying, deduplicated by storage."""
        return self._storage_bytes(
            value
            for slot in self._slots
            for value in slot.payload.values()
            if isinstance(value, torch.Tensor) and value.untyped_storage()._cdata in slot.retained_storages
        )

    @property
    def live_bytes(self) -> int:
        """Unique live storage referenced by the pool, not allocator reserved memory."""
        return self._storage_bytes(
            value
            for slot in self._slots
            for values in (slot.buffers.values(), slot.payload.values())
            for value in values
        )

    @property
    def budget_bytes(self) -> int:
        """Maximum retained payload plus snapshots, including dynamic slots."""
        return self._budget_bytes if self._reserved else self.reserved_bytes

    @property
    def dynamic_enabled(self) -> bool:
        return self._allow_dynamic

    @property
    def pending_count(self) -> int:
        """Posted sends, including transfers that may already be complete."""
        self._reap_completed()
        return len(self._pending)

    def reserve(self, tensors: Mapping[str, Any], *, allow_dynamic: bool = False) -> None:
        """Reserve snapshots and budget retained outputs before KV allocation."""
        if self._pending or self._active is not None:
            raise RuntimeError("Cannot reserve PP output storage while it is in use")
        self._allow_dynamic = allow_dynamic
        self._dynamic_specs = {
            key: (value.shape, value.dtype, value.device, value.numel() * value.element_size())
            for key, value in tensors.items()
            if allow_dynamic and isinstance(value, torch.Tensor)
        }
        self._budget_bytes = self.capacity * sum(
            value.numel() * value.element_size() for value in tensors.values() if isinstance(value, torch.Tensor)
        )
        for slot in self._slots:
            slot.buffers = {
                key: torch.empty_like(value, memory_format=torch.contiguous_format)
                for key, value in tensors.items()
                if isinstance(value, torch.Tensor) and not allow_dynamic
            }
        self._reserved = True

    def _check_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("PP send buffer cannot be reused after a communication failure") from self._failure

    def _finish_oldest(self) -> None:
        index = self._pending[0]
        slot = self._slots[index]
        try:
            # Device work waits order subsequent slot writes on the caller's
            # stream. CPU metadata buffers must also remain alive until done.
            for work in slot.works:
                work.wait()
        except Exception as error:
            self._failure = error
            raise
        slot.works = ()
        slot.payload = {}
        slot.retained_storages.clear()
        self._pending.popleft()
        self._free.append(index)

    def _reap_completed(self) -> None:
        self._check_failure()
        while self._pending:
            slot = self._slots[self._pending[0]]
            # GroupCoordinator retains Gloo metadata separately. Its Work
            # does not reliably report completion until wait() is called.
            if not all(work.is_completed() for work in slot.works[self._metadata_handles :]):
                break
            self._finish_oldest()

    def acquire(self) -> None:
        """Reserve a slot before starting a forward that can produce output."""
        self._check_failure()
        if self._active is not None:
            raise RuntimeError("A PP output slot has already been acquired")
        self._reap_completed()
        if not self._free:
            self.num_backpressure_waits += 1
            self._finish_oldest()
        self._active = self._free.popleft()

    def cancel(self) -> None:
        """Release a reservation when this step produces no PP output."""
        if self._active is not None:
            self._free.appendleft(self._active)
            self._active = None

    def protect_outputs(self):
        """Intercept writes before an in-flight independent output is overwritten."""
        if self._allow_dynamic and any(self._slots[index].retained_storages for index in self._pending):
            return _PPOutputWriteGuard(self)
        return nullcontext()

    def _before_write(self, values):
        storages = set()
        stack = list(values)
        while stack:
            value = stack.pop()
            if isinstance(value, torch.Tensor):
                storages.add(value.untyped_storage()._cdata)
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif isinstance(value, Mapping):
                stack.extend(value.values())
        last_conflict = -1
        for position, index in enumerate(self._pending):
            slot = self._slots[index]
            conflicts = slot.retained_storages & storages
            if conflicts:
                last_conflict = position
                for value in slot.payload.values():
                    if isinstance(value, torch.Tensor) and value.untyped_storage()._cdata in conflicts:
                        storage = value.untyped_storage()
                        self._reused_storages[storage._cdata] = StorageWeakRef(storage)
        if last_conflict >= 0:
            self.num_reuse_waits += 1
            # A transfer already reading the original storage cannot be redirected
            # to a snapshot. Finish it before the first discovered overwrite.
            for _ in range(last_conflict + 1):
                self._finish_oldest()

    def _was_reused(self, value):
        identity = value.untyped_storage()._cdata
        reference = self._reused_storages.get(identity)
        if reference is not None and reference.expired():
            del self._reused_storages[identity]
            return False
        return reference is not None

    @staticmethod
    def _view(buffer: torch.Tensor, value: torch.Tensor) -> torch.Tensor | None:
        if buffer.device != value.device or buffer.dtype != value.dtype:
            return None
        if buffer.shape == value.shape:
            return buffer
        if (
            value.ndim
            and buffer.ndim == value.ndim
            and buffer.shape[1:] == value.shape[1:]
            and buffer.shape[0] >= value.shape[0]
        ):
            return buffer[: value.shape[0]]
        return None

    def send(
        self,
        tensors: Mapping[str, Any],
        send_fn: Callable[[dict[str, Any]], Sequence[SendWork]],
        on_snapshot: Callable[[], None] | None = None,
        *,
        reused_storages: Collection[int] = (),
    ) -> None:
        self._check_failure()
        if self._active is None:
            raise RuntimeError("Acquire a PP output slot before sending")
        index = self._active
        slot = self._slots[index]
        payload: dict[str, Any] = {}
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                payload[key] = deepcopy(value)
                continue
            if key in self._dynamic_specs:
                shape, dtype, device, max_bytes = self._dynamic_specs[key]
                fits = value.shape == shape or (
                    value.ndim
                    and value.ndim == len(shape)
                    and value.shape[1:] == shape[1:]
                    and value.shape[0] <= shape[0]
                )
                if not fits or value.dtype != dtype or value.device != device:
                    raise ValueError(f"PP tensor {key!r} exceeds its reserved output storage")
                if (
                    value.untyped_storage()._cdata not in reused_storages
                    and not self._was_reused(value)
                    and value.is_contiguous()
                    and value.untyped_storage().nbytes() <= max_bytes
                ):
                    slot.buffers.pop(key, None)
                    payload[key] = value.detach()
                    slot.retained_storages.add(value.untyped_storage()._cdata)
                    self.num_retained_tensors += 1
                    continue
            buffer = slot.buffers.get(key)
            view = None if buffer is None else self._view(buffer, value)
            if view is None:
                if self._reserved and key not in self._dynamic_specs:
                    raise ValueError(f"PP tensor {key!r} exceeds its reserved output storage")
                slot.buffers[key] = torch.empty_like(value, memory_format=torch.contiguous_format)
                view = slot.buffers[key]
            payload[key] = view.copy_(value)
            self.num_snapshots += 1

        slot.payload = payload
        if on_snapshot is not None:
            on_snapshot()
        context = nullcontext()
        if self._device_module is not None:
            assert self._comm_stream is not None
            self._comm_stream.wait_stream(self._device_module.current_stream())
            context = self._device_module.stream(self._comm_stream)
        try:
            with context:
                slot.works = tuple(send_fn(payload))
        except Exception as error:
            # A failed post may have issued some transfers. Keep its storage
            # alive and prevent further reuse rather than guessing completion.
            self._failure = error
            raise
        self._pending.append(index)
        self._active = None
        self.num_sends += 1

    def drain(self) -> None:
        """Finish all sends before shutdown, sleep, or releasing the buffers."""
        self._check_failure()
        self.cancel()
        while self._pending:
            self._finish_oldest()


class _PendingReceive:
    def __init__(self, device_module, tensors, handles, postprocess):
        self.device_module = device_module
        self.tensors = tensors
        self.handles = handles
        self.postprocess = postprocess
        self.transport_waited = False
        self.consumed = False
        self.failure = None
        self.done = None

    def wait_transport(self):
        if self.failure is not None:
            raise RuntimeError("A PP receive previously failed") from self.failure
        if not self.transport_waited:
            try:
                for handle in self.handles:
                    handle.wait()
            except Exception as error:
                self.failure = error
                raise
            self.transport_waited = True

    def wait(self):
        if self.consumed:
            return
        self.wait_transport()
        try:
            consumer = self.device_module.current_stream()
            for tensor in self.tensors.values():
                if hasattr(tensor, "device") and tensor.device.type == "npu":
                    tensor.record_stream(consumer)
            for callback in self.postprocess:
                callback()
        except Exception as error:
            self.failure = error
            raise
        self.handles = ()
        self.postprocess = ()
        self.consumed = True


class _ValidatedReceiveGroup:
    """Validate transport metadata before GroupCoordinator allocates tensors."""

    def __init__(self, group, validate):
        self._group = group
        self._validate = validate

    def __getattr__(self, name):
        return getattr(self._group, name)

    def recv_object(self, *args, **kwargs):
        metadata = self._group.recv_object(*args, **kwargs)
        self._validate(metadata)
        return metadata


class PPRecvBuffer:
    """Bound live receive batches without reserving or copying their payloads.

    Call release after forward to fence all reads of the active input. A full
    queue waits for the oldest consumer event before allocating another batch.
    """

    def __init__(self, capacity: int, device_module: Any, pp_group: Any):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("PP receive buffer capacity must be a positive integer")
        self.capacity = capacity
        self._device_module = device_module
        self._pp = pp_group
        self._comm_stream = device_module.Stream()
        self._pending = deque()
        self._active = None
        self.num_backpressure_waits = 0
        self.num_receives = 0
        self._specs = None
        self.max_batch_bytes = 0
        self.budget_bytes = 0

    def reserve(self, tensors: Mapping[str, Any], *, all_gather: bool = False):
        """Budget live inputs plus temporary reconstruction without allocating a pool."""
        if self.pending_count:
            raise RuntimeError("Cannot change receive budget while inputs are in use")
        self._specs = {
            key: (tuple(value.shape), value.dtype, value.device.type)
            for key, value in tensors.items()
            if isinstance(value, torch.Tensor)
        }
        self.max_batch_bytes = sum(
            value.numel() * value.element_size() for value in tensors.values() if isinstance(value, torch.Tensor)
        )
        self.budget_bytes = (self.capacity + int(all_gather)) * self.max_batch_bytes

    def _validate_metadata(self, metadata):
        for key, value in metadata:
            if not all(hasattr(value, attr) for attr in ("size", "dtype", "device")):
                continue
            spec = self._specs.get(key)
            shape = tuple(value.size)
            device_type = torch.device(value.device).type
            if spec is None:
                raise ValueError(f"Unbudgeted PP receive tensor {key!r}")
            maximum, dtype, expected_device = spec
            fits = shape == maximum or (
                len(shape) == len(maximum) and bool(shape) and 0 <= shape[0] <= maximum[0] and shape[1:] == maximum[1:]
            )
            if not fits or value.dtype != dtype or device_type != expected_device or any(dim < 0 for dim in shape):
                raise ValueError(f"PP receive tensor {key!r} exceeds its startup budget")
        keys = [key for key, _ in metadata]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate PP receive metadata keys")

    @property
    def pending_count(self):
        return len(self._pending) + int(self._active is not None)

    def receive(self, all_gather_group):
        if self._active is not None:
            raise RuntimeError("Release the preceding PP input before receiving again")
        while self._pending and self._pending[0].done.query():
            self._pending.popleft()
        if len(self._pending) == self.capacity:
            self.num_backpressure_waits += 1
            self._pending[0].done.synchronize()
            self._pending.popleft()
        # Allocation and HCCL submission must not inherit the compute stream tail.
        with self._device_module.stream(self._comm_stream):
            if self._specs is None:
                tensors, handles, postprocess = self._pp.irecv_tensor_dict(all_gather_group=all_gather_group)
            else:
                if getattr(self._pp, "use_cpu_custom_send_recv", False):
                    raise RuntimeError("Budgeted PP receive requires the standard tensor-dict transport")
                group = _ValidatedReceiveGroup(self._pp, self._validate_metadata)
                tensors, handles, postprocess = type(self._pp).irecv_tensor_dict(
                    group, all_gather_group=all_gather_group
                )
        assert tensors is not None
        self._active = _PendingReceive(self._device_module, tensors, handles, postprocess)
        self.num_receives += 1
        return tensors, [self._active], []

    def release(self):
        if self._active is None:
            return
        # Even an unused input must finish transport before its storage is freed.
        self._active.wait_transport()
        self._active.done = self._device_module.Event()
        self._active.done.record(self._device_module.current_stream())
        self._pending.append(self._active)
        self._active = None

    def drain(self):
        if self._active is not None:
            self._active.wait_transport()
        self._device_module.synchronize()
        self._active = None
        self._pending.clear()
