# Pipeline communication buffers

PP communication buffers manage the lifetime of intermediate tensors for fixed-partition pipeline parallelism with Model Runner V2. Sending can use independent snapshots or retain the original output storage. Receiving uses bounded dynamic allocations and a separate submission stream. Both implementations live in `vllm_ascend/worker/pp/buffer.py` and use standard vLLM communication interfaces.

## Configuration

Enable sending alone:

```bash
--additional-config '{"pp_send_buffer_size":5}'
```

Enable sending and receiving:

```bash
--additional-config '{"pp_send_buffer_size":5,"pp_recv_buffer_size":5}'
```

| Option | Default | Meaning |
| --- | --- | --- |
| `pp_send_buffer_size` | `0` | Maximum send slots on each non-last PP stage |
| `pp_recv_buffer_size` | `0` | Maximum live receive batches on each non-first PP stage |

Zero disables the corresponding feature. Sizes must be non-negative integers; booleans are rejected. A positive size requires Model Runner V2. The options are independent, count batches rather than requests or tokens, and do not add threads or change RPC scheduling. PP=1 does not create either buffer.

The fixed layer partition, output length and scheduling semantics remain unchanged. Buffering does not introduce a `max_tokens=1` requirement or require chunked prefill to be disabled. Existing model and platform constraints still apply.

## Send lifecycle

Without the send pool, the worker posts an asynchronous send and waits on its handles at the beginning of the next execution. This protects reused output storage but can order the next computation after the previous transfer.

With the send pool:

1. After model loading, `PPSendRuntime` builds the maximum outgoing template using `max_num_batched_tokens`. It prefers the intermediate-tensor factory's `for_send` method when available.
2. Before a scheduled forward, `prepare()` acquires a slot. Completed sends are reaped first; a full pool waits for the oldest send.
3. The worker executes the model under the output write guard when required. A step without PP output cancels its reservation.
4. `send()` retains eligible output storage or copies into a slot snapshot. Non-tensor metadata is deep-copied.
5. The send submission stream waits for the current producer stream, then posts `isend_tensor_dict`. Payloads and communication handles remain referenced by the slot.
6. Completion or capacity pressure releases slots in FIFO order. Communication failures prevent further pool reuse and retain potentially in-flight storage.

A capacity of two allows A to be sending while B uses the second slot. C waits for A only if both slots remain occupied. It does not remove the downstream dependency on receiving A before computing A, or guarantee a throughput improvement when another stage is the bottleneck.

Sleep and model-reload lifecycle hooks drain outstanding communication. Draining waits for sends and drops payload references while retaining reusable snapshot buffers. These lifecycle integrations still require device regression testing.

## Automatic output storage management

The pool does not require model whitelists or user-supplied tensor names such as `pp_dynamic_output_keys`.

| Execution or storage condition | Strategy |
| --- | --- |
| Explicit eager execution, compilation mode zero, no speculative decoding and no LoRA | Permit reference retention with write protection |
| Alias of a known model tensor attribute/buffer or runner intermediate input | Snapshot |
| Storage previously observed being overwritten while in flight | Snapshot on subsequent sends |
| Noncontiguous output or a view whose backing allocation exceeds the field budget | Contiguous snapshot |
| Graph, compiled, speculative or LoRA execution | Conservative snapshots |

Eligible outputs are detached without copying and retained until send completion. Storage identity uses the underlying storage, so offset views can still be recognized as aliases. Shape, dtype and device must fit the reserved template; an unexpected tensor field or oversized shape fails rather than silently expanding the startup budget.

### Protecting the first overwrite

Comparing consecutive output addresses after forward is too late: the next forward may already have overwritten the data being sent.

When retained outputs are pending, `protect_outputs()` installs a `TorchDispatchMode` around model execution. The guard inspects writable arguments declared in operator schemas, including tensor lists and `out=` arguments. Before a write to retained in-flight storage, it waits for the conflicting sends and earlier FIFO sends, then records that storage as reusable. Subsequent sends of the same live storage use snapshots. `StorageWeakRef` distinguishes a live allocation from later allocator address reuse.

The first conflict must wait because an already submitted transfer cannot be redirected to a new snapshot. A device work wait must establish the appropriate dependency on the caller's stream before the write proceeds.

This mechanism depends on correctly declared mutations passing through PyTorch dispatch. Opaque CANN writes or custom operators with undeclared side effects require explicit integration or a conservative snapshot policy. The current guard is an implementation prototype, not evidence that every model's custom operators are safe. Its Ascend compatibility and overhead remain to be validated.

## Send memory accounting

Let `B_send` be the sum of maximum tensor payload bytes from the outgoing template:

```text
B_send = sum(max_numel(field) * element_size(field))
M_send = pp_send_buffer_size * B_send
```

Automatic KV sizing deducts `M_send` for either storage strategy. Snapshot-only execution preallocates slot buffers. Reference-retention execution initially records specifications and budget, then allocates fallback snapshots as needed. Unused snapshot headroom is not returned to KV because later steps may switch to copying.

Copying can hold both the source output and a snapshot while the source is live. Retaining avoids the duplicate allocation but extends the source lifetime. No-copy therefore does not mean zero memory cost.

| Property | Meaning |
| --- | --- |
| `reserved_bytes` | Live snapshot backing storage, including idle slot buffers |
| `retained_bytes` | Backing storage referenced by retained no-copy payloads |
| `live_bytes` | Unique union of snapshot and payload backing storage |
| `budget_bytes` | Maximum payload budget used for startup planning |

Actual storage statistics count full backing allocations and deduplicate aliases. They measure pool-referenced storage, which may also be referenced by the model, rather than exclusively pool-owned memory. They are not device allocator reserved-memory statistics and exclude fragmentation, alignment and communication workspace.

## Send and receive templates

Outgoing and incoming PP boundaries can carry different tensor sets. Transport factories may expose `for_send` to reserve outgoing Top-K indices and auxiliary states accurately; receive budgeting uses the ordinary incoming factory. A stage's send budget must not be inferred from its receive template.

The pool operates on tensor dictionaries and has no PP=2 restriction. Models must still provide accurate transport templates. Parallel layouts must agree on payload shape and TP slicing/all-gather behavior.

## Independent receive submission

Receive storage is dynamically allocated from incoming metadata without a staging copy or a fixed preallocated receive pool. `PPRecvBuffer.receive()` runs allocation and `irecv_tensor_dict` on a separate submission stream, avoiding an implicit dependency on the preceding compute stream tail during HCCL submission.

Before consuming tensors, the compute stream waits on receive handles. Receive storage is recorded on the consumer stream before reconstruction callbacks. Completed all-gather reconstruction releases old handle and callback references rather than retaining the pre-gather backing storage for the whole queue.

After forward, `release()` records a consumer event and queues the batch. Completed batches are reaped; at capacity, the oldest event is synchronized before another batch is allocated. The capacity includes the active receive and released batches whose consumer events are not complete. An unused receive is drained without starting all-gather callbacks.

A separate stream does not prefetch future RPCs. The next receive can be submitted only when the corresponding worker call and metadata are available. Actual overlap with preceding device computation also depends on stream dependencies and communication resources.

## Receive memory budget

At model load, each non-first stage derives `B_recv`, its maximum incoming payload, from `make_empty_intermediate_tensors`, `max_num_batched_tokens` and field dtypes. Only the specifications and budget are retained after the temporary template is released.

```text
M_recv = pp_recv_buffer_size * B_recv
# Current worker conservatively adds reconstruction headroom when TP > 1:
M_recv += B_recv  # only when TP > 1
```

For example, a maximum incoming batch of 4 GiB and receive capacity five budgets 20 GiB with TP=1, or 24 GiB with TP>1. Send storage is budgeted separately on stages that also send.

Before standard transport allocates tensors, a local validating group view checks received tensor metadata for allowed keys, shapes, dtypes, device types and duplicate keys. Smaller leading batch dimensions are accepted within the template maximum. The view does not mutate the shared group or require changes to vLLM. The custom CPU send/receive transport cannot use this validation path.

Automatic KV sizing deducts both applicable communication budgets and fails if no positive KV budget remains. Dummy profiling does not exercise the multi-batch receive queue, so the full receive budget is deducted without assuming a profiled single batch can be credited back. This conservative calculation may overlap some activation headroom.

Explicit `kv_cache_memory_bytes` retains its existing user-managed semantics and bypasses automatic deductions. Payload budgets do not cover all allocator and communication overhead, so they do not provide an unconditional OOM guarantee.

## Compatibility and validation status

Graph and speculative execution select snapshot sends; this is a storage policy, not a claim that all model/feature combinations have passed device validation. SP uses the worker's selected all-gather group in the send runtime. Templates and payload layouts still require device validation. PCP and DCP are currently rejected by the AscendConfig buffering validation until their communication layouts are validated; the size helper alone does not express that restriction. The worker's existing external-launcher restriction remains applicable.

The current local implementation has passed 39 CPU unit-test cases across:

- `tests/ut/worker/test_pp_send_buffer.py`
- `tests/ut/worker/test_pp_recv_runtime.py`

Two delayed-receiver communication cases were excluded. The executed unit tests use simulated communication handles and device stream/event objects; they do not exercise NPU/HCCL. Ruff and `git diff --check` passed. Automatic storage selection and the latest receive budgeting have not yet completed Ascend end-to-end validation; earlier-version measurements are not results for these changes.

Device validation should cover shared-output correctness, custom-operator writes, mixed batch sizes, stable memory peaks, graph/speculative/context-parallel combinations, and sleep/wake behavior. Use profiling to inspect actual stream/event dependencies. Compare buffer sizes zero and positive using `vllm bench serve` with identical weights, layer partitions, request sequences and feature settings. Report elapsed time, throughput, TTFT and TPOT; reduced elapsed time is a latency improvement. A communication microbenchmark alone does not establish serving gains.

See the [PP send-buffer RFC](rfc_pp_send_buffer.md) for motivation, alternatives and review questions.
