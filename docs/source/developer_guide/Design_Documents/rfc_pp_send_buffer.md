# [RFC]: Bounded PP Send Buffer with Automatic Output Storage Management on Ascend

## Motivation.

在固定分层的 Pipeline Parallelism（PP）中，非末阶段完成一个 batch 的计算后，需要将中间张量发送给下一个阶段。现有 worker 使用异步 `isend_tensor_dict`，但在下一次 `execute_model` 开始时等待上一轮发送 handle，再执行新的计算。

异步通信接口并不自动意味着连续 batch 的计算与发送能够重叠。对于会复用输出存储的模型执行路径，如果上一轮发送尚未完成，下一轮计算就覆盖同一存储，下游可能收到被修改的数据。逐轮等待提供了存储保护，但也会把上一批发送的依赖带到下一批计算前。

本 RFC 提议引入有界 PP 发送缓冲，使发送数据的生命周期由发送槽位管理。对于复用的存储创建快照；对于可安全保留的独立输出保留引用。只在容量不足或即将覆盖仍在发送的存储时等待，从而为 batch A 的发送与 batch B 的计算提供重叠机会。

预期适用场景包括较大的 prefill 中间激活、不同阶段耗时不均衡，以及有足够调度请求让多个 batch 在流水线上运行的负载。缓冲不会缩短最慢阶段的计算时间，也不改变 PP 的理论吞吐上限；收益来自减少存储复用导致的等待、降低流水线空泡。负载已经饱和、通信很短或下游持续成为瓶颈时，收益可能有限，复制还可能带来额外开销。

## Proposed Change.

### 1. 接口与范围

通过已有 `additional_config` 配置启用，不增加模型字段声明：

```bash
--additional-config '{"pp_send_buffer_size":5}'
```

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `pp_send_buffer_size` | `0` | 每个非末 PP 阶段的发送槽位数；正整数启用，零关闭 |
| `pp_recv_buffer_size` | `0` | 可选配套能力：每个非首阶段保留的接收 batch 数；与发送容量独立 |

容量按 batch 计，不是字节、token 数或请求数。一个 batch 的多个中间张量属于同一槽位。配置拒绝负数、布尔值和非整数；启用时要求 Model Runner V2。PP=1 或末阶段不创建发送池。

实现全部位于 vLLM Ascend，复用标准 vLLM 张量字典通信接口，不要求修改 vLLM 核心协议。分层保持固定，不涉及动态切层、KV 移交或调度策略修改，也不新增线程。

### 2. 从初始化到发送的执行流程

1. **模型加载完成后建立运行时。** Worker 创建 `PPSendRuntime`。从模型的 `make_empty_intermediate_tensors` 获取模板；存在 `for_send` 时优先使用发送模板，避免把接收边界的张量集合误用于发送。
2. **确定最大负载和显存预算。** 使用 `max_num_batched_tokens`、各字段形状和 dtype 计算最大单批发送字节数。复制路径提前创建槽位快照；允许动态保留的路径记录规格和预算，按需分配回退快照。
3. **计算前取得槽位。** `prepare()` 对有调度 token 的步骤调用 `acquire()`，先回收已完成发送；只有全部槽位占用时，才等待最旧发送释放槽位。
4. **执行模型。** 在自动存储保护上下文中调用 `model_runner.execute_model()`。如本轮没有需要跨 PP 发送的 `IntermediateTensors`，或计算异常，调用 `cancel()` 释放本轮槽位预约。
5. **构造发送负载。** 对每个 Tensor 判断保留引用或复制；非 Tensor 元数据深复制，避免后续修改影响已提交的发送。
6. **提交通信。** 专用发送提交 stream 等待生产输出/快照的当前 stream，再调用 `isend_tensor_dict`。这规定了生产先于发送，不要求每个后续 forward 都先等待上一批发送。
7. **持有到完成。** 槽位持有 payload 和通信 handle。回收时等待相应 handle 后释放引用；容量压力按 FIFO 回收，保持发送提交顺序。
8. **生命周期结束时排空。** Sleep、模型重新加载或关闭相关路径调用 `drain()`，完成未结束通信后再释放存储。通信提交/等待失败后禁止发送池继续复用，保留可能仍被通信使用的存储。

槽位状态为：

```text
FREE --acquire--> ACTIVE --send--> PENDING --completion/wait--> FREE
                       \--cancel-----------------------------> FREE
```

例如容量为 2：A 占用槽位 0 发送时，B 可以取得槽位 1 计算和发送；准备 C 时，如果 A 仍未完成，则等待 A。它不会允许无限制地积累发送数据。

下游仍必须先收到 A 才能计算 A。发送池解除的是输出存储生命周期带来的额外串行依赖，不解除模型的数据依赖，也不保证通信硬件最终实现重叠。

### 3. 自动决定保留引用或复制

用户不需要知道 `hidden_states`、`residual`、Top-K 或辅助字段的具体名字，也不需要配置 `pp_dynamic_output_keys`。

| 情况 | 当前策略 |
| --- | --- |
| 显式 eager、编译关闭、无投机、无 LoRA | 允许自动保留独立输出，并启用写入保护 |
| 输出与已知模型持久 Tensor 或 runner 中间输入共享存储 | 复制到发送槽位 |
| 已经观察到后续步骤会改写的同一存储 | 后续发送使用快照 |
| 非连续输出，或小视图持有的底层分配超过该字段最大预算 | 复制为独立连续张量 |
| 图模式、编译、投机或 LoRA 路径 | 保守使用快照 |

存储身份依据底层 storage，而非只比较 Tensor 对象或 `data_ptr()`；带偏移的视图也可能共享同一存储。

#### 首次发现复用时的保护

仅在下一轮 forward 结束后比较输出地址无法保证正确性，因为数据可能已经被覆盖。当前原型在存在未完成的保留输出时，使用 `TorchDispatchMode` 检查算子 schema 中声明为可写的参数，包括 Tensor 视图、Tensor 列表和 `out=` 参数。

当一个写操作即将触及仍在发送的存储：

1. 在执行写操作前，等待所有冲突发送及其之前的 FIFO 发送；设备 handle 的等待需建立调用 stream 到通信完成的依赖。
2. 用 `StorageWeakRef` 记录该存储会被复用，避免把存储释放后的地址重用误判为同一对象。
3. 允许写操作继续；随后该存储的发送使用复制路径。

已经开始读取原存储的发送无法事后改为读取快照，因此首次冲突需要等待，后续才能通过快照避免同类依赖。

**正确性边界：** 该机制依赖算子正确声明写入，并经过 PyTorch dispatcher。绕过 dispatcher 的 CANN 写入、错误声明副作用的自定义算子，不能仅靠此机制自动保证安全；需要审计、集成明确的写入保护或保持保守快照。这是设备验收前必须解决的条件，不能将本地 UT 通过视为所有模型都已安全适配。

### 4. 张量模板与模型集成

发送模板必须覆盖当前 PP 输出边界的所有字段。接收模板描述输入边界，二者不必相同。例如跨阶段传递 Top-K 索引或辅助状态时，一个阶段可能接收某个字段、在本阶段更新后发送另一集合。

发送池本身按张量字典工作，不按模型名称选择策略，也不限制 PP=2。模型已有的中间张量工厂应准确表达发送规格；需要时通过 `for_send` 提供发送侧模板。若字段超过模板规格，发送前直接报错，不静默扩大已承诺的显存预算。

### 5. 显存预算与实际占用

定义最大单批发送量：

```text
B_send = sum(max_numel(field) * element_size(field))
M_send_budget = pp_send_buffer_size * B_send
```

启动时自动计算 KV cache 大小时，保守扣除 `M_send_budget`。不复制仍会延长原输出的存活时间，而且运行中可能回退为复制，所以不能把当前省下的快照空间直接分配给 KV cache。

| 指标 | 含义 |
| --- | --- |
| `reserved_bytes` | 实际快照底层存储，包含空闲槽位仍持有的快照 |
| `retained_bytes` | 保留引用路径持有的输出底层存储，按 storage 去重 |
| `live_bytes` | 发送池引用的快照与 payload 存储并集，按 storage 去重 |
| `budget_bytes` | 启动时用于规划最大 payload 的安全预算 |

复制路径在源输出仍存活时会同时持有源和快照；不复制路径避免该副本，但仍持有源。视图按整个底层分配计费，多个别名不重复计费。上述实际指标表示池引用的存储，不等同于池独占增量或设备分配器的 reserved memory。

预算是 payload 的保守上限，不包括所有 allocator 对齐、碎片、通信 workspace；模型激活峰值仍由 profiling 估计。显式指定 `kv_cache_memory_bytes` 时，保持原有用户管理 KV 大小的语义，绕过自动扣减。

### 6. 可独立启用的接收配套能力

```bash
--additional-config '{"pp_send_buffer_size":5,"pp_recv_buffer_size":5}'
```

接收不复制到固定池，而是动态分配。独立提交 stream 运行标准 `irecv_tensor_dict`，避免接收提交继承上一轮计算 stream 尾部的依赖。计算消费输入前等待对应接收 handle；forward 返回后记录 consumer event，容量满时等待最旧事件，再分配下一批。

容量包含正在接收及计算完成事件尚未结束的 batch。它不预先从 RPC 队列读取未来任务；只有下一次 worker 调用已经到达，才可能提交下一次接收。因此是否实际与上一批设备计算重叠，取决于 RPC、元数据、stream 依赖和通信资源。

每个非首阶段按接收模板计算 `B_recv`，自动 KV 规划扣除：

```text
M_recv_budget = pp_recv_buffer_size * B_recv
# 当前 TP>1 时保守增加一批 all-gather 重建余量：
M_recv_budget += B_recv
```

接收元数据在张量分配前校验字段、形状、dtype、设备类型及重复键。检查通过标准通信函数的局部代理实现，不修改共享 group 或 vLLM 核心。该预算校验路径不支持自定义 CPU 收发实现。重建完成后释放旧 handle/callback 引用，避免原始接收存储随整个消费队列长期保留。

当前 dummy profiling 不执行多批接收队列，因此按整队列保守扣减，不假设可以抵扣已 profile 的某一批；可能与部分激活余量重叠，但不会基于未经验证的抵扣扩大 KV。

### 7. 兼容性与实施文件

发送缓冲不新增 `max_tokens=1` 或关闭 chunked prefill 的要求，不改变解码语义。图模式、投机路径采用快照并不等于其所有组合已经通过设备测试；模型、平台及 MRV2 自身的限制仍适用。

SP 发送路径现已沿用 worker 选择的 all-gather 参数，仍需核对实际 tensor layout 和发送模板。PCP、DCP 当前由 AscendConfig 缓冲校验拒绝，待对应布局验证后再放开。Sleep 的生命周期已接入 drain，但实际 allocator/sleep/wake 行为仍需设备回归。保留 worker 通用路径已有的 external launcher 限制。

| 文件 | 职责 |
| --- | --- |
| `vllm_ascend/worker/pp/config.py` | 发送/接收容量读取和基础校验 |
| `vllm_ascend/worker/pp/buffer.py` | 槽位、快照/引用、写入保护、通信生命周期、接收预算与容量 |
| `vllm_ascend/worker/pp/runtime.py` | 模板选择、执行模式策略及发送通信集成 |
| `vllm_ascend/worker/worker.py` | 初始化、KV 预算、forward 前后接入与 drain |
| `tests/ut/worker/test_pp_send_buffer.py` | 输出正确性、保活、写入冲突和显存统计测试 |
| `tests/ut/worker/test_pp_recv_runtime.py` | 接收顺序、容量反压及分配前预算校验测试 |

### 8. 验证与验收计划

**当前版本已完成：** 本地 CPU 隔离运行上述两个 UT 文件，排除 `delayed_receiver` 通信测试后，39 项通过、2 项未运行；Ruff 和 `git diff --check` 通过。stream/event/HCCL 相关对象在这些 UT 中为模拟对象。新增自动存储选择及接收预算尚未完成 Ascend 设备端到端验证，不引用旧版本的性能结果作为本版本结论。

设备验收应包含：

- 容量 0/1/2/5，PP2/PP4；固定相同分层、权重、请求序列、并发和其他配置。
- 延迟下游接收并持续覆写上游输出，验证无数据污染；覆盖 fresh 输出、共享视图和实际自定义算子。
- 覆盖 prefill、chunked prefill、多 token decode，并分别验证 graph、投机和上下文并行组合。
- 测量启动 KV 预算、运行时 allocated/reserved 峰值和池存储统计；混合长度及长期压力下检查显存是否稳定。
- Profiling 确认 A 发送与 B 计算的 stream/event 关系，并检查快照 copy 及写入保护的额外开销。
- 使用 `vllm bench serve` 对比总耗时、请求吞吐、token 吞吐、TTFT 和 TPOT；以同负载结果计算变化，耗时降低才表示时延改善。不把合成通信测试的重叠比例直接当成服务性能提升。

## Feedback Period.

建议自 RFC 发布之日起至少一周；正式发布日期和截止日期在发布时填写。

## CC List.

待作者填写相关 PP、MRV2、通信和显存管理维护者；本地文档不自动通知任何人。

## Any Other Things.

- 状态：本地实现草案，尚未发布 RFC。
- 当前工作分支：`codex/ascend-pp-send-buffer-only`；本文包含尚未提交的最新修改。
- 希望讨论：自动写入保护的可接受开销与自定义算子安全边界；发送/接收预算与 profiling 的去重策略；接收独立 stream 是否与发送缓冲分阶段合入。
- 备选方案：所有输出始终快照，正确性边界更清晰但增加复制；全部只保留引用，无法保护共享存储；逐批等待原发送，维持原有串行依赖。当前提议在有界容量下结合引用保留与快照，并对未验证执行模式保持保守策略。
- 本文由 AI 辅助整理，发布前需由提交者审阅实现和验证结论。
- 模板来源：[vLLM Ascend 官方 RFC issue 模板](https://github.com/vllm-project/vllm-ascend/blob/main/.github/ISSUE_TEMPLATE/750-RFC.yml)。本文沿用其五个顶层栏目，内容依据本地当前实现编写。
