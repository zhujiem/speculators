# PR: perf(ascend): flat-index KV cache gather/scatter + tolerate aborted requests in KV-transfer finalization

- 日期: 2026-11-13
- 来源: s67 容器 `vllm_ascend_v0.30.0` 内 `/vllm-workspace/vllm` 工作区实测 git diff（两节点部署态，双容器三文件 md5 一致：extract_hidden_states.py `f8296ef8` / example_hidden_states_connector.py `027e142e` / scheduler.py `e169977f`）
- 基线: vllm v0.30.0 (ced6857, detached HEAD)
- 补丁文件: vllm_hidden_states_s67.patch（5362 字节）
- 补丁 MD5: b3c073d8382bc42bb21b9fb251c57eb3
- 与 devserver 生成版 `PR_vllm_ascend_flat_index_kv_xfer/vllm_ascend_flat_index_kv_xfer_20261113.patch` **逐字节一致**（同 md5），本目录为部署机实况存档
- 状态: 已应用于 s67+s68 双容器并在其上完成全部验证

## Purpose (可直接粘贴到 PR 描述)

The hidden-states extraction pipeline (example KV connector + `extract_hidden_states`
model) reads and writes the paged KV cache with three-subscript advanced indexing
(`kv_cache[block, :, pos]`). On Ascend 910b these `aten::index` / `aten::index_put_`
kernels fall back to AiCPU, costing ~91.5ms per scatter on a production-shaped
`[985, 6, 128, 6144]` bf16 tensor. Beyond raw slowness, the AiCPU path intermittently
deadlocks on large ops (~2.4GB scatters for 32K-token requests) and the 600s AICPU
watchdog then kills the whole engine — reproduced deterministically at 64-client
concurrency (62/64 completions, engine death at wall = stuck-op + 600s).

This PR rewrites both the gather (`extract_from_kv_cache`) and the scatter
(`basic_cache`) to a flat single-subscript form when the cache tensor is contiguous:
`row = (block * num_heads + head) * block_size + pos`, then one `view(-1, head_size)`
index op. The row mapping is bit-exact versus the original three-subscript form
(synthetic full-shape bench 34.5ms -> 0.1ms for the gather; 31/31 real production
scatter ops replayed bit-identically; ~3.8ms/op vs 91.5ms/op). Non-contiguous caches
keep the original code path.

Two robustness fixes ride along for requests that abort while a KV transfer is in
flight: the scheduler's `assert req_id in self.requests` becomes a warning + skip,
and the connector's `get_finished` pops with a default instead of raising KeyError.

## What changed

- `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` (+19/-1):
  flat single-subscript gather on contiguous caches (fallback to original form);
  `get_finished` pop-with-default, late aborts return `(False, None)`.
- `vllm/model_executor/models/extract_hidden_states.py` (+19/-1):
  `basic_cache` scatter rewritten to the same flat-index form; padding-slot redirect
  to the null block and the branch-free/sync-free property preserved.
- `vllm/v1/core/sched/scheduler.py` (+12/-2):
  `finished_recving` / `finished_sending` loops warn + `continue` for unknown or
  already-aborted request ids instead of asserting.

## Verification (2026-11-13, 2×Atlas A3 节点 s67/s68, 16-32K 混合数据集)

32×16K + 32×32K prompts (1.57M tokens), 64 samples per concurrency level,
hidden states (tokens, 6, 6144) bf16 exported via mooncake over TCP:

| 部署 | c16 | c32 | c64 | 结论 |
|---|---|---|---|---|
| 2×独立 TP16 + LB（本 PR 底座） | 62/64*, 4599 tok/s | 64/64, 4756 tok/s | 64/64, 4424 tok/s | 吞吐全场最优 |
| DP2 seqs2 16K窗（本 PR 底座） | 64/64, 4650 tok/s | 64/64, 4531 tok/s | 64/64, 4222 tok/s | 192/192 全绿 |
| DP2 seqs2 32K窗（本 PR 底座） | 64/64, 3897 tok/s | 64/64, 3960 tok/s | 64/64, 3678 tok/s | 32K 单步 prefill 安全 |
| PR 前最优 (DP2-g, 分块缓解) | 64/64, 4504 | 64/64, 4599 | **62/64 @953s 引擎死亡** | AiCPU 死亡复现 |

\* 2 个 `Mooncake get_into -707` 偶发传输错误，引擎无损，数据未损坏。

关键对照：PR 前 64 并发尾部单 op 卡死 → 600s 看门狗处决引擎（死亡时刻精确
等于卡死点 + 600s，跨 3 次复现）；PR 后同一负载三架构全部通过，引擎零死亡，
且 32K-token 单步 prefill（PR 前必挂场景）全程安全。

## Apply

    cd /vllm-workspace/vllm && git apply vllm_hidden_states_s67.patch

(基于 tag v0.30.0 / ced6857 的工作区生成; 其他基线如冲突需人工核对上下文行。)

## 备注 (按仓库 PR 规范)

- 本改动为 AI 辅助完成; 提交前需人工逐行审查, PR 中需声明 AI assistance。
- "I (a human) have written or reviewed the code" 勾选框保持未勾选。
- 上游提交建议拆分: (1) flat-index gather/scatter 性能优化与 AiCPU 死锁消除,
  (2) aborted-request 健壮性修复 (scheduler assert -> warning; connector pop default)。
- 配套部署侧修改（不在本 PR 内）: mooncake connector 读路径改用扁平 gather +
  视图切片、store 8M 元素 DtoH 子块、pinned 暂存。
