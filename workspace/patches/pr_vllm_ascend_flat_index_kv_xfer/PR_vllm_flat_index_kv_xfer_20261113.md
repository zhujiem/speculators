# PR: perf(ascend): flat-index KV cache gather/scatter + tolerate aborted requests in KV-transfer finalization

- 日期: 2026-11-13
- 目标仓库: /vllm-workspace/vllm (devserver-111.127.52.27, 容器内, detached HEAD @ tag v0.30.0, ced6857 "[CI][Build] Harden triton-cpu sleef submodule fetch in CPU image build (#57871)", origin = github.com/vllm-project/vllm.git)
- 补丁文件: vllm_ascend_flat_index_kv_xfer_20261113.patch
- 补丁 SHA256: 16df4e7237dbd0e32f4dfa2147dfd7832e5cd4a61391dc3ba00ab369a81f54cc
- 补丁 MD5: b3c073d8382bc42bb21b9fb251c57eb3 (5362 字节, 与远端 git diff 逐字节校验一致)
- 状态: 已应用于工作区 (3 files modified, +46/-4), 未提交; 工作区另有若干 `.flatbak` / `.orig_dsh` / `.bak_kvassert` 备份文件, 为本地实验残留, 未包含在补丁中

## Purpose (可直接粘贴到 PR 描述)

The hidden-states extraction pipeline (example KV connector + `extract_hidden_states`
model) reads and writes the paged KV cache with three-subscript advanced indexing
(`kv_cache[block, :, pos]`). On Ascend 910b these `aten::index` / `aten::index_put_`
kernels fall back to AiCPU, costing ~91.5ms per scatter on a production-shaped
`[985, 6, 128, 6144]` bf16 tensor and dominating the step time of the hidden-state
collection run used to build speculator training data.

This PR rewrites both the gather (`extract_from_kv_cache`) and the scatter
(`basic_cache`) to a flat single-subscript form when the cache tensor is contiguous:
`row = (block * num_heads + head) * block_size + pos`, then one `view(-1, head_size)`
index op. The row mapping is bit-exact versus the original three-subscript form
(synthetic full-shape bench 34.5ms -> 0.1ms for the gather; 31/31 real production
scatter ops replayed bit-identically, including padding slots redirected to the null
block and duplicate targets; ~3.8ms/op vs 91.5ms/op). Non-contiguous caches keep the
original code path, so behavior on GPU/other backends is unchanged.

Two robustness fixes ride along, both hit in production when a request aborts while
a KV transfer is in flight:

- `Scheduler._update_from_output` asserted `req_id in self.requests` when the
  connector reported `finished_recving` / `finished_sending` for a request that had
  already been aborted and removed. The assert is replaced with a warning + skip.
- `ExampleHiddenStatesConnector.get_finished` did `self._request_filenames.pop(req_id)`
  which raises `KeyError` on a late abort (or duplicate finish) after finalization;
  it now pops with a default and returns `(False, None)`.

## What changed

- `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py` (+19/-1):
  - `extract_from_kv_cache`: flat single-subscript gather on contiguous caches,
    falling back to the original three-subscript form otherwise.
  - `get_finished`: `pop(req_id, None)`; missing entry returns `(False, None)`
    instead of raising.
- `vllm/model_executor/models/extract_hidden_states.py` (+19/-1):
  `basic_cache` scatter rewritten to the same flat-index form on contiguous caches;
  padding-slot redirect to the null block (block 0) and the branch-free/sync-free
  property preserved.
- `vllm/v1/core/sched/scheduler.py` (+12/-2):
  both `finished_recving` and `finished_sending` loops warn + `continue` for
  unknown/aborted request ids instead of asserting.

## Design notes

- Flat mapping `row = (block * num_heads + head) * block_size + pos` matches how the
  cache is laid out for `basic_cache`-style tensors, so gather and scatter share one
  row-mapping definition; both directions were verified bit-exact against the
  original indexing on real production ops.
- The optimization is guarded by `kv_cache.is_contiguous()`, so sliced or
  non-contiguous caches (and any backend where `aten::index` is fast) keep the
  previous code path -- the fast path is Ascend-motivated but not Ascend-specific.
- The scheduler hardening turns a crash (assert) into a logged skip: a finished
  transfer for an already-aborted request has nothing left to free or resume, and
  the connector-side `_request_filenames` entry was already consumed at
  finalization time.

## Verification

- Synthetic full-shape benchmark (gather): 34.5ms -> 0.1ms per op.
- 31/31 real production scatter ops replayed bit-exact vs the original
  three-subscript form (incl. padding slots and duplicate targets);
  ~3.8ms/op vs 91.5ms/op on `[985, 6, 128, 6144]` bf16.
- Abort-in-flight scenario no longer crashes the engine: scheduler warns
  "KV xfer finished for unknown/aborted request ...; skipping" and continues;
  connector `get_finished` returns `(False, None)` for late aborts.

## Apply

    cd /vllm-workspace/vllm && git apply vllm_ascend_flat_index_kv_xfer_20261113.patch

(基于 tag v0.30.0 / ced6857 的工作区生成; 其他基线如冲突需人工核对上下文行。)

## 备注 (按仓库 PR 规范)

- 本改动为 AI 辅助完成; 提交前需人工逐行审查, PR 中需声明 AI assistance。
- "I (a human) have written or reviewed the code" 勾选框保持未勾选。
- 上游提交建议拆分: (1) flat-index gather/scatter 性能优化, (2) aborted-request
  健壮性修复 (scheduler assert -> warning; connector pop default)。
- 传输说明: 补丁经终端 base64/分块校验和比对落盘, md5/sha256 与远端 `git diff`
  输出逐字节一致后存档。
