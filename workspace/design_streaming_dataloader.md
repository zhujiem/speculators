# Backend 无关流式 StreamingDataLoader 设计(方案 A:manifest 随 vLLM 响应回流)

> 分支:`speculators_ascend`
> 前置文档:`design_tq_hidden_states_backend.md`(TQ backend 传输层,相互独立又可组合)
> 状态:已实现(实现备注见 §2.3/§4 修订)
> 日期:2026-11-17

## 0. 定位与动机

现有在线基线(`VllmHiddenStatesSource` + torch DataLoader):

```
SpeculatorsDataset(固定 records)
  → torch DataLoader + MultipackDistributedBatchSamplerV2(预先按 approx_lengths 划批)
  → worker 里 VllmHiddenStatesSource.get_batch(逐 record 发 chat 请求、等响应、
     经 HiddenStatesTransfer.get_generated 取回 hidden states)
```

问题:样本的获取顺序被固定数据集和预划批**串行化**——每个 batch 内的请求全部完成才能出批,batch 之间又是 DataLoader 顺序迭代;vLLM 并发度受 batch 边界制约,慢样本(长序列)拖死整批。

流式 loader 改为:**请求与响应解耦、响应到达即增量组批、批关闭即拉取**。改动集中在新增一个 loader 组件,替换 torch DataLoader 的批生成与取数路径;训练循环、损失、checkpoint 均不变。

本方案采用**发现方式 A**(manifest 随 vLLM 响应回流),因此:

- **backend 完全无关**——handle 由 vLLM 响应体携带,`file / http / mooncake / (未来的) transfer_queue` 全部可用,存储层不需要任何 list/ready 端点;
- 是对现有基线改动最小的流式化路径(方案 B 的轮询+就绪标记协议留作后续演进)。

## 1. 架构

### 1.1 数据流总图

```
┌─────────────────────── 训练进程 rank r (torchrun DP×N) ────────────────────────────┐
│                                                                                   │
│  dataset.records (ArrowDataset,顺序已知;输入 ids 现成,无需生成)                 │
│        │ 按极大前缀发送批提交(X-Batch-Id,占用 ≤ max_queued_batches)                │
│        ▼                                                                           │
│  ┌──────────────────────────────────────────────┐                                 │
│  │ StreamingDataLoader                           │                                 │
│  │                                               │                                 │
│  │  ① 发请求     ResponseManifestSource          │   async chat 请求池             │
│  │     .request(record) ────────────────────────┼──────────────▶ vLLM             │
│  │                    ◀─────────────────────────┼── 响应体含 transfer_manifest     │
│  │                                               │   (handle + tensors spec)       │
│  │  ② 增量组批   StreamingPacker                 │                                 │
│  │     按 approx_length 贪心装箱 batch_max_len;  │                                 │
│  │     满箱即闭批 → 提交取数                      │                                 │
│  │                                               │                                 │
│  │  ③ 拉样本     HiddenStatesTransfer(backend    │   file / http / mooncake / tq   │
│  │     无关,现有接口) .get_generated(manifest) ──┼──────────────▶ 存储层           │
│  │     .delete(handle)  (读后即删,幂等)          │                                 │
│  │                                               │                                 │
│  │  ④ 组 BatchType(复用 SpeculatorsDataset 的   │                                 │
│  │     preprocess/collate 路径) → yield          │                                 │
│  └──────────────────────────────────────────────┘                                 │
│        ▼                                                                           │
│  训练循环(不变)                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘

rank 间关系:每个 DP rank 独立跑 ①-④,rank 各自消费 dataset 的
Multipack 均分切片(records 切分沿用现有分布式采样规则),
无跨 rank 通信、无广播——方案 A 下这是天然成立的。
```

### 1.2 与基线的对照(改动面)

| 组件 | 基线 | 流式 | 改动 |
| --- | --- | --- | --- |
| 批划分 | `MultipackDistributedBatchSamplerV2`(离线全局) | `StreamingPacker`(在线贪心,rank 局部) | 新写,装箱规则复用 |
| 请求发送 | `VllmHiddenStatesSource.get_batch`(batch 内同步栅栏) | `ResponseManifestSource.request`(异步池,无栅栏) | 包装现有 `generate_hidden_states_async` |
| 取数 | `HiddenStatesTransfer.get_generated` | 同一接口 | **零改动**(backend 无关的原因) |
| batch 组装 | `SpeculatorsDataset.__getitem__ + collate` | 同一 preprocess 函数 | 提取复用 |
| 训练循环/损失/ckpt | — | — | 零改动 |
| 入口开关 | — | `--dataloader {torch, streaming}` | `create_train_val_loaders` 加分支 |

## 2. 组件与接口

新文件:`src/speculators/train/streaming_loader.py`(单文件,~350 行,避免新抽象)。

### 2.1 `ManifestSource`(协议,发现方式 A 的实现点)

```python
class ManifestSource(Protocol):
    """把一条训练记录变成 vLLM 请求,返回带 manifest 的回执。"""

    async def request(self, record: DatasetRecord) -> GenerationReceipt:
        """发送 chat 请求(复用 vllm_client.generate_hidden_states_async
        与 build_client_item),响应完成即返回;receipt.hs_transfer_params
        即 vLLM 回流的 transfer_manifest(handle + tensors spec)。"""

    async def as_completed(self) -> AsyncIterator[tuple[DatasetRecord, GenerationReceipt]]:
        """按响应完成顺序(而非提交顺序)产出 (record, receipt)。"""

    async def close(self) -> None: ...


class ResponseManifestSource:
    """发现方式 A 的唯一实现:manifest 随响应回流。"""

    def __init__(self, *, endpoint: str, model: str,
                  request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES): ...
```

将来若演进到方案 B(轮询存储),只需新增 `PollingManifestSource` 实现同一协议,loader 其余部分不动——这是把发现方式收敛到这个协议的意义。

### 2.2 `StreamingPacker`(在线装箱)

```python
class StreamingPacker:
    """响应到达即装箱;箱满闭合,产出本批的 (record, receipt) 列表。

    装箱规则与 MultipackDistributedBatchSamplerV2 的 rank 局部行为
    一致:按 approx_length 贪心填入当前箱,batch_max_length 截止;
    样本超长单独成箱(与现有 drop/溢出策略一致)。
    """

    def __init__(self, *, batch_max_length: int,
                 max_batches: int | None = None): ...

    def add(self, record: DatasetRecord, receipt: GenerationReceipt) -> list[list[...]] | None:
        """加入一个就绪样本;若当前箱闭合,返回该批内容。"""

    def flush(self) -> list[list[...]]:
        """末尾清箱(drop_last 语义沿用现有参数)。"""
```

与离线 Multipack 的差异声明:装箱从"全局排序后最优"退化为"到达顺序贪心",填充率略降;换来的是慢请求不再阻塞整批。装箱只用到 `approx_length`(来自 record,请求发出前已知),**不依赖响应内容**,所以请求飞行途中即可预占箱容。

### 2.2a 发送批与训练批解耦(混合方案,已实现)

上文的 `StreamingPacker` 产出**训练批**(yield batch)。请求发送侧另有**发送批**(send batch)概念,两者解耦:

- **发送批 = rank 切片的极大 token 前缀**(`_send_batches`):顺序扫描切片,累计 `batch_max_length` 预算,装不下下一颗即切刀。确定性、离线可算(装箱只用 `approx_length`),所以**发请求前批成员已知**。
- 同一发送批的所有请求携带同一 `X-Batch-Id` 头(`generate(record, extra_headers={"X-Batch-Id": ...})`)。该头被基线 b76ab6a 的 batch-affinity 负载均衡用于把同批请求钉到同一 vLLM 实例——发送批 token 预算 = vLLM `max_num_batched_tokens`,恰好填满一个 engine step。
- **批级 in-flight gate**(`in_flight_batches`,取 CLI `--prefetch-factor`,默认 4):在飞发送批数受限。gate 不仅是限流,更是 co-batch 成立的必要条件——同批请求须同窗排队才会被 vLLM 合入一个 step;每实例 1-2 批是 b76ab6a 的原语义。请求级无独立闸——与 LB 的槽位口径一致:一个发送批占一个并发位,槽位持有期 = 该批最长请求,在飞请求数由批闸 × 批内样本数隐式决定。
- **管道深度门控**(`max_queued_batches`,默认 4,CLI `--max-queued-batches`):提交与消费进度耦合。管道各站按本站批型计数——在飞发送批 + 开口的 packer 箱 + 已闭箱待取数的批(`fetch_pending` 计数器) + 成品队列(`qsize`)——占用达到预算即停止发新批。训练停顿时占用不再下降、请求停发,transfer 存储不再堆积、vLLM 不再为无人消费的样本空算;训练恢复后占用下降、自动补批(阻塞且在飞为空时 50ms 轮询重查,避免空集 await)。计数无重叠(future 在 put 成功后才 done)。下限校验 ≥2(开口箱 1 + 在飞 1 否则自锁)。
- **接收侧忽略发送批边界**:响应到达后按到达序进 `StreamingPacker` 重组训练批。慢样本只影响它落入的训练批,已到达样本自行成批先走——队头阻塞保持为零。稳态下(同批 co-scheduled,响应成簇到达)训练批 ≈ 发送批,乱序时自动纠偏。
- 极大前缀 = 贪心装箱家族的 Next-Fit(单开口箱、不回头)。与基线 LPT 的关系:基线装箱 = "洗牌 → 二分选块(顶满 replicas×预算)→ 块内降序 LPT(均衡分给 replicas 个 rank)";发送批 = 同一循环在 replicas=1 的特例——二分塌缩为 cumsum 查表(单箱可行=前缀和达标),LPT 塌缩为恒等(单箱无均衡对象)。基线排序/均衡服务的"多 rank 同步出批"需求,流式各 rank 独立流水本就不存在。
- 续跳时从切片减去消费位图后重新极大前缀切分,边界与崩溃前略偏移——发送批只是传输提示,不重不漏由位图保证。

### 2.3 `StreamingDataLoader`

**对外契约(抄 verl 的 `StatefulPromptDataLoader` 契约式设计,契约而非继承)**:

```python
class StreamingDataLoader:
    """backend 无关的流式 loader。

    契约(verl/utils/dataset/dataloader.py 同构):
      __iter__            必需:yield BatchType
      state_dict          可选:序列化消费状态(位图续跳,见 §4.1)
      load_state_dict     可选:恢复消费状态
      close               可选:取消在飞请求、shutdown 线程池
    另定义 DataLoaderExhausted:切片耗尽且无法组出下一个完整批时显式抛出。
    """

    def __init__(
        self,
        records: Sequence[DatasetRecord],      # 本 rank 的均分切片
        source: ManifestSource,                # 方式 A:ResponseManifestSource
        transfer: HiddenStatesTransfer,        # 任意已注册 backend
        preprocess: Callable[[DatasetRecord, dict[str, Tensor]], dict],
        *,
        batch_max_length: int,
        rank: int, world_size: int,
        in_flight_batches: int = 2,            # 在飞发送批数(co-batch gate)
        max_queued_batches: int = 4,           # 管道深度预算(批数,耦合消费)
        fetch_workers: int = 4,                # get_generated 线程池
        drop_last: bool = True,
        seed: int = 0,
    ): ...

    def __iter__(self) -> Iterator[BatchType]:
        # 实现形态:IterableDataset 内核 + torch DataLoader 外壳
        #   内核 __iter__ 里跑 asyncio 事件循环驱动:
        #   1) 从本 rank 切片的极大前缀发送批发请求(同批同一 X-Batch-Id)
        #   2) as_completed 到达 → packer.add;闭合的批提交取数
        #      (fetch_workers 线程池调 transfer.get_generated,完成即 delete)
        #   3) 双重门控补批:管道占用(max_queued_batches,批数)/在飞发送批
        #      (in_flight_batches,LB 式一批一槽)
        #   4) 取数完成的批 → preprocess → yield BatchType
        # 失败处理:请求/取数失败 → 按现有 on_missing 策略(skip/warn/raise),
        #          error 样本记日志后跳过并计数

    def state_dict(self) -> dict: ...          # 位图续跳(§4.1)
    def load_state_dict(self, state: dict) -> None: ...
    def close(self) -> None: ...

    def __len__(self) -> int:  # 由切片大小与平均装箱估算,供日志/进度用
```

> 实现备注(2026-11-17 落地时修订):`DataLoaderExhausted` 未实现——整批失败的可见性
> 由 `error_records`/`RecoveryMetadata` 与日志承担,迭代终止统一走 `StopIteration`,
> 不要求 trainer 增加捕获逻辑;`ManifestSource` 协议也未单独立抽象,由
> `VllmHiddenStatesSource` 的公共 `generate()` 直接充当(§7 的 simplicity-first)。

**实现形态(落地修订:独立类,非 IterableDataset+torch 外壳)**:落地为独立类 + 自管
后台事件循环线程 + 有界 `queue.Queue`(≈80 行,换来零凿穿)。放弃 torch 外壳的原因:
①外壳的价值主要在 `num_workers>0` 多进程预取,而内核持有 AsyncOpenAI client/事件循环,
不可 pickle,只能 `num_workers=0`,此时外壳只剩 pin_memory 线程;②trainer 直接访问
`train_loader.batch_sampler`(`set_epoch`/fast-skip),iterable 路径下 torch DataLoader 的
`batch_sampler` 恒为 None,需要子类化凿穿;③`state_dict` 同样要凿穿转发。生态对照:
SpecForge 在线 `FeatureDataLoader`、verl-SpeCo 均为独立类,流式路径无一复用 torch 外壳。
`_EpochController` shim 提供 `set_epoch`(换 epoch 时清消费位图)。

取数细节:`get_generated`/`delete` 走 `HiddenStatesTransfer` 现有接口;HTTP 的 GET 本身读后即删、mooncake/tq 由 `delete` 显式删、file 删文件——各 backend 语义已经统一,**loader 不感知存储形态**。

### 2.4 入口接线

```python
# dataloader.py
def create_train_val_loaders(*, dataloader: str = "torch", ...):
    if dataloader == "streaming":
        train_loader = StreamingDataLoader(...)   # 仅训练集;val 保持 torch 路径
    else:
        ...  # 现路径不动
```

CLI:`--dataloader {torch, streaming}`(默认 torch)。val loader 暂不流式化(val 小且要求确定性,torch 路径更合适)。

## 3. 时序与并发

```
时间 ─▶
请求:  r1 ────────┐ r2 ──────┐ r3 ───────────────┐ r4 ───┐ ...
响应:              ◀── m1     ◀── m2               ◀──────── m3
装箱:              [m1+        m2 → batch#1 闭]    [m3+ ...
取数:               batch#1: get(m1),get(m2) ──▶ yield#1
训练:                                            step#1 训练中 ∥ batch#2 取数
```

- 关键不变量:**在飞发送批数 ≤ in_flight_batches**(LB 式一批一槽,隐式约束在飞请求数 = 批数 × 批内样本数),取数线程池深度 = fetch_workers,两者解耦请求与拉取;
- 慢请求不阻塞:箱 #1 闭合即取数 yield,慢的 r3 只是晚到晚装箱;
- 背压:yield 慢(训练 step 长)→ 内部队列满 → 自动暂停补充新请求(水位控制),vLLM 压力受控。

## 4. 失败模式与边界

| 情形 | 处理 |
| --- | --- |
| on_missing(四值,流式下为失败策略) | **generate**:与基线完全同链——复用 source 的 `GenerationRecoveryGuard`(同一份 `generation_validation_retries` 配置,默认 2 次重试/3 次尝试;取数侧同预算),耗尽 → `SampleUnavailable(counts_as_failure, consecutive, fatal)` 随批进 collate → trainer 分布式断路器;**skip**:降级不记日志;**warn**:降级+逐样本告警;**raise**:首次失败即中止上抛。降级永远走结构化结果进批(error_records 可观测),不静默丢 |
| manifest 到手但 get_generated 返回 None | 同上策略:取数失败在 generate 模式重试一次,其余按 skip/warn/raise 处理,转 `SampleUnavailable` 计数入批 |
| 响应成功但存储里样本缺失(error 标记) | 同上;HTTP/mooncake 的 put_error 标记路径已存在 |
| rank 间不均衡 | stride 切片(`shuffled[rank::world]`,随机排列下期望均衡)。不用 Multipack LPT 均分:LPT 均衡 token,但流式闭批由到达触发、批数本就漂移,token 均衡换不回批数对齐;瓶颈在共享 vLLM;超长样本已被切片过滤,N/W 大时 stride 相对差 <1% |
| 断点续跑 | **样本级续跳**(详见 §4.1);trainer 侧已接线:`_save_training_state` 顺带写 per-rank `loader_state_rank{r}.json`(hasattr 门控),`setup_trainer` 恢复;`set_epoch` 换 epoch 清位图 |
| 进程退出 | `close()`:取消在飞请求、shutdown 线程池、close source |
| 末箱 | 基线 Multipack 无 drop_last(恒产出尾部不满箱),流式保持一致 |

### 4.1 断点续训:样本级续跳

流式的批按响应到达顺序装箱,**批边界不确定**,因此恢复语义是"样本不重复"而非"step 级复现":

| 语义 | 基线(torch loader) | 流式 |
| --- | --- |
| 精确复现(恢复后第 k+1 步批组成与崩溃前一致) | ✅ 离线确定装箱 | ❌ 到达顺序贪心装箱,批序列不可复现 |
| 样本不重复(已训 record 不再训) | ✅ | ✅ 实现(本节) |

对训练效果而言样本不重复已足够(批组成差异等价于 shuffle 扰动);严格复现式调试可用可选的"重放模式"(见下)。

**实现**(约 30 行,契约挂 §2.3 的 `state_dict`/`load_state_dict`,确定性根基:rank 切片与 record 集合由 Multipack 均分预先确定,不确定的只是装箱/消费顺序):

```
checkpoint:随常规 ckpt 持久化 per-rank 已消费 record index 位图
           (切片数万条 → 几 KB,随 step 周期性 flush)
恢复:      loader 从切片的未消费子集启动(切片本来就是构造参数,
           无需模拟采样器游标),跳过位图已训 record;
           step 计数、LR 调度从 ckpt 恢复(与现状一致)
```

**崩溃窗口的代价**:读后即删意味着崩溃瞬间"在飞/在队列但未训"的样本
(≈ `in_flight_batches × 批内样本数 + 队列深度`,几十个)请求作废、需重新生成(分钟级);
存储里可能已写但被跳过的残留由恢复时 `delete` 清理,或靠 HTTP FIFO 逐出 /
TQ 任务级清理兜底。已训过的样本无需重新生成。"免重生成续训"是 backend 层
的事(file 天然持久;http/tq 加持久 journal),与 loader 无关。

**可选:重放模式**(非首版)。恢复后的首个 epoch 用 `--streaming-replay`:
装箱改为按 record 顺序(放弃到达顺序),可与崩溃前的批序列逐步对齐,用于
严格复现调试;代价是装箱并行度略降。

## 5. 测试计划

- `tests/train/test_streaming_loader.py`:
  - mock `ManifestSource`(可控到达顺序/延迟/失败)+ mock `HiddenStatesTransfer`;
  - 验证:装箱正确性(与 Multipack 相同输入下批内长度约束)、到达乱序不丢不重、失败样本 skip、背压(慢消费时在飞发送批不超 in_flight_batches)、close 幂等;
  - yield 的 BatchType 与 torch 路径对同一批 record 的输出逐字段一致(黄金对照);
  - 断点续跳:mock 崩溃点(消费位图截断)→ 恢复后产出 batch 只含未消费 record,且样本集合不重不漏。
- 集成冒烟:WSL 本地 fake vLLM(aiohttp 桩返回 manifest)+ `file` backend 小样本端到端;随后 Ascend 上以 `http` backend 走 `workspace/train_glm5.3_dspark` step2/step3 实测。
- backend 覆盖:file / http 参与单测,mooncake/tq 后续环境就绪后补。

## 6. 与 TQ backend 设计的关系

- 本文档与 `design_tq_hidden_states_backend.md` **正交**:前者定义 loader,后者定义传输层;组合点即 `HiddenStatesTransfer` 接口;
- 先落地本设计(方案 A + 现有 http backend)即可验证流式收益;TQ 传输层就绪后作为 backend 直接换用,loader 零改动;
- 方案 B(轮询发现)若将来需要,新增 `PollingManifestSource` 即可,`StreamingDataLoader`/`StreamingPacker` 不变。

## 7. 实施约定

- 单个 commit(约 +250 行新文件 + dataloader.py/train.py 开关接线);
- commit 前改动清单先交用户 review;单一 `Signed-off-by: zhujiem <7032227+zhujiem@users.noreply.github.com>`;
- 遵循 simplicity-first:不引入多余抽象;`ManifestSource` 协议在方案 B 有明确预期时保留。
