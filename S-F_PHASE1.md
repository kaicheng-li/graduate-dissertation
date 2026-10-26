> 历史记录：本文描述旧版本，不是当前训练配置。单步、gate、旧脚本名称及旧计算量不适用于当前 verl 流程。当前入口和行为以 [README.md](README.md) 及代码为准。

# S-F v3 阶段一：零训练 gold 似然探针

本目录是可单独上传训练机的源码快照。本文描述可选的阶段一探针：不加载 Manager、不创建优化器、不更新权重。完整训练入口为 `scripts/train_manager_sf.py`，见 [训练说明](SF_TRAINING.md)。按用户最新要求，训练可不提供阶段一报告直接开始；仅在显式提供报告时核验 GO。

## 基线选择

- 来源：本地 `baseline-lora`，固定提交 `87f13c579c3121fe9a76119ac5a15b059681954d`，未进行网络拉取。
- 唯一移植：`baseline_sd` 中 `scripts/construct_memory_bank.py` 的 JSON 异常修复，即完整 `memory` 数组只缺外层右花括号时允许恢复；动作仍由原环境校验。
- 原训练器、启动脚本、F1、检索、提示词、GRPO 优势均保持该提交原样；没有移植 S-D/S-C 优势代码。
- `baseline_origin.json` 记录原文件 SHA256 与唯一例外。测试逐文件核验，不能只看分支名称判断基线。
- 本地现有的训练/验证 Stage-0、抽取事实文件额外复制到 `data/`，不代表这些文件由该提交生成；每次探针另记输入哈希。

原规格中的旧目标目录 `baseline_sc` 不再适用，本次实现选在 `baseline_sf/memory_r1_reproduction`。原规格文件未修改。

## 实现和固定公式

入口 `scripts/probe_gold_logp.py`；实现分在 `sf_probe/`。脚本不 import 训练主循环。

原 S-A 判定采用 **4 个候选全部 reward** 的 float32 总体标准差：`std > 1e-12` 才有信号。探针目标组为：S-A 无信号、合法候选至少 2 个、同分小于 1。`[0.4,0.4,0.4,0]` 中最后一个候选非法时仍有原 S-A 信号，绝不启用辅助。

合法候选分数为 gold token 的平均条件 logP，使用自然对数并 clip 到 `[-10,0]`。离线预览固定为：

```text
u_i   = (ell_i - mean_legal(ell)) / max(range_legal(ell), 0.5)
aux_i = 0.3 * (1 - r_tied) * u_i
```

非目标组及非法候选 `aux=0`；合法分数极差不超过 `1e-6` 时也为零。探针入口只统计幅度；独立训练入口在相同评分协议下将其接入 backward。数值回归和本地验证范围见训练说明，真实 GPU 轨迹尚需训练机验证。

计分复用选定 runtime 的 `top_k_per_speaker(...,30)`、`build_answer_input` 和实际 chat 模板，按原路径左截断到 4096 prompt tokens。assistant 前缀固定 `**Answer:** `，只计 gold；不计前缀、不加 EOS、不自由生成推理。前缀和 gold 联合分词；跨边界 token 仅允许包含前缀末尾空白。模型用原加载器的冻结 NF4 base，eval/no_grad，FP32 归约，保存恢复 Python/NumPy/Torch/CUDA RNG。当前评分器针对基线 Qwen decoder/lm_head 结构。

这衡量的是固定前缀下的答案支持度，不是操作事实正确性，也不是原自由生成完整回答的概率。

## 运行前需要什么

1. epoch-1 **全部 rank** 的原始 `candidate_group` 日志，每组 G=4，每候选含 `raw_action` 或 `raw_output`、`valid`、`reward`。可兼容额外的 `actions`、`decisions`、`bank_sha256`、`prediction` 核验。
2. 对应的训练 Stage-0 states 与 timelines。默认指向本目录 `data/stage0/*train*`；若原 run 不同必须显式指定。拒绝验证/测试 states 和 timelines。
3. **产生原日志时保存的源码快照**、run manifest 或补充溯源文件，以及同 run 的本地 Answer base 模型目录。
4. GPU 环境沿用原训练环境，需 CUDA PyTorch、Transformers、bitsandbytes、sentence-transformers；核心版本参见保留的 `requirements/memfactory.txt`。该文件未列 bitsandbytes，需确认原环境已安装。阶段一不需要运行 vLLM 服务。不要为探针随意升级原训练环境。

干净 `baseline-lora` 的原训练器本身不输出这里要求的完整候选日志。本交付不会改写它来补日志。现有 S-C 原始日志可以先做历史诊断；若它的运行时代码与本目录基线不同，不能直接据此给这条新基线开训练绿灯，需在最终目标 runtime 的候选上重做探针。

本机只有 CPU，且未找到原始 `groups*.jsonl`；因此当前只完成代码与合成样例/小模型测试，没有真实三判据数值。

## 先生成并核查受控证据

以下命令在 Linux 训练机、本目录下执行；将 `/runs/sc` 换为真实旧 run。若仅有 `groups.jsonl`，对应修改通配符。所有输出目录必须是**新目录**。

```bash
python scripts/probe_gold_logp.py \
  --groups '/runs/sc/groups*.jsonl' \
  --states /runs/sc_snapshot/data/stage0/manager_states_train.jsonl \
  --timelines /runs/sc_snapshot/data/stage0/dialogue_timelines_train.json \
  --prepare-controls \
  --output-dir ./probe_outputs/sf_v3_prepare
```

这一步无需加载 Answer/检索模型，但需要 CPU PyTorch。按与评分无关的固定哈希顺序，预选最多 20 个不同问题的目标组，写出 `control_templates.jsonl`。**不按 logP 方向挑题，不在评分后换题。**

复制该文件为 `controls_reviewed.jsonl` 后逐条检查：

- `evidence_entries` 最初来自该题 evidence turns 的事实。删去无关条目，补全必要上下文/时间信息，确保剩下的每一条都属于必需证据，合起来足以支持 gold。不能因为来自证据轮就认定内容充分。
- `remove_ids` 指定从 old_bank 删除的证据；检查同义改写和跨条目重复，补全要删除的 ID。程序还会删除与新增证据文本规范化后完全相同的旧条目。
- 核查后将 `reviewed`、`sufficient_evidence_confirmed`、`duplicate_check_confirmed` 设为 `true`，并填写具体 `review_notes`。这是原方案所需的证据语义核查，不是程序能自动证明的条件。

评分时强制检查：**所有必需证据都被检索，完整文本都在截断后的最终输入中，minus 输入不再出现这些文本，两个最终输入确实不同**。不合格对会记录剔除原因，不会补换更有利的问题。少于 20 个可用问题时方向门槛为 `INCONCLUSIVE`，不能宣称通过。

## 历史指纹怎么提供

`--runtime-root` 指向原 run 的源码快照根目录（目录下有 `scripts/` 和 `memfactory/`），不一定是当前探针目录。`--run-manifest` 读原 run 元数据；不完整时用 `--run-evidence` 补充，冲突值拒绝覆盖。补充文件格式示例：

```json
{
  "sources": {
    "scripts/construct_memory_bank.py": "历史源文件SHA256",
    "memfactory/chat.py": "历史源文件SHA256",
    "memfactory/lora.py": "历史源文件SHA256",
    "memfactory/manager_env.py": "历史源文件SHA256",
    "memfactory/modules/memory_updater.py": "历史源文件SHA256",
    "memfactory/modules/memory_retriever.py": "历史源文件SHA256",
    "evaluation/metrics.py": "历史源文件SHA256"
  },
  "files": {"data_path": "原训练states文件SHA256"},
  "settings": {
    "manager_top_k": 5,
    "answer_top_k_per_speaker": 30,
    "max_prompt_tokens": 4096,
    "variance_epsilon": 1e-12,
    "num_generations": 4,
    "answer_adapter": null
  },
  "retrieval": {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu"
  },
  "group_logs": {"rank0": "原日志SHA256", "rank1": "另一rank日志SHA256"},
  "model": {
    "config.json": "原模型文件SHA256",
    "tokenizer.json": "原模型文件SHA256",
    "model-00001-of-00004.safetensors": "原权重文件SHA256"
  }
}
```

上面为**结构示例**，不可原样使用。`model` 必须完整覆盖模型目录的可见 `.json/.safetensors/.bin/.model/.txt/.jinja/.py` 文件，相对路径为键；不是只列示例里的三个文件。如果历史 runtime 还含 `memfactory/memory_runtime.py`、`retrieval.py`、`generation.py`，也必须加入相应 `sources` 哈希。检索配置需与原 run 一致，并在运行前设置对应 `MEMORY_EMBEDDING_MODEL`、`MEMORY_EMBEDDING_DEVICE`。

哈希必须来自原 run 留存文件/可追溯快照。**把今天源码的哈希填进补充文件，不会使它成为历史证据。**无法恢复原运行环境时，可以用 `--diagnostic-only` 做明确标注的诊断，最终结论始终不能为 GO。

## 执行评分

```bash
python scripts/probe_gold_logp.py \
  --groups '/runs/sc/groups*.jsonl' \
  --states /runs/sc_snapshot/data/stage0/manager_states_train.jsonl \
  --timelines /runs/sc_snapshot/data/stage0/dialogue_timelines_train.json \
  --runtime-root /runs/sc_snapshot \
  --run-manifest /runs/sc/run_manifest.json \
  --run-evidence /runs/sc/historical_evidence.json \
  --controls ./probe_outputs/controls_reviewed.jsonl \
  --model /models/Qwen2.5-7B-Instruct \
  --device cuda:0 \
  --output-dir ./probe_outputs/sf_v3_full
```

没有原 manifest 时可只提供可信补充文件；旧数据无法核验则追加 `--diagnostic-only`。冒烟可加 `--limit 8`，必须另选输出目录；部分运行一律不判 GO。

先重放全部原始候选；任一候选不一致时**剔除整组**，绝不删掉某候选再重判目标组。候选不一致比例 `>2%` 时在计分前中止。随后核验模型文件，才加载 Answer 评分；不加载 Manager。

若旧 runtime 与本次干净基线源码指纹不同，即使数值三门槛通过，报告也只给 `INCONCLUSIVE`，注明需在最终目标 runtime 上复查。

## 固定判据与产物

三项工程门槛预先固定，不用探针结果调 λ/τ：

| 判据 | 分母 | 通过条件 |
|---|---|---|
| 输入覆盖 | 重放接受的目标组 | 至少两个有效输入 token 指纹的组占比 ≥40% |
| 受控方向 | 预选且核查通过的独立问题证据对 | 至少 20 对，`ell_plus > ell_minus` 占比 ≥75% |
| 实际信号 | 有效输入互异的目标组 | 真实候选 ell 极差 >0.1 nat 的组占比 ≥50% |

成对 F1/ell 排序一致率以 50% 为随机基准、60% 为弱门槛，只比较 F1 不同的合法候选对，ell 平局记 0.5；它不阻止后续训练，但样本量与问题聚类 bootstrap 区间必须报告。全局相关仅为描述统计。Wilson 区间按组计，重复/补齐问题并非独立样本；报告另列 unique question 数，不能把门槛或区间当作理论保证。

输出包括：

- `probe_manifest.json`、`audit.json`、`replay_audit.json`：实现/协议/源码/数据/模型哈希、版本、重放异常。
- `groups.jsonl`、`examples.jsonl`：逐候选操作、F1、ell、ΔΦ、bank/实际输入双指纹、离线 aux；样例最多 10 个目标组。
- `controls.jsonl`、`control_inputs.jsonl`：证据对结果、剔除原因、最终输入文本，可人工检查检索/截断/重复证据。
- `report.json`、`report.md`：三个判据的分子分母与区间、辅助幅度分布、有效辅助信号占全部训练组的比例、耗时与显存峰值。
- 中途失败写 `error.json`，不生成伪造的成功报告；已存在输出目录拒绝覆盖。所有产物只在指定新目录，不写训练 run/checkpoint。

评分退出码：`0=GO`，`2=NO_GO/INCONCLUSIVE/执行失败`；准备模板成功返回 0，但状态为 `PREPARED_NOT_SCORED`。先看报告状态，不能把模板命令的退出码当作验收通过。

## CPU 验证

```bash
python -m unittest discover -s tests -v
python scripts/probe_gold_logp.py --help
```

新增测试覆盖：原基线文件哈希、JSON 修复、非法候选反例、辅助数值/λ 缩放/零均值、真实 BPE 的前缀边界、随机小 Qwen 的计分与完整前向对齐、RNG 保存恢复、重放和受控证据、三判据与命令行冒烟。合成测试通过只说明这些实现检查通过，不能代替真实 7B/NF4 GPU 探针，也不能证明训练收益。
