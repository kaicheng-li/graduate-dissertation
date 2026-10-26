> 历史记录：本文描述旧版本，不是当前训练配置。单步、gate、旧脚本名称及旧计算量不适用于当前 verl 流程。当前入口和行为以 [README.md](README.md) 及代码为准。

# S-F v3 审核与正式训练

本轮交付独立训练入口 `scripts/train_manager_sf.py`、启动脚本 `scripts/run_sf_training.sh`。`grpo` 是 S-A 对照；`grpo-goldaux` 是 S-F。原 `train_manager_grpo.py` 未修改。共享环境的重复目标规则已按用户要求改为同一 ID 最后一条操作生效，例外及哈希记录在 `baseline_origin.json`。所有改动位于 `baseline_sf`。

## 重复目标 ID 的执行规则（2026-10-20）

同一输出重复引用已有 ID 时，仅执行最后一条 UPDATE、DELETE 或 NONE，基于操作前的记忆库计算最终状态。例如 DELETE 后 UPDATE 保留最后的更新，UPDATE 后 NONE 保留原始记忆。ADD 仍逐条新增，由环境分配不同 ID；有效修改数在合并重复目标后计入最多 5 条的限制。所有原始条目仍须格式正确、引用可见 ID；非法条目仍原子拒绝。

训练、single-step 验证、roll-forward 验证和 Answer 数据构建共用此规则。使用本目录 `sa` 与 `sf` 入口比较时规则一致；旧版本 S-A 使用拒绝重复目标的规则，需要在实验记录中注明差异。

此修改不增加模型调用。已有 Python 进程不会自动采用新规则，需要在下次启动时加载。不要在同一实验中途切换规则；续训仍检查源码合同。补丁不修改 JSON 解析器：其已接受说明/代码块中的完整 memory JSON，且仅修复完整 memory 数组缺少外层右花括号的情形。其他 parse 报错须结合保存的 raw 输出定位，不能从错误摘要判断是否因生成截断。

## 直接开始完整 S-F 训练

### 辅助参数调整（2026-10-20）

当前默认 λ 从 0.3 降至 **0.1**，新增 `--aux-min-range` 默认 **0.05 nat**，替代原先写死的 1e-6 极差阈值；τ 仍为 0.5。仅合法候选似然极差**严格大于**该阈值才启用辅助更新。阈值是实验参数，不保证排序正确；全对组和原 F1 有区分度的组保持原局部计算路径。提高阈值仍须先计算似然，不能省掉这部分前向；只减少本 rank 的辅助 backward，全局 optimizer 是否 step 仍由所有 rank 的信号共同决定。

可在启动命令末尾覆盖，例如 `bash scripts/run_sf_training.sh sf --aux-lambda 0.1 --aux-min-range 0.05`。λ 和极差阈值允许非负有限数，λ=0 关闭辅助评分；两项进入 manifest/checkpoint 配置，启动摘要输出实际值，逐组日志保存 `ell_range`。本次调整应使用新输出目录重新训练；不接旧参数 checkpoint。其他训练参数、验证及选模流程未改。这是两项参数同时调整的试验，不能据此单独归因某一项的收益。

当前入口不再强制 precheck、阶段一探针、GO 或冒烟，也不需要先重跑 S-A。启动时仍自动核验真实训练/验证数据。以下命令在 Linux 的 `memory_r1_reproduction` 目录执行，沿用已有 CUDA/QLoRA 和 embedding 环境：

```bash
export MANAGER_MODEL=/root/autodl-tmp/Qwen2.5-7B-Instruct
export ANSWER_MODEL="$MANAGER_MODEL"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC=4

PRECHECK= PROBE_REPORT= RESUME_FROM= \
TRAIN_OUTPUT=output/sf_l010_range005_seed42 \
bash scripts/run_sf_training.sh sf --aux-lambda 0.1 --aux-min-range 0.05
```

空的报告变量用于覆盖终端里之前遗留的路径。输出目录须为空或尚不存在。该命令从 base 模型最多训练五轮，包含验证、选模、实时日志和 checkpoint；不是 smoke。当前默认 λ=.1、τ=.5、极差阈值=.05 nat；优势公式、F1 奖励和原 epoch-1 训练门槛保持原设计。没有执行探针时，仅能报告训练与评测结果，不能声称探针判据已通过。

`PRECHECK` 和 `PROBE_REPORT` 现在均为可选项；显式指定时仍严格核验，文件缺失或不合格会报错。未提供时 manifest 对应哈希为 null。断点续训应保持与初次运行相同的可选报告配置。旧入口 `train_manager_grpo.py` 保持原行为。

启动日志现在逐 rank 输出 `sf-startup` 的 collectives、output_check、data_audit、contract、manager_load 和 answer_load 阶段。模型指纹逐文件输出 start/done，读取大文件时每约五秒输出已读字节数。启动阶段超过 120 秒会自动打印 Python 调用栈，并每 120 秒重复；这是诊断信息，不会终止训练，模型加载完成后停止打印。非空输出目录会在模型指纹计算之前报错；重新从头启动须选择新目录，不能重复使用已产生 manifest 的目录。

## 数据审核结果

`python scripts/audit_sf_data.py` 已在随附真实数据上通过，报告为 `data/stage0/sf_data_audit.json`。

| 划分 | 对话 | 单步题数 | 不同目标轮 | 时间线轮数 | 时间线题数 |
|---|---|---:|---:|---:|---:|
| train | conv-26 | 150 | 113 | 419 | 152 |
| validation | conv-30 | 81 | 66 | 369 | 81 |

逐题核对 question/gold/category、原 question_id、证据到目标轮绑定、参与者、当前事实及说话人/时间戳；逐条核对 old_bank 与目标轮之前的完整事实库完全一致，排除当前轮/未来轮泄漏。训练和验证对话无交集。

训练时间线的题 30、46 没有证据，因此只有 150 条可绑定单步训练状态；roll-forward 仍使用全部 152 题。验证题 1、78 的目标轮没有抽取事实，和时间线一致，保留全部 81 题评测。没有擅自补事实、删除这两题或改变分母。

正式入口在加载 7B 模型之前重复数据核验。仅在显式提供 Stage-0.5 precheck 时，核验其全部 QA 身份、gold、目标轮、重算 F1 和原 15 题门槛，防止误用其他数据的报告。

## 算法与方案的对应

每题从同一 old_bank 生成 G=4 个独立 Manager 候选；合法候选各生成一次冻结 Answer 答案，以原连续 token F1 为奖励，非法候选 reward=0。Manager prompt 只有当前事实和检索记忆，gold 不进入 Manager。

先按原 S-A 的设备/float32/总体标准差计算：

```text
signal_SA = std(all_four_rewards, unbiased=False) > 1e-12
A_SA      = (r - mean(r)) / (std(r) + 1e-6), if signal_SA; else 0
```

仅 `grpo-goldaux` 且 λ>0、S-A 无信号、至少两个合法候选、同分小于 1 时评分。冻结 Answer 的输入复用该候选生成答案时的实际检索 prompt，并逐候选核对 chat 模板和截断后的 token 指纹。评分和探针共用 `sf_probe/scoring.py`：固定 `**Answer:** ` 后 teacher forcing gold，只计 gold，平均自然 logP，clip 到 [-10,0]，不自回归生成、不加 EOS，保存恢复 RNG。

```text
u_i = (ell_i - mean_legal(ell)) / max(range_legal(ell), 0.5)
A_i = aux_lambda * (1-r_tied) * u_i   # 默认 aux_lambda=0.1
```

不对辅助优势再次标准化。合法 ell 极差≤aux_min_range（默认 0.05 nat）时仍跳过；全对组、F1 可分组维持 S-A。非法候选不获得辅助优势；在新启用的辅助组内，也不对非法候选额外执行 KL/backward，损失按合法非空候选数取均值。原 S-A 有信号组的非法候选处理和 KL 路径完全沿用原实现。

例如 r=[0,0,0,0]、ell=[-1,-2,-3,-4]，默认 A=[.05,.016667,-.016667,-.05]；r=[.4,.4,.4,0]（最后非法）已有 S-A 信号，完全不启用似然。

λ=0 不构造 scorer、不调用似然评分，跳步及优化器路径退回 S-A。测试验证多组参数、AdamW 状态、scheduler 和 did_step 序列逐值相同。不能据此宣称真实 NF4、多 GPU 全轨迹已验证。

多卡保留原 CPU 投票和梯度平均：本 rank 无信号时仍参加全局更新同步。epoch-1 的原 F1 信号比例≥.15、roll-forward F1 不低于 epoch-0 的门槛保持一致；辅助组比例独立报告，没有用它偷换原门槛。

## 可选的预验证流程

以下流程用于训练前研究诊断，不是直接训练的前置要求；不需要预验证时使用本文开头的直接训练命令。

### 先区分训练数据和启动报告

| 文件 | 用途 | 是否进入训练计算 |
|---|---|---|
| `manager_states_train.jsonl` | 单步训练样本：问题、gold、当前事实、old_bank | 是；gold 仅用于评分，不进入 Manager prompt |
| `dialogue_timelines_*.json`、验证 states | 时间线评测、验证和选模 | 用于评测 |
| `precheck_train_f1.json` | 可选的原 S-A Stage-0.5 校验报告 | 否；显式提供时校验，哈希写入运行合同 |
| S-F 探针 `report.json` | 可选的辅助信号可用性校验报告 | 否；显式提供时核验 GO 和指纹 |

precheck 不是额外训练集，不从它抽样，也不使用其中的预测/F1 计算训练 reward、优势或 loss。它检查冻结 Answer 在 old_bank 与 old_bank+当前事实两个条件下是否存在可改善的题目：150 题各生成两次，按分层匹配规则从错变对的题数至少为 15 才通过；这不是“15 题 F1 上升”或严格 EM 门槛。

当前交付的 `sa`、`sf`、`smoke` 均可不提供 precheck 和探针报告。原 `train_manager_grpo.py` 的强制 precheck 要求未修改。旧版启动报错是报告依赖缺失，并非训练样本缺失；新版已去掉默认报告路径。

以下是 Linux GPU 指令，从此目录运行。沿用复现 S-A 的已可用 CUDA/QLoRA 环境，需有 `peft`、`bitsandbytes` 和本地 embedding 模型；原 requirements 没列全这两项，不要直接用本机 CPU 测试环境跑 NF4。S-A、探针和 S-F 使用同一环境、同一 base 模型和同一检索设置。

```bash
export MANAGER_MODEL=/root/autodl-tmp/Qwen2.5-7B-Instruct
export ANSWER_MODEL="$MANAGER_MODEL"
export MEMORY_EMBEDDING_MODEL=/home/models/all-MiniLM-L6-v2
export MEMORY_EMBEDDING_DEVICE=cpu
export NPROC=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/audit_sf_data.py
```

### 准备或复用 precheck（一次性）

若已有相同数据、冻结 Answer 和检索配置下的 S-A precheck，直接复用，无需重新生成。文件名不决定有效性；例如旧文件叫 `precheck_train.json`，也可指向它，入口仍会验证报告内容：

```bash
export PRECHECK=/实际已有报告路径/precheck_train_f1.json
```

若旧报告保留了预测、但使用旧评分口径，可仅在 CPU 重评分，不重新加载模型生成：

```bash
python scripts/precheck_counterfactual.py \
  --rescore-input /实际已有报告路径/precheck_train.json \
  --output data/stage0/precheck_train_f1.json
export PRECHECK=data/stage0/precheck_train_f1.json
```

本地交付不含真实 precheck 报告。选择做预验证且没有可复用报告时，可执行原基线入口生成，不能手填通过；此命令单独占一张卡，不要和四卡训练同时运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/precheck_counterfactual.py \
  --states data/stage0/manager_states_train.jsonl \
  --model-path "$ANSWER_MODEL" --device cuda:0 \
  --output data/stage0/precheck_train_f1.json
```

### 准备 S-A 候选日志和 S-F 探针

已有完整候选日志及可核验同期溯源信息时，先按 `S-F_PHASE1.md` 核验并复用，不要求重跑整个 S-A。缺少这些材料时，下面命令跑完整五轮 S-A 对照：

```bash
TRAIN_OUTPUT=output/sa_seed42 bash scripts/run_sf_training.sh sa
```

若目前只为阶段一探针采集 epoch-1 日志，可改为下面的一轮命令；这不能替代最终完整训练的 S-A 对照：

```bash
TRAIN_OUTPUT=output/sa_probe_seed42 bash scripts/run_sf_training.sh sa --epochs 1
```

后续命令中的 `output/sa_seed42` 应替换为实际日志目录。

新 S-A 入口每个 rank 输出 `groups_rank_N.jsonl`；每次 checkpoint 同步更新 `run_manifest.json` 的日志哈希。旧基线没有这些完整日志时，不能用衍生统计代替候选重放。四卡将 150 题补齐为 152 组（每卡 38 组），保留原 S-A 的 padding 语义；报告同时记录组数与独立问题数。

用 S-A epoch-1 的全 rank 日志准备受控证据对；输出目录要新建且放在训练输出目录之外：

```bash
python scripts/probe_gold_logp.py \
  --groups 'output/sa_seed42/groups_rank_*.jsonl' \
  --run-manifest output/sa_seed42/run_manifest.json \
  --prepare-controls --output-dir probe_outputs/sf_controls
```

按 `S-F_PHASE1.md` 核查并复制修改模板为 `probe_outputs/controls_reviewed.jsonl`：确认足够证据、无重复证据，填写 review 字段；评分时还会自动核对最终输入可见性。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/probe_gold_logp.py \
  --groups 'output/sa_seed42/groups_rank_*.jsonl' \
  --run-manifest output/sa_seed42/run_manifest.json \
  --model "$ANSWER_MODEL" --device cuda:0 \
  --controls probe_outputs/controls_reviewed.jsonl \
  --output-dir probe_outputs/sf_v3
```

也可单独冒烟，实测生成、似然和反向时间及实际显存；这一步是可选项：

```bash
TRAIN_OUTPUT=output/sf_smoke bash scripts/run_sf_training.sh smoke
```

`smoke` 每卡只跑 3 组、不做验证，允许工程运行时尚无 GO 报告；它不会产出探针 GO。自然样本未覆盖某类组时，不能宣称已经实测该分支成本；检查逐组日志中的 `sa_signal/aux_eligible/aux_active`。冒烟 checkpoint 与正式训练模式/组限合同不同，不能接着当正式训练起点。

正式 S-F 从同一 base 模型开始，当前默认 λ=.1、τ=.5、极差阈值=.05 nat：

```bash
PROBE_REPORT=probe_outputs/sf_v3/report.json \
TRAIN_OUTPUT=output/sf_seed42 bash scripts/run_sf_training.sh sf
```

上面是显式提供报告的训练方式：入口会核验 GO 报告、非局部运行、全部训练题覆盖、数据/评分源码/协议/模型/检索指纹一致。探针必须提供 manifest 绑定的全部 rank 日志；若日志有生成时的实际 Answer 输入 token 指纹，重放还会逐候选核对，发生不一致立即中止该次探针，不能用漂移输入获得 GO。没有报告也可使用本文开头的直接训练方式，但不能声称探针判据已通过。原研究规格文件未改；本次仅按用户要求解除训练入口的强制报告依赖。

断点续训示例（保持同一个输出目录、同一参数和卡数）：

```bash
PROBE_REPORT=probe_outputs/sf_v3/report.json \
TRAIN_OUTPUT=output/sf_seed42 \
RESUME_FROM=output/sf_seed42/checkpoints/epoch_1_group_10 \
bash scripts/run_sf_training.sh sf
```

checkpoint 保存 LoRA、optimizer/scheduler、每 rank RNG、epoch/组位置和完整运行合同。跨模式、λ/τ、协议、源码、数据、模型或卡数变化会拒绝 resume；不兼容旧基线 checkpoint，避免误接。若崩溃后日志多于所选 checkpoint，仅裁掉 checkpoint 之后的记录，并保留 `.before_resume` 备份，防止同一候选组重复计数。每 epoch 的汇总进度日志可能仍保留旧尝试，探针只读取裁好的候选日志。

## 输出与验证范围

- `groups_rank_N.jsonl`：原始 Manager 文本、解析决策、合法性、预测/F1、bank/实际 Answer 输入指纹、ell/u/最终优势、跳步和三段耗时。
- `epoch_N_training_summary.json`：原 F1 信号组、辅助目标/实际启用/输入坍缩组数、比例、评分耗时及新增组反向耗时（各 rank 累加，不能当 wall time）。
- `run_manifest.json`、`data_audit.json`：源文件、数据、配置、依赖版本、模型、检索、日志哈希。
- `epoch_N/`、`checkpoints/`、`best_checkpoint.json`：沿用基线验证与选模。

本机完整执行 `python -m unittest discover -s tests -v`：58 项中 57 项通过、1 项跳过，日志为 `test_results_sf.txt`。覆盖真实数据 CPU 审计、原始 S-A 文件哈希核验、真实小 Qwen 的 gold 位置/完整前向对照、S-F 分支和更新测试、候选重放、训练驱动和续训。测试小模型/替代模型不等同于真实 7B/NF4 训练；本机无 CUDA，尚未执行真实 GPU 探针、NF4 加载、显存测试或正式训练，不承诺收益或固定的额外耗时比例。

启动脚本已通过 Git Bash 的 `bash -n` 语法检查。两进程梯度平均/联合跳步测试在本机 Windows CPU PyTorch 上报 Gloo `unsupported gloo device`，因此 Windows 跳过该项；Linux 训练机可执行 `python -m unittest discover -s tests -p test_sf_training.py -v` 核验。此处没有把未执行的多卡测试计为通过。
