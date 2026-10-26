> 历史记录：本文描述旧版本，不是当前训练配置。单步、gate、旧脚本名称及旧计算量不适用于当前 verl 流程。当前入口和行为以 [README.md](README.md) 及代码为准。

# S-F 总计算量复核（2026-10-20）

口径：当前 `scripts/run_sf_training.sh sf`，4 rank、G=4、5 epochs，从头运行；PRECHECK、PROBE_REPORT、RESUME_FROM 为空。此文件记录源码和本地实际数据可确定的工作量，不把调用次数当 FLOPs，也不据此虚构 GPU 小时数。

## 数据与调度

- 训练 150 QA，补齐至每轮 152 组，每 rank 38 组。五轮 760 组、190 个四卡同步轮次。
- 验证 single-step：66 个唯一状态，64 个有非空事实，因此 64 次 Manager 生成、81 次 Answer 生成。
- 验证 roll-forward：369 轮，350 轮有非空事实，因此 350 次 Manager 生成、81 次 Answer 生成。
- 每次验证 414 次 Manager + 162 次 Answer = 576 次生成；仅 rank 0 执行。
- 从头跑满五轮：epoch 0..5 共 6 次验证，即 2484 次 Manager + 972 次 Answer = 3456 次生成。

## 精确工作量表达式

令 V 为五轮合法训练候选总数，K 为实际参与策略更新的候选总数，H 为实际进行 gold 评分的候选总数。

| 路径 | 五轮次数 |
| --- | ---: |
| 训练 Manager.generate | 3040 |
| 训练 Answer.generate | V，至多 3040 |
| 全部验证 generate | 3456 |
| 冻结 Answer gold 整段前向 | H |
| Manager 策略整段前向 | K |
| Manager reference 整段前向 | K（默认 beta=.01） |
| backward 调用 | K |

总生成次数严格为 **6496+V**，上限 **9536**。非生成整段前向为 **H+2K**；反向调用为 **K**。梯度检查点在 backward 内的重算另计，不能当成额外免费操作。

K ≤ 3040。H 的松上界也是 3040，但默认首轮验收要求原 F1 信号组比例 ≥.15，首轮至少 ceil(152*.15)=23 组有 S-A 信号。这些组不会评分 gold。因此默认能跑满五轮时 H ≤ 4*(760-23)=2948。

五轮工作量包络：**至多 9536 次完整生成 + 9028 次非生成整段前向 + 3040 次 backward**。这是上界，不是实际使用量；三类操作成本不同，不能相加称为“同等 forward 次数”。

若首轮验收失败：执行 1 轮训练、2 轮验证。此时包络为 **2368 次完整生成 + 1824 次非生成整段前向 + 608 次 backward**。

## Token 和前向范围

- 生成 prompt 上限 4096 tokens，max_new_tokens=2048。9536 次生成的输出 token 总预算上限为 **19529728**；EOS 会提前结束，实际值依赖运行输出。
- 每次生成含一次 prompt prefill，随后为带 KV cache 的逐 token decode；一次 generate 不是一次普通前向。
- 策略和 reference 每次都处理 prompt+Manager completion，序列至多 6144 tokens，彼此不共享本次前向结果。词表 head 只在 completion 位置计算。
- gold 评分每次重新处理 Answer prompt+固定前缀+gold，长度为实际 prompt tokens+prefix/gold tokens。没有复用 Answer.generate 的 KV cache。仅 gold 预测位置经过词表 head，并按 32 tokens 分块；不生成文本。
- gold 完全同分或候选输入相同，当前实现仍先评分所有合法候选，之后才判断是否有辅助信号。

真实总 FLOPs 必须对上述实际序列长度、生成长度、模型结构和反向/重算路径求和；当前本地没有训练机模型文件及该次运行输出，无法给出真实数值。4bit 主要影响存储及内核成本，不能把 FLOPs 简单除以 4；LoRA 也不能按可训练参数占比缩放整个前向/反向成本。

## 四卡耗时的口径

在训练候选全合法的生成次数上界下，每卡训练生成 1520 次；rank 0 另外执行 3456 次验证生成，因此 rank 0 共 4976 次，另三卡各 1520 次。

故不能用 9536/4=2384 来估计生成路径的串行长度。实际训练每个同步轮次还要等最慢的 rank；所有 rank 的 score_seconds/backward_seconds 累加值不是墙钟时间。

默认优化器全局同步更新至多 190 次；每次四 rank 都执行 optimizer.step，所以本地 step 调用总数至多 760 次。760 个训练组不等于 760 次全局更新。

## 启动与保存

- 同 Manager/Answer 基座路径时：4 rank 各计算一次完整模型目录指纹，共 4 次完整逻辑扫描；操作系统文件缓存可能减少物理磁盘读取。
- 每 rank 分别加载 Manager 和 Answer，共 8 次 7B 模型实例加载；KL reference 通过关闭 Manager adapter 获得，不加载第三份模型。
- 数据审计每 rank 各执行一次；默认 MiniLM 检索器每 rank 懒加载，文本向量有进程内缓存，实际编码量依赖生成的新记忆文本。
- 每 rank 每轮 38 组，组 10/20/30 保存，轮末再保存：5 轮共 20 个 checkpoint。仅 rank 0 保存 adapter、tokenizer、optimizer/scheduler 状态；每个 checkpoint 另存 4 个 rank RNG 文件，共 80 个 RNG 文件。保存时重新扫描候选日志计算哈希。
- 当前入口不自动额外执行 precheck、离线探针、新 S-A 训练或 old_bank Answer 对照生成。

## 源码依据

`scripts/run_sf_training.sh`（配置）；`scripts/train_manager_sf.py`（调度、验收、验证、保存）；`sf_probe/training.py`（候选、辅助评分与更新）；`sf_probe/scoring.py`（gold 前向）；`scripts/train_manager_grpo.py`（generate、log_probability、验证与 checkpoint）；`evaluation/manager_validation.py`（验证缓存与空事实分支）；`memfactory/lora.py`（加载、分片、同步）；`sf_probe/run_contract.py` / `sf_probe/audit.py`（启动扫描）。
