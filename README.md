# Memory-R1 连续训练（verl）

当前训练入口为 `verl_memory/train.py`，执行器为 `verl_memory/workers.py`。
Manager、Answer 使用 verl / BF16 / vLLM；Manager 训练使用两卡 Actor 和两卡冻结 Answer，Answer 训练使用四卡 Actor。

## 运行入口

| 任务 | 入口 |
| --- | --- |
| Baseline Manager | `bash scripts/run_sf_training.sh sa` |
| S-F Manager | `bash scripts/run_sf_training.sh sf` |
| 固定 Manager 构建 Answer 训练数据 | `bash scripts/run_sf_training.sh answer-data` |
| Answer 训练 | `bash scripts/run_sf_training.sh answer` |
| 连续正式 test | `bash scripts/run_sf_test.sh` |
| 训练后画图 | `python scripts/diagnose_rl.py --run OUTPUT_DIR` |
| 抽取 teacher facts | `scripts/build_manager_training_data.py` |
| 从缓存 facts 准备连续时间线 | `scripts/rebuild_manager_states.py` |

模型路径通过 `.env.example` 列出的环境变量或命令行提供；脚本不会自动加载 `.env.example`。
`--help` 显示当前参数。Judge 默认关闭；`RUN_JUDGE=1` 才会在 test 完成后单独运行。

## 训练和 bank

Baseline 和 S-F 均从空 bank 连续处理全部对话 turn，再对最终 bank 回答全部有效 QA。
每条轨迹维护独立 bank；默认采样四条轨迹。Baseline 每个 QA 在轨迹组内按总体标准差标准化 F1，再对 QA 取平均，作为该轨迹的优势。
S-F 在此基础上按 `--credit-interval`（默认 4）保存 bank；通过冻结 Answer 对 gold 的平均 token log probability 计算块间变化，加入该块操作的优势。
整组最终优势为零时跳过优化器更新。无 epoch-0 验证；epoch 末保存和验证。没有单步训练入口、gate 或 oracle 对照。

训练、验证、test 共用 `evaluation/manager_validation.py` 和 `memfactory/manager_env.py`。
Manager 由 embedding 检索 top-5 memories；Answer 每位参与者检索 top-30。Manager 使用用户在 `memfactory/modules/memory_updater.py` 中设置的 prompt。

ADD 的实际 ID 由代码根据本轮开始时完整 bank 的最大数字 ID 加一分配，无数字 ID 时从 0 开始。
旧非数字 ID 保留。模型 ADD ID 作为本轮别名，映射记录在 `id_mappings`；已有可见 ID 优先于别名。
同一个已有 ID 的单个有效 UPDATE 与 DELETE 同时出现时执行 UPDATE，与输出顺序无关；NONE 与单个修改兼容。
其他重复修改只拒绝该目标，不撤销其他有效操作。不存在整个 bank 每轮最多五次修改的限制。

## 数据、日志和错误

训练前清理坏 fact 和无有效 question/gold 的 QA，记录原始位置和原始值，不把 null 转成字符串。
异常记录在 `data_issues/`；跳过 QA 会改变有效评分分母。无法恢复时间顺序的对话会记录后跳过。
原始输出、bank 前后状态、ID 映射、解析错误和修复记录保存在轨迹 JSONL 与压缩 evidence 中。
逐组训练指标写入 `training_metrics.jsonl`，验证结果写入 `evaluation/epoch_N.json`，画图输出 CSV/PNG/HTML。
训练中的底层错误原样传播并记录，不额外捕获后跳组。启动/续训的自定义检查仅提示，不主动拒绝。
命令行仍采用 argparse；参数拼写和类型错误由 argparse 正常报告。Shell 保留必要路径变量检查及子进程非零退出状态。

## 本地数据与验证范围

本地连续训练时间线为 `data/stage0/dialogue_timelines_train.json`，验证时间线为 `data/stage0/dialogue_timelines_validation.json`。
正式 test 默认读取 `data/stage0/dialogue_timelines_test.json`；若未生成，可使用：

```bash
python scripts/rebuild_manager_states.py --raw data/locomo10.json --facts-jsonl data/manager_training_data_vllm.jsonl --output-dir data/stage0 --splits test
```

`manager_states_*.jsonl` 和旧审计 JSON 是历史单步数据，不被当前 trainer 读取。
旧 S-F Markdown 和 `test_results_sf*.txt` 是历史记录，不代表当前配置或当前测试结果。
`python -m unittest discover -s tests` 验证本地 CPU 逻辑；不代表真实四卡训练已验收。
