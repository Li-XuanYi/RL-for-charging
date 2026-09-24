# E5 / E6 GPU 并行实验交接文档

> 本文件由仓库当前状态核对后撰写，所有路径、行号、种子、预算与哈希均来自实际文件。
> 目标读者：在另一台 Linux + NVIDIA 机器上接手运行 **实验 5（E5）** 与 **实验 6（E6）** 的协作者。
> 配套打包脚本：`package_e5_e6_handoff.sh`（同目录）。

---

## 1. 概述

`RL-for-charging` 是一套围绕锂电池充电控制的多智能体强化学习复现与"修改稿补充实验"流水线。
E1 辨识出三节电芯的物理参数并冻结成 handoff；E2 确定任务与安全口径；E3/E4/E5/E6 依次训练、
消融、敏感性与规模化，最后汇总成一条可审计的证据链。

固定实验协议（E3–E6 共用）：**决策周期 90 s，每个学习任务 2,500 个环境步，种子 `7,17,27,37,47`，
`device=cuda:0`，`exploratory=true`**。这是短预算数值探索，**不是收敛或方法优越性的最终证明**。

当前进度：

| 阶段 | 状态 | 说明 |
|---|---|---|
| E1 | ✅ 已完成 | 参数辨识 + handoff（`revision_results/experiment_1/20260920_105021_516059`） |
| E2 | ✅ 已完成（90 s / 2 seeds 子集） | 冻结子集清单 `revision_gpu/configs/e2_completed_90s_manifest.json` |
| E3 | ✅ 已完成 | 公平基线，`revision_results/experiment_3_gpu_parallel_2500_5seeds_90s/20260922_185259_914157` |
| E4 | ⏳ **运行中** | 组件消融；新时间戳目录 `.../20260924_175214_871584`（`status=running`） |
| E5 | ⬜ 待交接运行 | 需要 E4 完成后才能启动 |
| E6 | ⬜ 待交接运行 | 需要 E5 完成后才能启动 |

**本次交接的一句话目标**：把一份**逐字节一致**的源码树与上游结果树交到另一台机器，
使协作者能在**不重跑 E1/E2**、**不改动任何 `revision_gpu/*.py`** 的前提下，按序运行 E5 与 E6，
并得到与原机相同的 `code_fingerprint` / lineage / 哈希门禁结果。

---

## 2. E5 实验具体内容

### 2.1 科学问题

在**同一套任务、安全机制、种子、交互预算与验证/测试集**下，检验：

1. **奖励权重敏感性**：时间 / 均衡 / 电压 / 温度四类惩罚权重各自 ×0.5、×2 时结果如何？
2. **单位增益对照**：把四类权重都设为 1（raw 与 normalized 两种模式各一组）会怎样？
3. **探索率敏感性**：QMIX / DQN 在 `epsilon_start = 0.5` 与 `1.0` 下的差异。
4. **域随机化**：训练时加入环境/参数随机化是否改变结果。
5. **冻结策略泛化**：把 E3 训练好的 QMIX / MAPPO 与 E5 标称/域随机 QMIX **冻结**，
   放到 13 类扰动工况上测退化（配对到同一标称工况）。

> 关键方法学：**这是一张预先声明的重训练网格（predeclared retraining grid），不是冻结扰动**。
> 每个奖励/探索配置都**重新训练**，预算与种子完全相同；测试结果**从不**用于回调参数、
> 训练预算或选择 checkpoint（`revision_gpu/experiment5.py` 顶部 docstring 与 `protocol`）。

### 2.2 敏感性规格族（`_sensitivity_specs`，`experiment5.py:26-48`）

基线权重 `DEFAULT_WEIGHTS = {time:.75, balance:50, voltage:20, temperature:2}`（`experiment5.py:19`）。
`common` 使用 `method='qmix'`, `shield=True`, `reward_mode='raw'`, `epsilon_start=.5`。

| # | spec id | 相对基线的唯一改动 |
|---|---|---|
| 1 | `qmix_reward_reference` | 无（标称 QMIX 对照，作为统计基准） |
| 2 | `qmix_reward_all_unit_raw` | 四类权重全部 = 1，模式仍 `raw` |
| 3 | `qmix_reward_all_unit_normalized` | 四类权重全部 = 1，`reward_mode='normalized_unit'`，归一化尺度 `time=1, balance=.1, voltage=.1, temperature=10` |
| 4 | `qmix_reward_time_half` | `time × 0.5`（= 0.375） |
| 5 | `qmix_reward_time_double` | `time × 2.0`（= 1.5） |
| 6 | `qmix_reward_balance_half` | `balance × 0.5`（= 25） |
| 7 | `qmix_reward_balance_double` | `balance × 2.0`（= 100） |
| 8 | `qmix_reward_voltage_half` | `voltage × 0.5`（= 10） |
| 9 | `qmix_reward_voltage_double` | `voltage × 2.0`（= 40） |
| 10 | `qmix_reward_temperature_half` | `temperature × 0.5`（= 1） |
| 11 | `qmix_reward_temperature_double` | `temperature × 2.0`（= 4） |
| 12 | `qmix_epsilon_one` | `epsilon_start = 1.0` |
| 13 | `dqn_epsilon_reference` | 方法换为 `dqn`，`epsilon_start = 0.5`（DQN 对照） |
| 14 | `dqn_epsilon_one` | 方法换为 `dqn`，`epsilon_start = 1.0` |
| 15 | `qmix_domain_random` | 在标称 QMIX 上开启 `domain_randomization=True`（`experiment5.py:233`，**单独构造**，不属于 `_sensitivity_specs` 列表） |

共 **14 + 1 = 15 组 × 5 seeds = 75 个训练任务**
（`experiment5.py:281-286` 构造 `train_job`，`:286` 首次 `execute_jobs`）。

**单因子法说明**：4–11 号每个只改**一个**分量的权重；2/3 号是"整体缩放/归一化"的多系数对照，
在 `protocol['one_factor_at_a_time']` 中显式声明。归一化尺度在训练前冻结，
数值失败惩罚 `numerical_failure_penalty` 始终不变并单独报告。

### 2.3 冻结 / 泛化面板（第二段）

第二段 `execute_jobs`（`experiment5.py:298-317`）跑 **20 个冻结评测任务**：

- E3 冻结的 QMIX × 5 seeds + E3 冻结的 MAPPO × 5 seeds = 10（取自 E3 `model_registry.json`）
- E5 自己训练的 `qmix_reward_reference` × 5 + `qmix_domain_random` × 5 = 10

合计 **20 个模型 × 260 个扰动案例 = 5,200 个测试回合**。
260 = **20 个基础初态**（`generalization_random_count=10` 随机 SOC + `generalization_balanced_count=10`
初始均衡）× **13 个扰动族**（`_generalization_cases`，`experiment5.py:51-109`）。

| # | family | 扰动内容 |
|---|---|---|
| 1 | `nominal` | 无扰动（配对基准） |
| 2 | `ambient_15C` | 环境温度 `288.15 K` |
| 3 | `ambient_34C` | 环境温度 `307.15 K` |
| 4 | `ambient_35C` | 环境温度 `308.15 K`，`feasibility_negative_control=True` — **显式筛选可行性负对照** |
| 5 | `parameters_minus_5pct` | 参数 `[0,1,6,7]` ×0.95 |
| 6 | `parameters_plus_5pct` | ×1.05 |
| 7 | `parameters_minus_10pct` | ×0.90 |
| 8 | `parameters_plus_10pct` | ×1.10 |
| 9 | `sensor_noise_low` | 观测噪声 `[.005,.005,.25]` |
| 10 | `sensor_noise_high` | 观测噪声 `[.02,.02,1.]` |
| 11 | `sensor_bias` | 观测偏置 `[.02,-.02,-1.]` |
| 12 | `observation_delay_one_decision` | 观测延迟 1 个决策步 |
| 13 | `current_cap_drop_30pct` | 第 900 s 起每电芯电流上限降为 70%（7.5 A → 5.25 A，离散网格向下取整为 5.0 A） |

> **35 °C 负对照的含义**：默认温度裕量 1 K，筛选阈值 308 K。35 °C（308.15 K）即便零电流也已越过阈值，
> 所以它是**安全筛选可行性的负对照**，其失败**不能**解释为"算法学习能力差"或"物理任务不可行"。
> 34 °C 才是在初始裕量内的高温测试。对应说明见 `experiment5.py:257-261` 与
> `E5_INTERPRETATION.md`（`experiment5.py:332-356`）。

### 2.4 从 E3 读取的输入

- **E3 模型注册表** `model_registry.json`：由 `study_core.load_chain` 在 `study_core.py:170` 加载。
- E5 要求每个 seed 恰好有一个 N=3 的 qmix 与 mappo 条目，否则报错让重跑 E3（`experiment5.py:112-123`）。
- **E3 checkpoint 哈希校验**：`experiment5.py:196` 用 `entry['checkpoint_sha256']` 校验
  `project_path(entry['checkpoint'])` 的 SHA-256；不一致则拒绝。
- 任务/安全协议必须与 E3 冻结策略一致（`decision_s, voltage_margin_V, temperature_margin_K,
  soc_max, horizon_s, hold_s`，`experiment5.py:199-203`）。
- E3 的 QMIX 学习率必须唯一、DQN 学习率必须唯一（`experiment5.py:204-209, 230-231`）。
- 另经 `load_chain` 间接依赖 E1 handoff、E2 冻结子集（见 §4/§5）。

### 2.5 E5 产出文件（`out` 目录）

- 输入冻结：`validation_cases.json`、`sensitivity_test_cases.json`、`generalization_test_cases.json`、
  `effective_config.json`、`device_info.json`、`lineage.json`、`e1_model_used.json`、
  `e2_completed_90s_evidence.json`、`e2_slice_manifest_used.json`、`protocol.json`
- 敏感性训练：`model_registry.json`、`sensitivity_episode_metrics.csv`、
  `sensitivity_statistics/`、`dqn_exploration_statistics/`、
  `training/<spec>/seed_<seed>/`、`sensitivity/<spec>/seed_<seed>/`、`parallel_sensitivity_jobs/`
- 泛化面板：`transfer_model_registry.json`、`generalization_episode_metrics.csv`、
  `frozen_checkpoint_audit.json`、`paired_generalization_degradation.csv`、
  `paired_generalization_summary.csv`、`generalization/<method>/seed_<seed>/`、`parallel_generalization_jobs/`
- 汇总：`episode_metrics.csv`、`summary.csv`、`scientific_assessment.json`、`REPORT.md`、`E5_INTERPRETATION.md`
- 顶层：`requested_config.json`、`provenance.json`、`source_snapshot/`、`run.log`、`status.json`、`EXPLORATORY_RUN.json`

### 2.6 依赖与工作量

- 代码依赖：`load_chain(root,out,cfg,required=[1,2,3,4])`，**`experiment5.py:185`**。
- 工作量（@6 workers）：**≈ 40–48 h**（估计，基于 E3 实测基线，见 §7.4）。

---

## 3. E6 实验具体内容

### 3.1 三个问题

1. **规模扩展**：电芯数 3/6/12 时，闭环控制成功率与成本如何变化？（**重新训练**，非零样本迁移）
2. **推理延迟**：CPU 与 CUDA 上 `policy.act()` 的单次墙钟延迟、参数量与显存占用。
3. **证据链闭合**：把 E1–E5 的产出汇总成审稿意见对照表——**但不自动宣称审稿意见已全部解决**。

### 3.2 闭环规模 vs 仅推理规模

| 面板 | pack sizes | 是否训练 | 有无闭环控制证据 |
|---|---|---|---|
| `closed_loop_pack_sizes` | **[3, 6, 12]** | ✅ 训练 + rollout | ✅ 真实控制证据 |
| `inference_only_pack_sizes` | **[24, 48, 96]** | ❌ 未训练网络 | ❌ 仅延迟/参数量，**不能**当作控制成功率或泛化证据 |

配置见 `revision_gpu/configs/experiment6.json:29-38`；代码校验两者不得重叠、闭环必须包含 N=3
（`experiment6.py:284-293`）。大包是"已标定三节参数的合成复制"，不是新增实测电芯
（`_tile_vectors`，`experiment6.py:144-147`）。

### 3.3 30 个训练任务

**3 sizes × 2 methods（qmix, mappo）× 5 seeds = 30**，方法学习率沿用 E3 声明值、不做 E6 调参
（`experiment6.py:309-319` 构造 specs，`:359-366` 构造 `train_job`，`:366` 执行）。
每个规模都重新训练，训练与测试都先抽 3 节 SOC 三元组再同步复制，保持每节 SOC 分布
（`replicate_training_triplet=True`，`experiment6.py:277-279`）。

### 3.4 串行 CPU/CUDA 计时面板

- 训练全部结束后，才对已训练模型按 `benchmark_devices=[cpu, cuda:0]` **逐个设备串行**计时
  （`_profile_entry`，`experiment6.py:131-141`）。
- 另设**未训练**大包面板：3 sizes × 2 methods × 2 devices = **12 次 benchmark**
  （`experiment6.py:402-413`）。
- 每次 benchmark = `warmup 50 + repetitions 500 = 550` 次 `policy.act()` 调用
  （`inference_warmup=50`、`inference_repetitions=500`，`experiment6.json:39-40`；
  `_benchmark`，`experiment6.py:78-128`；循环 `experiment6.py:97`）。
- CUDA 计时显式 `torch.cuda.synchronize`，包含主机↔显卡传输（`experiment6.py:101-105`）。

> **必须独占机器**：本项目所有训练 worker 退出后才开始计时，但**其他用户的后台任务仍可能干扰**
> （`experiment6.py:367` 注释、`protocol['inference_timing_isolation']`，`experiment6.py:326`）。
> 需独占条件做**训练成本**比较时设 `MAX_WORKERS=1`。

### 3.5 `_closure` 与审稿证据映射

`_closure`（`experiment6.py:179-270`）：

- 逐个上游目录读取 `scientific_assessment.json` / `summary.json` / `handoff.json`（`_read_assessment`，`:168-176`）。
- 若存在 `e2_completed_90s_evidence.json` 则并入 E2 切片证据（`:182-185`）。
- 写出：`reviewer_evidence_chain.json`、`reviewer_evidence_map.csv`、`E3_E6_EVIDENCE_CHAIN.md`。
- **断言 `all_reviewer_comments_resolved = False`**（`experiment6.py:234`），并在返回值中同样返回
  `all_reviewer_comments_resolved: False`（`:432`）——证据链是"生成了可审查的证据"，不是"意见已解决"。

### 3.6 E6 产出文件

- `effective_config.json`、`protocol.json`、`machine.json`
- 闭环：`N{3,6,12}/validation_cases.json`、`N{3,6,12}/test_cases.json`、
  `N{n}/<method>/seed_<seed>/`、`model_registry.json`、`closed_loop_episode_metrics.csv`、
  `training_and_evaluation_cost.csv`、`parallel_scaling_jobs/`
- 计时：`inference_timings.csv`、`inference_only/N{24,48,96}/*.json`、各训练目录下的 `policy_timing_*.json`
- 统计/汇总：`paired_method_statistics/`、`episode_metrics.csv`、`summary.csv`、
  `scientific_assessment.json`、`REPORT.md`
- 证据链：`reviewer_evidence_chain.json`、`reviewer_evidence_map.csv`、`E3_E6_EVIDENCE_CHAIN.md`
- 上游冻结：`e2_completed_90s_evidence.json`、`e2_slice_manifest_used.json`、`e1_model_used.json`、
  `lineage.json`、`device_info.json`
- 顶层：`requested_config.json`、`provenance.json`、`source_snapshot/`、`run.log`、`status.json`、`EXPLORATORY_RUN.json`

### 3.7 依赖与工作量

- 代码依赖：`load_chain(root,out,cfg,required=[1,2,3,4,5])`，**`experiment6.py:274`**。
- 工作量（@6 workers）：**≈ 22–36 h**（估计，基于 E3 实测基线，见 §7.4）；**计时面板不能并行**。

---

## 4. 依赖关系与并行限制

### 4.1 依赖 DAG

| 阶段 | required（`load_chain`） | 代码位置 |
|---|---|---|
| E3 | `[1,2]` | `experiment3.py:14` |
| E4 | `[1,2,3]` | `experiment4.py:11` |
| E5 | `[1,2,3,4]` | `experiment5.py:185` |
| E6 | `[1,2,3,4,5]` | `experiment6.py:274` |

即：**E4 ← E3**，**E5 ← E4**，**E6 ← E5**。

### 4.2 显式并行限制（务必遵守）

1. **E4 运行期间不能启动 E5**，双重保护：
   - `_completed` 要求标签目录下存在 `LATEST_COMPLETED.txt`，且其内容必须等于 `LATEST_STARTED.txt`，
     并且该 run 的 `status.json` 为 `"completed"`（`study_core.py:86-102`，关键判断在 `:95-101`）。
   - `launch.lock`（`revision_cache/gpu_environment/launch.lock`）使用**非阻塞独占 flock**
     （`LOCK_EX | LOCK_NB`，`bootstrap_gpu.py:149-161`）：同一项目目录下 E3/E4/E5/E6 启动器**互斥**，
     重复启动会直接报错 `Another GPU study launcher is active in this project.`
2. **需要两套独立的项目目录**才能让两个阶段并发：lock 与结果目录都是**项目目录本地**的。
3. **E5 与 E6 不能在同一目录重叠**：E6 依赖 E5 完成（required 含 5），且同样受 `launch.lock` 约束。
4. **E4 交接前置条件**：`revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/` 必须
   - 存在 `LATEST_COMPLETED.txt`，
   - `LATEST_STARTED.txt` 与 `LATEST_COMPLETED.txt` 内容一致，
   - 对应 run 的 `status.json` 为 `"completed"`。
   当前 E4 的 `LATEST_STARTED.txt` 指向 `.../20260924_175214_871584`，且**尚无** `LATEST_COMPLETED.txt`
   （`status.json = {"status":"running","experiment":"4"}`）→ **E4 未完成，E5 现在不能启动**。
5. **清理游离目录**：`revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/20260924_175023_378893`
   是一次失败的探测（`status.json = {"status":"failed","error":"Worker 0 failed (exit -9)..."}`，
   仅运行 ~5.9 s 即失败），应在交接前删除，避免混淆（它不在 `LATEST_*` 指向链上，
   不会阻塞门禁，但会造成误解）。

---

## 5. 交接文件清单（逐字节）

目标机上的目录布局必须与源机保持一致：
`revision_gpu/`、`revision_experiments/`、`revision_results/` 保持相对位置（`paths.py:4-8`）。

需要复制的组：

**(a) 源码树**
- `revision_gpu/`（整个目录；可排除 `__pycache__/`）
- `revision_experiments/`（整个目录；可排除 `__pycache__/`）

**(b) E1 pinned run**
- `revision_results/experiment_1/20260920_105021_516059/`（由 `experiment_5.json` 的
  `upstream_results["1"]` 指定；必须含 `handoff.json`、`protocol.json`、`data_audit.json`、
  `summary.json`、`metrics.json`、`source_snapshot/`）

**(c) E2 run dir + 冻结清单**
- `revision_results/experiment_2_exploratory_2500_2seeds/20260922_114425_221647/`
  （由 `revision_gpu/configs/e2_completed_90s_manifest.json` 的 `source_run` 指定）
- `revision_gpu/configs/e2_completed_90s_manifest.json`

> ⚠️ **E2 字节警告**：该子集是在 **Linux 上以 LF 重新生成**的；Windows 原始 artifact 哈希因
> CRLF + 缺失 `.pt` 已失效（见 manifest 的 `scope` 字段）。因此：
> - **不要重跑 E2**；
> - **不要**让这些文件经过任何会改变行尾 / 编码的工具（如 Windows 文本编辑器、`unix2dos`、
>   某些 Git `autocrlf` 配置）；
> - **不要**用 `e2_completed_90s_manifest.windows_backup.json` 替换活动清单。
> `e2_slice.py:14-16` 会逐个校验 manifest 里的 `artifact_sha256`，任一字节变化都会拒绝启动 E4/E5/E6。

**(d) E3 run dir + LATEST 指针**
- `revision_results/experiment_3_gpu_parallel_2500_5seeds_90s/20260922_185259_914157/`（`status=completed`）
- `revision_results/experiment_3_gpu_parallel_2500_5seeds_90s/LATEST_STARTED.txt`
- `revision_results/experiment_3_gpu_parallel_2500_5seeds_90s/LATEST_COMPLETED.txt`

**(e) E4 run dir + LATEST 指针（仅在 E4 **完成之后**）**
- `revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/<完成时间戳>/`
- `revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/LATEST_STARTED.txt`
- `revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/LATEST_COMPLETED.txt`
- **交接前先删除**失败的游离目录 `.../20260924_175023_378893/`。

**(f) E2 直接读取但不在 manifest 内的文件**
- `revision_results/experiment_2_exploratory_2500_2seeds/20260922_114425_221647/source_snapshot/control.py`
  （`study_core.py:122-123` 会与当前 `revision_experiments/control.py` 逐字节比较）
- `revision_results/experiment_2_exploratory_2500_2seeds/20260922_114425_221647/model_handoff_used.json`
  （`study_core.py:124-126` 校验它与 E1 handoff 同源同哈希）
- 注意：这两个文件在 (c) 的整个 run dir 里已包含；此组仅用于强调他们**也**被门禁直接读取。

**(g) E5 run dir（仅 E6 需要）**
- E5 完成后，把 `revision_results/experiment_5_gpu_parallel_2500_5seeds_90s/` 连同
  `LATEST_STARTED.txt` / `LATEST_COMPLETED.txt` 一并交给运行 E6 的一方。

> ⚠️ **警告框：绝不要修改任何 `revision_gpu/*.py` 或 `revision_experiments/*.py` / `control.py` / `model.py` /
> `original_parameters.json`。**
> `study_core.code_fingerprint()`（`study_core.py:80-83`）对 `revision_gpu/*.py` 与
> `revision_experiments/{control.py, model.py, original_parameters.json}` 取哈希，并写入 lineage。
> 任何改动都会让 E3/E4（及后续）的 `code_fingerprint` 与当前源码不一致，
> 触发 `study_core.py:130-137` 的 `Experiment N used a different control/algorithm implementation` 而**拒绝启动**。
> 同样地，E1 的 `model.py` / `original_parameters.json` 有独立校验（`study_core.py:119-120`），
> E2 的 `control.py` 有快照校验（`study_core.py:122-123`）。

---

## 6. 启动命令与环境变量

### 6.1 启动命令（在项目根目录，先按 `bash` 顺序）

```bash
# E5（等 E4 完成后）
bash run_experiment_5_gpu_parallel_2500_5seeds_90s.sh

# E6（等 E5 完成后）
bash run_experiment_6_gpu_parallel_2500_5seeds_90s.sh
```

两个 `.sh` 都是 `set -euo pipefail`，解析自身目录为 `ROOT`，然后用
`exec "$BOOTSTRAP_PYTHON" -u revision_gpu/bootstrap_gpu.py <5|6> "$@"` 启动
（`run_experiment_5_...sh:10`、`run_experiment_6_...sh:10`）。
**`"$@"` 会被完整转发**：`bootstrap_gpu.py` 再把 `sys.argv[1:]` 交给 `run_study.py`
（`bootstrap_gpu.py:172`），因此 `--config` 与 `--resume` 均可透传。

可选参数：

| 参数 | 作用 |
|---|---|
| `--config <path>` | 覆盖默认配置（默认 `revision_gpu/configs/experiment_{5,6}.json`，`run_study.py:34`） |
| `--resume <result_dir>` | 复用失败/中断 run 中**已完成**的独立 job（`run_study.py:32,41-53`） |

### 6.2 环境变量

| 变量 | 默认 | 作用 | 代码位置 |
|---|---|---|---|
| `GPUS` | 未设置（自动选卡） | 逗号分隔的物理卡号或 GPU UUID；受 `CUDA_VISIBLE_DEVICES` 调度器可见性限制 | `bootstrap_gpu.py:59-69` |
| `GPU_INDEX` | — | 单卡兼容写法；`GPUS` 优先 | `bootstrap_gpu.py:59` |
| `WORKERS_PER_GPU` | `1` | 每张卡的 worker 槽位数 | `bootstrap_gpu.py:71` |
| `MAX_WORKERS` | `6` | 总 worker 上限 | `bootstrap_gpu.py:72` |
| `ALLOW_BUSY_GPU` | 未设置 | `=1` 时允许共享已分配但繁忙（利用率 > 30%）的卡；**仍需显式设置 `GPUS`**，显存门槛保留 | `bootstrap_gpu.py:70,74-75` |
| `REPRO_PYTHON` | 未设置 | 指定 3.11/3.12 解释器完整路径用于建 venv | launcher `:5-6`；`bootstrap_gpu.py:103-104` |
| `REPRO_PIP_INDEX_URL` | `https://pypi.org/simple` | pip 索引 URL（离线/镜像环境） | `bootstrap_gpu.py:137` |
| `CUDA_VISIBLE_DEVICES` | 继承 | 调度器可见性；`GPUS` 不能越过它 | `bootstrap_gpu.py:50-58` |
| `SLURM_CPUS_PER_TASK` | — | 存在时用于限制 CPU 预算 | `bootstrap_gpu.py:84` |

### 6.3 有效并行度公式

选卡后（要求空闲显存 ≥ `4096 × WORKERS_PER_GPU` MiB 且利用率 ≤ 30%），槽位数为
`slots = [每张选中卡重复 WORKERS_PER_GPU 次]`，然后：

```
limit = min( MAX_WORKERS,
             max(1, cpu_count // 4),
             max(1, MemAvailable_bytes // (4 GiB)),
             len(slots) )
```

即 **`min(MAX_WORKERS, cpu//4, MemAvailable//4GiB, #slots)`**
（`bootstrap_gpu.py:82-90`；`cpu_count` 取 `os.sched_getaffinity`，Linux 下再由
`/proc/meminfo` 的 `MemAvailable` 限制）。最终 `CUDA_VISIBLE_DEVICES` 只暴露**第一张**卡给协调进程，
真正的槽位列表放在 `REVISION_GPU_SLOTS`（`bootstrap_gpu.py:91-92`）。

> 该内存预算是**调度启发式**，不是模型峰值保证（`START_GPU.md`）。

### 6.4 `launch.lock` 说明

`revision_cache/gpu_environment/launch.lock` 在 `bootstrap_gpu.py:151-161` 以
`fcntl.flock(LOCK_EX|LOCK_NB)`（Windows 用 `msvcrt.locking`）**非阻塞独占**加锁，
锁文件由 **进程退出自动释放**，无需手工删锁。它保证同一项目目录下 E3→E4→E5→E6 串行；
若检测到另一个 launcher 在跑，会抛
`RuntimeError('Another GPU study launcher is active in this project. Run E3 -> E4 -> E5 -> E6 serially.')`。

---

## 7. 运行注意事项

### 7.1 崩溃/重启后续跑（`--resume`）

```bash
bash run_experiment_5_gpu_parallel_2500_5seeds_90s.sh \
  --resume revision_results/experiment_5_gpu_parallel_2500_5seeds_90s/<实际时间戳>
```

- `run_study.py:41-53`：`--resume` 要求 `requested_config.json` **完全一致**、
  `provenance['source_files']` 每个源文件哈希**完全一致**；若 `status.json` 已是 `completed` 则直接返回。
- `parallel_jobs.execute_jobs`（`parallel_jobs.py:40-52`）：仅当 `job_*/result.json` 的
  **任务签名一致**（签名含 `code_fingerprint`，`parallel_jobs.py:36-37`）**且 checkpoint 字节哈希一致**
  时才复用；否则该 job 从头重训。
- **这不是优化器逐步断点续训**：只复用"完整独立的 job"，部分完成的 job 重新训练
  （`START_GPU.md`；`run_study.py:32`）。
- E3/E4 的规则电流验证、汇总统计、E6 的推理计时会重新执行。
- 中断请用 **Ctrl+C**：脚本只清理自己创建的子进程（`bootstrap_gpu.py:29-37`、
  `parallel_jobs.py:89-96`），**不要**强杀服务器上的全部 Python 进程。

### 7.2 长任务保活

- 终端退出会导致前台作业收到 SIGHUP 而中断；建议用：
  ```bash
  setsid bash run_experiment_5_gpu_parallel_2500_5seeds_90s.sh > e5.out 2>&1 &
  # 或
  nohup bash run_experiment_6_gpu_parallel_2500_5seeds_90s.sh > e6.out 2>&1 &
  ```
- 这能扛住**终端退出**，但**扛不住重启/断电**；重启后用 `--resume` 续跑（见 §7.1）。

### 7.3 E6 计时面板要求

E6 的 CPU/CUDA 计时必须在**本项目训练进程全部退出后、机器基本空闲**时进行；
若在计时阶段同时跑其他项目作业，延迟数字不可信（`experiment6.py:326,367`）。
E6 已内置"先训练、后计时"的顺序，但仍无法控制其他用户的后台任务。

### 7.4 典型墙钟时间（基于 E3 实测基线）

实测基线（`experiment_3_gpu_parallel_2500_5seeds_90s/20260922_185259_914157/status.json` 与
`model_registry.json`）：

- E3 墙钟 **`elapsed_s = 29772.97` ≈ 8.27 h**，共 **33 个 job**，`scheduler.json` 记录
  **6 个 GPU 槽位**（2 张物理卡各 `WORKERS_PER_GPU=3`，`job_count=33`）。
- 30 个学习型 job 的 `training_wall_s` 均值 **6264.7 s ≈ 1.740 h**（min 5158.8 / max 7395.4）。
  即 **每个 N=3 / 2500-step 训练任务 ≈ 1.74 h**（该数字即"1.74 h/任务"的来源）。

据此估算（@6 workers，含每个 scale 的验证与测试开销）：

| 实验 | 训练 job | 评测 job | 估计墙钟 |
|---|---:|---:|---|
| E5 | 75 | 20（各 260 例） | **≈ 40–48 h** |
| E6 | 30 | 30 闭环测试 + 计时面板 | **≈ 22–36 h** |

> 以上为**估计**，非实测；E6 的计时面板**不能并行**，且对机器空闲度敏感。

---

## 8. 打包脚本用法

在项目根目录运行：

```bash
bash package_e5_e6_handoff.sh
```

脚本会：

1. 以脚本自身所在目录为 `ROOT`；
2. 选取 `revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/` 下**最新的、`status.json` 为
   `"completed"`** 的时间戳目录；若找不到，默认**报错退出 1**，除非设置 `ALLOW_INCOMPLETE_E4=1`
   （此时仅告警并继续）；
3. 按 §5 的清单聚合成一个 tar.gz，**保留 ROOT 下的相对路径**，并排除
   `.venv-revision-gpu/`、`revision_cache/`、`__pycache__/`、`*.pyc` 以及清单外的其它实验目录；
4. 输出到 `$ROOT/E5_E6_handoff_<YYYYmmdd_HHMMSS>.tar.gz`；
5. 打印 tarball 路径、大小与 `sha256sum`，并生成 `handoff_manifest_sha256.txt`，
   内含关键固定文件的 SHA-256（E1 `handoff.json`、E2 清单、E3 `model_registry.json`、
   E3 `LATEST_COMPLETED.txt`）供协作者核对。

可选的调试开关：`DRY_RUN=1`（只打印将要打包的文件清单与关键哈希，**不创建 tarball**）。

环境变量：

| 变量 | 作用 |
|---|---|
| `E4_DIR` | 覆盖 E4 标签目录路径（默认 `$ROOT/revision_results/experiment_4_gpu_parallel_2500_5seeds_90s`） |
| `ALLOW_INCOMPLETE_E4=1` | E4 未完成时仅告警不退出（**正式交接不要用**） |
| `DRY_RUN=1` | 只显示文件清单，不打包 |

---

## 附：本次核对到的关键路径与哈希

| 对象 | 路径 | 备注 |
|---|---|---|
| E1 pinned run | `revision_results/experiment_1/20260920_105021_516059` | 来自 `experiment_5.json` `upstream_results["1"]` |
| E1 `handoff.json` | 上述目录下 | sha256 = `f1a07c158a64cd7259bf531231fd38af92dc65148e86e823dfcfbdf0d5a302ce`（**config 未固定该哈希**，运行时动态计算） |
| E2 run dir | `revision_results/experiment_2_exploratory_2500_2seeds/20260922_114425_221647` | 来自 E2 清单 `source_run` |
| E2 清单 | `revision_gpu/configs/e2_completed_90s_manifest.json` | sha256 = `b97fa992d649b941999e31ba1e631a731e0fce7ee12fe4a56aa15791780d81aa` |
| E3 run dir | `revision_results/experiment_3_gpu_parallel_2500_5seeds_90s/20260922_185259_914157` | `status=completed`，33 models / 462 episodes |
| E3 `model_registry.json` | 上述目录下 | sha256 = `b3ab93128cadf75a345ed686becc23d2ac2d33b17a367582febe6627216a3a52` |
| E3 `LATEST_COMPLETED.txt` | `.../experiment_3_gpu_parallel_2500_5seeds_90s/LATEST_COMPLETED.txt` | sha256 = `6fa5ebce77ed09224e69e6fb878f2ef5802032cf15d3bcbfdc46afca9f48f348` |
| E4 运行中目录 | `revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/20260924_175214_871584` | `status=running` |
| E4 失败游离目录 | `revision_results/experiment_4_gpu_parallel_2500_5seeds_90s/20260924_175023_378893` | `status=failed`（worker exit -9），**交接前删除** |
