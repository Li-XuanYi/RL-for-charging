# 实验 3–6：GPU 并行探索版

固定方案：90 秒决策周期，每个学习任务 2,500 个环境步，种子 7、17、27、37、47。这是短预算数值探索，不是论文充分收敛或方法优越性的最终证据。

作者交付时没有启动 Python、依赖安装、测试或实验；只做了源代码、配置、文件完整性检查。首次启动时才安装环境、检查 CUDA 和开始计算。

## Linux 服务器

把整个压缩包上传并解压到服务器任意一个有写权限的目录。保留 revision_gpu、revision_experiments 和 revision_results 的相对位置。不要只上传 .sh，也不要复制 Windows 的虚拟环境。需要 Python 3.11/3.12、venv、NVIDIA 驱动和软件包下载网络。

在解压目录按顺序运行（后一条只在前一条正常完成时开始）：

```bash
bash run_experiment_3_gpu_parallel_2500_5seeds_90s.sh && \
bash run_experiment_4_gpu_parallel_2500_5seeds_90s.sh && \
bash run_experiment_5_gpu_parallel_2500_5seeds_90s.sh && \
bash run_experiment_6_gpu_parallel_2500_5seeds_90s.sh
```

四个实验有上下游依赖，因此实验之间串行；每个实验内部的不同方法、种子或规模并行。E3 的 CC 电流验证选择在主任务前执行，属于 CPU 规则控制器计算。

默认自动选择当前调度器可见、空闲显存至少 4 GiB、利用率不超过 30% 的卡，每卡一个任务，最多六个；同时按可见 CPU 核数除以 4、Linux 主机可用内存每任务 4 GiB 的粗略预算限制进程数。此内存预算是调度启发式，不是模型实际峰值保证。独立任务使用不同进程、输出目录和随机种子；每个进程只看见分配给它的一张 GPU。没有对单个小网络使用 DDP。

如已分配到 0、1、2 三张卡，明确指定：

```bash
export GPUS=0,1,2
export MAX_WORKERS=3
bash run_experiment_3_gpu_parallel_2500_5seeds_90s.sh
```

GPUS 是 nvidia-smi 中的物理卡编号或 GPU UUID，不能越过 CUDA_VISIBLE_DEVICES 的调度器分配。支持原来的 GPU_INDEX 作为单卡兼容选项，GPUS 优先。不要在未分配的共享节点上直接占用别人的卡。

若确认自己的卡可以共享，但利用率检查阻止启动，可以额外设置 ALLOW_BUSY_GPU=1，必须同时明确设置 GPUS。显存门槛仍保留，不会结束现有进程。显卡空出来后重新启动即可，不会后台排队占用。

单卡显存和 CPU 资源充裕时，可设置 WORKERS_PER_GPU=2、MAX_WORKERS=相应总任务数。默认仍是一卡一任务，避免 CPU 仿真竞争。多卡不保证线性加速；PyBaMM/CasADi 电池积分和安全过滤仍在 CPU 上，CUDA 加速的是神经网络。

如 python3 不是 3.11/3.12，可设置 REPRO_PYTHON 为合适的解释器完整路径。环境放在项目下 .venv-revision-gpu；缓存放在 revision_cache。不会修改现有 Conda 或 .venv-reproduce 环境。CUDA 版本为 PyTorch 2.7.1/cu128，依据 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。若服务器无法访问 PyPI，可设置 REPRO_PIP_INDEX_URL；CUDA wheel 从 PyTorch 官方源获取。

## Windows

四个同名 .cmd 文件是 Windows 本地入口，需要本机也有兼容 NVIDIA 显卡。CMD 不会自动 SSH 到 Linux 服务器。双击对应 CMD，或在项目目录执行：

```cmd
set GPUS=0
set MAX_WORKERS=1
run_experiment_3_gpu_parallel_2500_5seeds_90s.cmd
```

也按 3→4→5→6 顺序运行。不要同时打开四个入口；项目锁会阻止交叉启动。并行由一个入口内的调度器完成。

## 实验内容和计算量

| 实验 | 学习任务 | 预算 | 对应拒稿意见 |
|---|---:|---:|---|
| 3 | 6 方法 × 5 种子 = 30 | 75,000 环境步 | QMIX、MAPPO、DQN、共享 IQL；另设连续 MAPPO/SAC 轨道及三种规则对照。相同轨道、初态、种子与交互预算配对 |
| 4 | 8 架构/机制 × 5 = 40 | 100,000 环境步 | 去 GRU、换 VDN、去 hypernetwork、去 mixer、去共享、独立 IQL、去均衡奖励；重训完整 QMIX 参考 |
| 5 | 15 组 × 5 = 75 | 187,500 环境步 | 时间/SOC/电压/温度权重半倍与双倍、全 1/归一化全 1、QMIX 与 DQN 的 epsilon=0.5/1 对照、域随机化及冻结策略泛化 |
| 6 | 3/6/12 电芯 × 2 方法 × 5 = 30 | 75,000 环境步 | 可扩展性、训练时间、进程 RSS、CUDA 分配/保留显存、参数量、CPU/CUDA 分开同步计时；24/48/96 只测未训练网络推理 |

总计 175 个学习任务、437,500 个电池包环境交互步，不是 437,500 次完整训练。验证和测试额外计算：E3/E4 各使用 12 个随机测试初态加 2 个熟悉 A/B 诊断；每组验证 4 个初态。E5 敏感性每模型 12 个测试，泛化 13 类条件 × 20 个基础初态 × 20 个冻结模型 = 5,200 个测试回合。E6 每模型 12 个测试。因此 E5/E6 仍可能耗时较长。

取消原来大量学习率搜索，学习方法统一预先固定 0.0002；CC 候选电流仍只在验证集选。MAPPO 每累计约 256 步更新，避免 2,500 步短预算内更新次数过少。每个方法仍有自己的更新规则，不能把等交互步数宣称成等算力或完全同构。

E4 继承 E3 的模型、种子、预算、验证/测试集及超参数。E5 使用 E3 指定模型，测试期间不更新、不挑最好种子；权重变化与探索率变化分开统计，DQN 对比自己的参考。E6 汇总整个证据链。源文件指纹/上游模型哈希不一致时停止，避免混用不同版本结果。

## 保存位置和日志

所有结果都在启动脚本所在项目中：

```text
revision_results/
  experiment_3_gpu_parallel_2500_5seeds_90s/时间戳/
  experiment_4_gpu_parallel_2500_5seeds_90s/时间戳/
  experiment_5_gpu_parallel_2500_5seeds_90s/时间戳/
  experiment_6_gpu_parallel_2500_5seeds_90s/时间戳/
```

run.log 显示任务分配与完成进度。parallel_*jobs/job_XXXX/worker.log 显示各组训练进度，progress.json 显示活动任务；原始模型、CSV、压缩轨迹和图放在各方法/种子子目录中。完成后检查 status.json、REPORT.md、scientific_assessment.json、summary.csv 和配对统计，不要将 COMPLETED 理解为“论文结论成立”。

独立任务失败后，保留已完成任务；同配置、同源文件、同项目路径恢复示例：

```bash
bash run_experiment_3_gpu_parallel_2500_5seeds_90s.sh --resume revision_results/experiment_3_gpu_parallel_2500_5seeds_90s/实际时间戳
```

必须替换实际时间戳。完整 result.json 的任务经配置和模型哈希检查后复用，部分完成的任务从头训练；不是优化器逐步断点续训。E3 规则电流验证、汇总统计和 E6 推理计时会重新执行。没有 --resume 时会新建运行目录，不覆盖旧结果。中断时使用 Ctrl+C，脚本清理自身子进程，保留日志；不要强制杀整个服务器上的 Python 进程。

## 上游证据与结论边界

包内复用已完成 E1 的模型交接结果，以及原 E2 的 90 秒、种子 7/17 全部完成的模型和指标。E2 快照 195 个文件逐个校验 SHA-256，无需重跑 E1/E2。30/10 秒结果、E2 全量轨迹没有打包，原项目继续保留。E2 父运行当时的 status 仍是 running，程序明确标记“仅使用已完成 90 秒子集”，不会伪造整体完成状态或五种子 E2 证据。

五个种子可以描述波动，但 2,500 步是否收敛必须由曲线判断；五种子的双侧精确符号翻转检验最小 p 值为 0.0625。90 秒结论限于该离散控制周期。扩大规模是已标定三种电芯的合成复制，不能说成真实 12 节电池验证。

E6 训练耗时来自并行负载下的测量，不能直接当作独占设备的算法复杂度证据。CPU/CUDA 推理在所有本项目训练进程退出后分开测量，CUDA 计时显式同步且包含主机/显卡数据传输；其他用户的服务器作业仍可能影响结果。如需独占条件的训练成本比较，正式单独运行时设置 MAX_WORKERS=1。RSS 是采样峰值，CUDA 显存是本进程 PyTorch 分配器统计，均明确口径。

模型物理真实性、独立充电数据、硬件实验、真实能量效率与寿命验证的缺口仍存在。输入 Wh 不等于损耗或效率，不能用 GPU 改造替代拒稿意见要求的科学证据。
