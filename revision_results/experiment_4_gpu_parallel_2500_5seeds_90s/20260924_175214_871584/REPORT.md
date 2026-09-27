# 短预算 GPU 探索结果：不能作为收敛或算法优越性的最终证明

# 实验结果

计算完成，以下结果包含失败与超时。是否支持方法优势需根据成功率、配对差异、终端 SOC 和限制判断。

| 方法 | 电芯数 | 动作 | 面板 / 工况 / 初态组 | 安全成功率 | 失败或超时 |
|---|---:|---|---|---:|---:|
| iql_independent | 3 | discrete | main / familiar_A_B / all | 0.900 | 1 |
| iql_independent | 3 | discrete | main / random_nominal / all | 0.983 | 1 |
| qmix_no_balance_reward | 3 | discrete | main / familiar_A_B / all | 0.800 | 2 |
| qmix_no_balance_reward | 3 | discrete | main / random_nominal / all | 0.800 | 12 |
| qmix_no_gru | 3 | discrete | main / familiar_A_B / all | 1.000 | 0 |
| qmix_no_gru | 3 | discrete | main / random_nominal / all | 1.000 | 0 |
| qmix_no_mixer_shared | 3 | discrete | main / familiar_A_B / all | 1.000 | 0 |
| qmix_no_mixer_shared | 3 | discrete | main / random_nominal / all | 1.000 | 0 |
| qmix_reference | 3 | discrete | main / familiar_A_B / all | 1.000 | 0 |
| qmix_reference | 3 | discrete | main / random_nominal / all | 1.000 | 0 |
| qmix_static_mixer | 3 | discrete | main / familiar_A_B / all | 1.000 | 0 |
| qmix_static_mixer | 3 | discrete | main / random_nominal / all | 1.000 | 0 |
| qmix_unshared | 3 | discrete | main / familiar_A_B / all | 0.800 | 2 |
| qmix_unshared | 3 | discrete | main / random_nominal / all | 0.800 | 12 |
| qmix_vdn | 3 | discrete | main / familiar_A_B / all | 1.000 | 0 |
| qmix_vdn | 3 | discrete | main / random_nominal / all | 1.000 | 0 |

置信区间按训练种子重采样，条件于固定测试集，不是硬件安全概率保证。
输入 Wh 不代表损耗或效率。先检查终端 SOC 是否可比，再讨论能量差异。
当前仍是数值证据，不能替代物理在线控制、微控制器验证、寿命验证或一般收敛证明。
