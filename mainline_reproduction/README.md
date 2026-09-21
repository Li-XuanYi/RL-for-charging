# 数值实验主线还原

本版只处理“原始放电数据 → 电池模型辨识 → MBA-RL 训练 → 论文关键指标”。不连接实物设备，也不将缺失的 DQL 用其他算法冒充。原版初步脚本和历史结果仍在 reproduction / reproduction_results；本版在 mainline_reproduction / mainline_results。

## 一键命令

```cmd
"C:\Users\zhengzihao\Desktop\RL-for-charging-main\run_mainline.cmd" --episodes 200
```

默认每节电芯 20 粒子、35 次 PSO 迭代；分别训练 Set A、Set B，各 200 回合，固定随机种子 7，每 25 回合评估并保存模型。它是主线还原预算，不能直接称为论文 1200 回合实验。

```cmd
"C:\Users\zhengzihao\Desktop\RL-for-charging-main\run_mainline.cmd" full
```

full 设置为每节电芯 30 粒子、100 次 PSO 迭代，每组训练 1200 回合。也可使用 `--seed 8` 更换随机种子。固定种子的重复不等于跨随机种子鲁棒性。

已经辨识过参数时可跳过重复辨识：

```cmd
run_mainline.cmd --episodes 200 --fit-from "C:\Users\zhengzihao\Desktop\RL-for-charging-main\mainline_results\具体运行目录\calibration"
```

仅辨识：`run_mainline.cmd --fit-only`。默认依原代码使用 5 A，若要检查论文 4 A 口径可加 `--current 4`，复用参数时电流设置必须相同。首次安装仍借助原 reproduction/bootstrap.py 的隔离环境；全部计算在 .venv-reproduce 内。

## 还原依据与改动

1. 原 data_true/main.py 给出的完整参数字典被保存为 original_parameters.json，包括热力学因子、密度、比热、换热系数等；上一版遗漏的参数恢复。保留 SPMe、Chen2020 基础参数和原空间网格 30/30/30/10/10。
2. 辨识按原脚本使用 5 A，并保留默认初始锂浓度。原文“4 Ah 电芯的 1C”与代码“5 A”相互矛盾，**实际测量电流仍未确认**。diagnostics/current_probe.json 展示原硬编码参数在 4/5 A 下的对照。它支持代码路径的选择，不证明实际实验电流。
3. 原参数中的负极活性材料比例 0.56，不在原 PSO 的 [0.60,0.65] 范围内。本版的范围包含原文件已有数值。粒径固定为原文件中的 3.98、4.18 微米；重新辨识正负极扩散率、正负极活性材料比例、换热系数及原比热的共同缩放系数，共 6 个自由参数。
4. 拟合目标明确为 `(V_RMSE/0.03)^2 + (T_RMSE/0.3)^2 + 100*(缺失时间比例)^2`。在全部测量时间上计算误差；模型提前终止后按末值比较并记录缺失区间，不通过截短实测曲线降低误差。拟合误差为同一曲线上的误差，未作独立实验验证。
5. 使用原 NN.py 的 MLP-GRU-MLP、单调 QMIX 和超网络。修复原工程的环境返回值、末状态、时间截断和目标网络同步逻辑；观测 SOC/V/T、动作 0~7.5 A、间隔 0.5 A、决策周期 90 秒，奖励 -0.75/-50/-20/-2 均按论文。RMSprop、2e-4、gamma=0.99、batch=128、回放800、目标网络每50回合同步；探索从0.5衰减，保留原 main.py 思路。
6. 初始化两极浓度时使用一致的化学计量关系，SOC 端点以 4.2 V 物理参考计算。修正原代码“先用随机初始电压设置两极、再仅覆盖负极浓度”的矛盾。
7. **移除上一版的前瞻电流替换。** 策略选择多少电流就施加多少。保留原代码已有的当前状态动作筛选，并适配非负动作：SOC>=0.95时只允许0 A；V>=4.2或T>=309时只允许0~1 A。电压和温度超限如实惩罚、记录。
8. 为观察软约束越限，数值求解器的停止阈值单独设为 4.5 V/330 K；这不是允许的充电安全上限，也不用于定义 SOC。达到数值阈值时，所有电芯在同一实际时刻终止本回合，记录为失败后继续下一回合；如果求解器无法产生有效区间，则记录0秒失败，保留原状态，不伪造后续数据。为避免通过失败提前结束获取更高奖励，失败回合额外扣2000分。这个终止处理和处罚是补齐原代码缺失逻辑的重建假设，不是论文原有奖励项。真正约束检查仍为4.2 V/309 K。
9. 控制环境缓存初始模型和变量计算函数，仅减少重复计算；已逐项核对缓存输出与 PyBaMM ProcessedVariable.entries 一致，不改变物理模型。

## 输出与判定

每次在 mainline_results 下建立独立时间目录，LATEST.txt 指向最近一次。

- calibration：三节参数、拟合曲线、原始/拟合 CSV、PSO 历史和 RMSE。
- training：逐回合奖励/损失/SOC目标/约束标志，以及每25回合的贪婪评估。
- checkpoints：每25回合的模型文件。当前没有精确断点续训入口。
- trajectories：未训练策略、各次评估、最终策略的原始时序数据。
- figures：SOC、电流、电压、温度、均衡标准差与训练曲线。
- summary.csv：首次均衡、达到90%时间、再次失衡、约束越限、充电电量等指标。
- config.json / status.json / run.log：本次假设、设置、完成状态和日志。

`soc_target_reached` 只表明最终 SOC 条件达标。`constraints_satisfied` 要求整个评估轨迹满足电压/温度约束。两者同时成立才记为 `charging_task_success`。持续均衡另看 `balance_maintained_after_first`。`status=completed` 不等于论文复现成功。

本版先还原主线，尚未覆盖独立泛化测试、完整DQL/CC-CV对比、硬件控制和寿命验证。对照论文数值应读取 config.json 中的 paper_reference，而不是把程序结束视为对齐。

完成后，可独立校验拟合模型、保存权重重演和指标一致性：

```cmd
"C:\Users\zhengzihao\Desktop\RL-for-charging-main\.venv-reproduce\Scripts\python.exe" "C:\Users\zhengzihao\Desktop\RL-for-charging-main\mainline_reproduction\verify_mainline.py" "C:\Users\zhengzihao\Desktop\RL-for-charging-main\mainline_results\具体运行目录"
```

检查结果写入运行目录的 verification.json。检查通过只代表实现及保存数据一致，不代表策略收敛。
