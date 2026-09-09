# MBA-RL 补充实验冻结协议

状态：设计冻结，结果待运行  
对应审稿意见：M2、M3、M6、M8、D2、D3、D4、P2、P4  
冻结日期：2026-08-29

## 1. 研究问题与判定边界

补充实验只回答以下问题：

1. 论文中的完整奖励、1200 episodes 和每 50 episodes 更新目标网络能否由公开代码复现？
2. 在相同的独立电流源接口、动作集合、状态信息、安全约束和终止定义下，QMIX 是否优于 VDN、DQL 和简单规则控制？
3. QMIX 的结果对训练随机种子、奖励分量及温度阈值是否稳健？
4. 结论应限定为三电芯仿真，除非另行完成经过定义的 larger-`n` 实验。

本协议不把 NEWARE 参数辨识数据描述为控制器硬件验证，也不恢复能效或大型电池包可扩展性主张。

## 2. 实验单位、随机化与阻断

- 学习方法的独立重复单位是完整训练过程，而不是 episode、时间步或电芯。
- QMIX、VDN 和匹配架构 DQL 使用五个训练种子：7、17、29、43、71。
- SOC Set A 与 Set B 构成两个预先指定的场景层。
- 相同 seed 和 SOC set 的方法构成配对 block。初始环境扰动、测试重置和评价规则必须一致。
- 确定性比例控制和 per-cell CC-CV 不把五个 seed 宣称为训练重复；这些 seed 仅表示与学习方法配对的环境 block。
- 运行顺序由 `experiments/build_manifest.py` 使用 seed 20260829 随机化，以避免设备温度、软件更新或人工选择顺序与方法混杂。

## 3. Phase 0：代码与结果来源复核

### E0.1 论文配置

- episodes：1200
- target update interval：50 training episodes
- action grid：-2.0 A 至 7.0 A，步长 0.5 A
- decision interval：90 s
- `beta=0.02`
- `SOC_ref=0.90`
- `V_max=4.20 V`
- `T_max=309 K`
- 完整奖励：time + balance + voltage + temperature

### E0.2 旧配置来源调查

分别保留 800/200 和 800/every-step 两组 provenance runs。它们只用于判断旧图可能来自哪一版实现，不进入方法优越性的主分析。若没有任何配置能够重现旧图，正文必须用修复后结果替换旧图并在回复信中说明。

旧实现还同时使用给定 SOC、随机初始电压以及仅覆盖负极的浓度重写，可能形成不一致的初始电化学状态。provenance runs 保留 `legacy_mixed` 初始化以调查旧图来源；所有 confirmatory、ablation 和 sensitivity runs 使用 PyBaMM 的 `soc_consistent` 初始化，同时设置正负极初始状态。两种初始化的结果不得合并。

### E0.3 通过标准

- 奖励与终止条件单元测试通过；
- 每次运行保存命令、seed、软件版本、训练历史、模型、evaluation trace 和 summary；
- 从空结果目录执行同一命令可得到数值容差内一致的结果；
- 论文表格、代码默认值和归档配置完全一致。

### E0.4 三电芯辨识参数门槛

当前 controller environment 对三个 cell 都继承 Chen2020，未发现三份可加载的 PSO 最优参数归档；`data_true/main.py` 只包含一组手工参数。该候选值与归档PSO边界存在冲突，PSO目标也只使用电压而未使用温度数据。旧版PyBaMM扩散率参数键在当前版本中已更名，加载器会显式迁移并记录键名，避免参数静默失效。正式 confirmatory runs 因此保持暂停。作者需提供 Battery1--Battery3 各自的参数名称、数值、单位、辨识数据来源和运行编号，并写入 `experiments/cell_parameters.identified.json`。加载器会记录文件 SHA-256。`experiments/identify_cell_parameters.py` 可执行新的确定性电压辨识，但新结果不是原最优值恢复，必须经过作者批准。若无法恢复或重新批准三组参数，论文必须改写为共享 Chen2020 参数的仿真，并撤回“不同 aging states 的 identified models”这一证据主张。

## 4. Phase 1：公平的主对照实验

方法：

1. MBA-RL/QMIX；
2. VDN，共享相同 DRQN agent，只把 mixer 改为加和；
3. matched DQL：集中式单智能体DRQN读取9维联合状态，并在同一 per-cell current interface 上选择三个电芯电流的笛卡尔积动作；
4. proportional feedback：每个 cell 根据 `SOC_ref-SOC_i` 产生电流；
5. per-cell CC-CV：每个 cell 使用相同独立电流源接口。

所有方法共同使用：

- 相同三组已归档、可加载的 identified-cell parameter sets；
- 相同两个初始 SOC vectors；
- 相同动作分辨率与 action mask；
- 相同 90 s 控制周期；
- 相同 voltage/temperature constraints；
- 同时报告首次达到 `sigma <= beta` 和全部 cells 达到 `SOC_ref` 的时间；
- 最大 50 个控制周期；超时作为失败而不是删失后忽略。

matched DQL 已完成实现级测试，但在正式多种子结果生成前仍不能声称算法比较充分。VDN 用于隔离 QMIX mixer 的贡献；规则方法用于回答“简单控制是否已足够”。

## 5. Phase 2：奖励消融

以 full QMIX 为参照，分别训练：

- no-time：`r_time=0`；
- no-balance：balance scale = 0；
- no-safety：voltage 与 temperature reward scale = 0，但硬 action mask 保留。

每个配置在两个 SOC sets 上运行五个训练种子。消融只改变指定奖励分量，网络、训练预算、探索调度和环境保持不变。若 no-safety 产生约束违反，只用于说明奖励作用，不作为可部署策略。

## 6. Phase 3：温度阈值敏感性

固定其他参数，比较 306、309 和 312 K。309 K 的 full-QMIX 主实验直接复用，不重复计数。评价重点是：

- time-to-balance 与 time-to-reference-SOC；
- 最大温度和超阈值步数；
- 终止成功率；
- SOC dispersion。

该实验只能证明 309 K 附近的局部稳健性；工程阈值的物理合理性仍需要电芯规格或安全文献支持。

## 7. 评价指标

### Primary outcomes

- `time_to_balance_s`；
- `time_to_reference_soc_s`；
- joint terminal success rate。

### Safety and quality outcomes

- terminal SOC standard deviation；
- maximum voltage and temperature；
- voltage/temperature violation steps；
- charge and discharge energy throughput。

Energy throughput 是描述性负荷指标，不等于 converter efficiency，不能用于声称系统能效提升。

## 8. 统计分析

- 每个 learned method 与场景报告 mean、SD 和 bootstrap 95% CI。
- 以相同 seed 的结果计算 paired differences，并报告差值的 bootstrap 95% CI。
- 由于每组仅五个训练种子，主分析不依赖正态性检验或单一 p-value。
- 超时运行保留为失败；时间指标可报告上限值并同时给出成功率，不得删除失败种子。
- 时间步、电芯和同一模型上的重复测试不得当作独立样本。

## 9. Larger-n 实验的暂停条件

当前环境只有三个明确的 identified-cell models。不能简单复制同一电芯模型并把复制品当作独立异质电芯。执行 n=5 或 n=10 前，作者必须选择并记录以下一种生成规则：

1. 新增实测电芯并重新辨识参数；或
2. 根据三组已辨识参数定义有界虚拟电芯分布，并进行分布敏感性分析。

在规则获得作者确认前，P2 保持“设计完成、执行暂停”，正文继续限定为 three-cell simulation。

## 10. 提交门槛

- M2/M3：Phase 0 全部通过；
- M6：五个独立训练种子全部完成；
- D2/D3/D4：matched DQL、VDN 与规则基线均完成，且终止指标一致；
- M8：三个奖励消融完成；
- P4：306/309/312 K 敏感性完成并给出工程依据；
- P2：若不做 larger-n，回复信撤回实验承诺并维持三电芯边界。

任何未运行项目只能写成 protocol 或 future work，不得在摘要、结果和结论中使用完成时态。
