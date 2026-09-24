# 探索版结果：不作为正式论文比较结论

每组 2500 步；种子 [7, 17]；独立目录 experiment_2_exploratory_2500_2seeds。

# 实验 2 安全与持续均衡结果

计算已完成。是否支持论文结论请查看 scientific_assessment.json 和 group_summary.csv。
原策略与同权重加安全层构成配对对照；安全层参与训练的版本独立训练；规则组仅运行一次，不伪造多种子重复。

| 决策周期 s | 方法 | 测试类别 | 安全任务成功率 | 经验约束满足率 |
|---:|---|---|---:|---:|
| 90 | qmix_frozen_shield | nominal_unseen | 0.750 | 1.000 |
| 90 | qmix_frozen_shield | reference_seen | 0.500 | 1.000 |
| 90 | qmix_frozen_shield | unseen_parameter_shift | 0.625 | 1.000 |
| 90 | qmix_raw | nominal_unseen | 0.042 | 0.042 |
| 90 | qmix_raw | reference_seen | 0.000 | 0.250 |
| 90 | qmix_raw | unseen_parameter_shift | 0.000 | 0.042 |
| 90 | qmix_trained_shield | nominal_unseen | 1.000 | 1.000 |
| 90 | qmix_trained_shield | reference_seen | 1.000 | 1.000 |
| 90 | qmix_trained_shield | unseen_parameter_shift | 1.000 | 1.000 |
| 90 | rule_shield | nominal_unseen | 1.000 | 1.000 |
| 90 | rule_shield | reference_seen | 1.000 | 1.000 |
| 90 | rule_shield | unseen_parameter_shift | 1.000 | 1.000 |

安全层为有限时域采样预测筛选，未证明不变集或全局安全；未作实物测试。
持续均衡只统计充电阶段，270秒终端保持另列。首次均衡恰好位于终点时，充电阶段持续时间为0。
失败/超时样本全部保留。成功样本充电时间是条件统计，成功率不同时不能据此直接排名。
端口输入 Wh 不是能量损耗；各节 Ah 之和不是串联电池包容量；没有寿命改善结论。
5个以上种子是预定正式规模，较小配置只能用于探索。安全层参数为预定工程假设，不是已由实验1确认的置信误差界。
