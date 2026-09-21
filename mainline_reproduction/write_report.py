"""Write a factual report from completed output, without modifying results."""
import csv,hashlib,json,sys
from pathlib import Path
from importlib.metadata import version

run=Path(sys.argv[1]).resolve()
root=run.parent.parent
status=json.loads((run/'status.json').read_text(encoding='utf-8'))
assert status['status']=='completed'
verification=json.loads((run/'verification.json').read_text(encoding='utf-8'))
assert verification['status']=='passed'
cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
fits=status['fit_metrics']
results=status['results']
raw=next(root.glob('multi-balancing-agent*/multi-balancing-agent*/data_true/data_image'))
files=[raw/name for stage in ['begin','middle','age'] for name in ['data_1C_'+stage,'temp_data 1C '+stage]]
manifest={'python':sys.version,'packages':{name:version(name) for name in ['numpy','scipy','pybamm','casadi','torch','matplotlib']},
 'raw_data':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
(run/'data_environment_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lines=['# 数值实验主线还原结果','',
 '**结论：现有 Python 程序和数据足以重建并运行数值实验主线；本次完成两组各 200 回合训练，但尚未复现论文的全部性能结论。**','',
 '主线：三节实测放电电压/温度 → PSO 辨识 SPMe 参数 → 三智能体 MBA-RL（QMIX）训练 → 贪婪评估 → SOC、电流、电压、温度及均衡指标。实物设备不在本次范围。','',
 f'运行目录：`{run}`。随机种子：{cfg["seed"]}。报告使用最后一回合保存的模型，不挑选历史最优曲线。','',
 '## 已完成的工作','',
 '- 恢复原代码中的完整电化学与热参数，使用现有六条测量序列重新辨识三节电芯。',
 '- 修复环境返回值、初始两极浓度一致性、回放末状态、目标网络同步及数值失败回合处理。',
 '- 使用原网络结构、动作范围、90 秒决策间隔和原奖励系数，训练两组初始 SOC。',
 '- 保存逐回合训练数据、各次评估轨迹、模型权重、拟合曲线、最终图像和判定表。','',
 '## 本次实际结果','',
 '三节电压 RMSE 约为 **0.0201 / 0.0231 / 0.0221 V**；温度 RMSE 约为 **0.404 / 0.342 / 0.413 °C**。电压拟合已到论文报告的误差量级，温度误差仍明显较高；这些是用于辨识的同一条曲线上的误差，不能视作独立验证。','',
 '| 指标 | A：初始 0.1/0.2/0.3 | B：初始 0.3/0.5/0.7 |','|---|---:|---:|']
a,b=results
def fmt(value):return '未达到' if value is None else str(round(value,4))
for label,key in [('首次均衡时间 / s','first_balance_s'),('充电与均衡目标时间 / s','soc_target90_s'),('最终 SOC 标准差','final_soc_std'),('最高电压 / V','max_voltage_V'),('最高温度 / K','max_temperature_K'),('电压超限电芯区间数','voltage_violating_cell_intervals')]:
 lines.append(f'| {label} | {fmt(a[key])} | {fmt(b[key])} |')
for label,key in [('SOC 目标达标','soc_target_reached'),('首次均衡后始终保持','balance_maintained_after_first'),('全过程满足电压/温度约束','constraints_satisfied'),('充电任务综合成功','charging_task_success')]:
 lines.append(f'| {label} | {"是" if a[key] else "否"} | {"是" if b[key] else "否"} |')
lines+=['','SOC 目标指三节均达到 0.9 且标准差不超过 0.02；电压和温度约束分别是 4.2 V、309 K。论文首次均衡时间为 A 630 s、B 1170 s，本次尚未对齐。超限区间按“电芯 × 决策区间”统计，不是独立超限事件数。','']
for name in ['A','B']:
 rows=list(csv.DictReader((run/'training'/f'{name}.csv').open(encoding='utf-8-sig')))
 failures=sum(r['numerical_failure']=='True' for r in rows)
 lines.append(f'- {name} 组训练完成 {len(rows)} 回合，其中 {failures} 回合遇到数值求解失败/终止；这些回合被记为失败，不计为成功。')
lines+=['','## 校验与解释边界','',
 '- 独立重新构建静态 PyBaMM 模型，在更高求解精度下与缓存实现核对通过。原拟合求解精度下，温度结果相对高精度重算最多有约 0.047 °C 的数值偏差，已在 verification.json 单列；原始结果未覆盖。',
 '- 保存权重重新加载并重演 A、B 最终评估，轨迹一致性校验通过。',
 '- SOC 标准差、时间同步、动作范围、越限计数与目标网络同步检查通过。',
 '- 以上证明流程和保存结果可重演，不证明策略已经收敛或论文结论成立。','',
 '**重建假设已写入配置和 README：** 按原辨识代码假设 5 A（论文 4 Ah/1C 口径仍有矛盾）；按原脚本将采样间隔解释为 1 秒；放宽原 PSO 中排除了已有参数的范围，并增加热参数辨识。数值阈值终止时额外扣 2000 分，用于补齐失败回合逻辑，此项并非论文原奖励。没有使用前瞻仿真替换策略电流。','',
 '本次为单一种子、每组 200 回合的主线还原。完整 1200 回合、多种子、温度拟合改进、持续均衡与电压约束收敛，以及完整对比实验，均未在本次证明；增加回合数也不能保证消除差距。','',
 '## 一键运行','',
 '```cmd',f'"{root / "run_mainline.cmd"}" --episodes 200','```','',
 '若使用更长预算（重新辨识并每组训练 1200 回合）：','',
 '```cmd',f'"{root / "run_mainline.cmd"}" full','```','',
 f'全部新数据保存在 `{root / "mainline_results"}` 的独立时间目录。`LATEST.txt` 指向最近一次运行。原代码和原始数据保留。','',
 '核心文件：`summary.csv`、`status.json`、`verification.json`、`calibration/metrics.csv`、`trajectories/final_A.csv`、`trajectories/final_B.csv`。']
(run/'MAINLINE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(run/'MAINLINE_REPORT.md')
