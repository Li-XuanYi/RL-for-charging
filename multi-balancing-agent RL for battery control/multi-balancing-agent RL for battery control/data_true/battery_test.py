import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# data1 = np.loadtxt("Mohtat2020")
# data2 = [i+3.5 for i in data1]
# np.savetxt("data2", data2)
# plt.plot(data2)
# plt.show()
# data2 = np.loadtxt("data_1C_middle")
# data3 = np.loadtxt("data_1C_age")

# x_old = np.linspace(0, 1, 114)
# x_new = np.linspace(0, 1, 700)
# f = interp1d(x_old, data1, kind='linear')  # 使用线性插值
# data2_interp = f(x_new)  # 插值后的数据2
# np.savetxt("new NCA_Kim2011 reward", data2_interp)

# episodes = range(1, len(data1) + 1)
# plt.figure(figsize=(12, 9))
# plt.plot(data1, linestyle='-', color='orange', linewidth=6, label="Battery 1")
# plt.plot(data2, linestyle='-', color='r', linewidth=6, label="Battery 2")
# plt.plot(data3, linestyle='-', color='b', linewidth=6, label="Battery 3")
#
# ax = plt.gca()
# ax.spines['top'].set_linewidth(5)  # 调整上边框线宽
# ax.spines['right'].set_linewidth(5)  # 调整右边框线宽
# ax.spines['left'].set_linewidth(5)  # 调整左边框线宽
# ax.spines['bottom'].set_linewidth(5)
#
# plt.xlabel('Time/s', fontweight="bold", fontsize=34)
# plt.ylabel('Voltage/V', fontweight="bold", fontsize=35)
#
# plt.xticks(fontsize=35, fontweight='bold')
# plt.yticks(fontsize=36, fontweight='bold')
# plt.legend(fontsize=36)
# plt.savefig("Voltage", dpi=300)
#
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置种子保证结果可重复
# np.random.seed(42)
#
# # 生成示例数据
Marquis = np.loadtxt("Marquis2019")
Mohtat = np.loadtxt("Mohtat2020")
Ai = np.loadtxt("Ai2020")
OKane = np.loadtxt("OKane2022")

Marquis = [i-0.29 for i in Marquis]
Ai = [i+1.8 for i in Ai]

episodes = range(1, len(Marquis)+1)
plt.figure(figsize=(10, 6))
plt.plot(episodes, Marquis, color='b', linewidth=2, label="Marquis2019")
plt.plot(episodes, Mohtat, color='orange', linewidth=2, label="Mohtat2020")
plt.plot(episodes, Ai, color='g', linewidth=2, label="Ai2020")
plt.plot(episodes, OKane, color='purple', linewidth=2, label="OKane2022")
#
# # 设置标签和标题
plt.xlabel('Episode', fontsize=20, fontweight='bold')
plt.ylabel('Cumulative reward', fontsize=20, fontweight='bold')
plt.grid(True)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

# 添加图例
plt.legend(fontsize=12)
plt.savefig("training batteries", dpi=300)
# 显示图形
plt.show()

# data1 = np.loadtxt("meta-NCA_Kim2011 200")
# data1_new = np.loadtxt("meta-NCA_Kim2011 200 new")
#
# data1_all = [(x + y)/2 for x, y in zip(data1, data1_new)]
# data1_all_new = [x + 3 for x in data1_all]
#
# data2_all_1 = np.loadtxt("datadata")
#
# plt.plot(data1_all_new, linewidth=1, color='r', label='meta-agent')
# plt.plot(data2_all_1, linewidth=1, color='b', label='agent')
# plt.legend(fontsize=18, loc="lower right")
# ax = plt.gca()
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)
# #
# plt.xlabel('Iteration', fontweight="bold", fontsize=12)
# plt.ylabel('Cumulative reward', fontweight="bold", fontsize=12)
#
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=13, fontweight='bold')
# plt.grid(True)
# plt.savefig('NCA_Kim interact 200', dpi=300)
# plt.show()