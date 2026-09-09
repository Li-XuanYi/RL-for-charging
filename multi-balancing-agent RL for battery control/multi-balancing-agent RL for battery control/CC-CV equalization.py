import matplotlib.pyplot as plt
import numpy as np
import math
from SPM import MultiSPM


def get_current(soc_diff):
    current = 0
    if soc_diff >= 0.1:
        current = 3.8
    if soc_diff >= 0.075 and soc_diff < 0.1:
        current = 2.8
    if soc_diff >= 0.05 and soc_diff < 0.075:
        current = 1.8
    if soc_diff < 0.05:
        current = 1

    current += np.random.uniform(-0.5, 0.5)

    return current

Multi_battery = MultiSPM(3, 9, 3, 19, 50, np.arange(-2, 7.5, 0.5))

battery1 = Multi_battery.spm1
battery2 = Multi_battery.spm2
battery3 = Multi_battery.spm3

CC_CURRENT = 4
CV_VOLTAGE = 4.1
beta = 0.02
balanced = False

# 数据记录
soc1, soc2, soc3 = [battery1.soc], [battery2.soc], [battery3.soc]
flag1, flag2, flag3 = False, False, False
i = 0

while not balanced:

    i += 1

    if not flag1 and battery1.voltage >= CV_VOLTAGE:
        flag1 = True
    if not flag2 and battery2.voltage >= CV_VOLTAGE:
        flag2 = True
    if not flag3 and battery3.voltage >= CV_VOLTAGE:
        flag3 = True

    if flag1 or flag2 or flag3:
        i -= 1
        if battery1.soc > battery2.soc:
            current = get_current(battery1.soc - battery2.soc)
            battery1.step(-1 * current)
            battery2.step(current)
        else:
            current = get_current(battery2.soc - battery1.soc)
            battery1.step(current)
            battery2.step(-1 * current)

        if battery2.soc > battery3.soc:
            current = get_current(battery2.soc - battery3.soc)
            battery2.step(-1 * current)
            battery3.step(current)
        else:
            current = get_current(battery3.soc - battery2.soc)
            battery2.step(current)
            battery3.step(-1 * current)

    else:
        battery1.step(CC_CURRENT)
        battery2.step(CC_CURRENT)
        battery3.step(CC_CURRENT)

    # 记录 SOC 数据
    soc1.append(battery1.soc)
    soc2.append(battery2.soc)
    soc3.append(battery3.soc)

    soc_mean = (battery1.soc + battery2.soc + battery3.soc) / 3
    unbal = math.sqrt(((battery1.soc - soc_mean) ** 2 +
                       (battery2.soc - soc_mean) ** 2 +
                       (battery3.soc - soc_mean) ** 2) / 3)

    if unbal < beta:
        balanced = True

episodes = [i * 90 for i in range(len(soc1))]
print(len(soc1))
# episodes = range(1, len(soc1)+1)
plt.plot(episodes, soc1, linestyle='-', color='orange', marker="o", markersize=7.5, markerfacecolor="none", markeredgewidth=2, markevery=4, linewidth=3, label="Battery1")
plt.plot(episodes, soc2, linestyle='-', color='r', marker="d", markersize=7.5, markerfacecolor="none", markeredgewidth=2, markevery=4, linewidth=3, label="Battery2")
plt.plot(episodes, soc3, linestyle='-', color='b', marker="s", markersize=7.5, markerfacecolor="none", markeredgewidth=2, markevery=4, linewidth=3, label="Battery3")
plt.axvline((len(soc1)-1)*90, color="g", linestyle="--", linewidth=3)
plt.axvline(i*90, color="y", linestyle="--", linewidth=3)
print(i)
print(soc1[-1], soc2[-1], soc3[-1])

ax = plt.gca()
ax.spines['top'].set_linewidth(3)  # 调整上边框线宽
ax.spines['right'].set_linewidth(3)  # 调整右边框线宽
ax.spines['left'].set_linewidth(3)  # 调整左边框线宽
ax.spines['bottom'].set_linewidth(3)

plt.xlabel('Time/s', fontweight="bold", fontsize=14)
plt.ylabel('SOC', fontweight="bold", fontsize=15)

plt.xticks(fontsize=15, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.legend(fontsize=17, loc="lower right")
plt.savefig("age-middle-new SOC_1", dpi=300)
plt.show()