import numpy as np
from SPM import SPM

# 初始化电池组，使用SPM模型
class BatteryPack:
    def __init__(self, num_cells=3):
        """
        初始化电池组，每个电池采用SPM模型
        """
        self.num_cells = num_cells
        self.cells = [SPM(init_soc=np.random.uniform(0.2, 1)) for _ in range(num_cells)]  # 初始化随机SOC
        self.socs = [cell.soc for cell in self.cells]

    def update_socs(self):
        """
        更新每个电池的SOC
        """
        self.socs = [cell.soc for cell in self.cells]


# 均衡逻辑代码
def calculate_energy_transfer(socs):
    """
    根据相邻电池的SOC计算能量转移方向和均衡电流。
    """
    u = np.zeros(len(socs) - 1)  # 初始化均衡电流数组
    for i in range(len(socs) - 1):
        if socs[i] > socs[i + 1]:  # 如果第i个电池SOC大于第i+1个电池
            u[i] = 2  # 从SOC高的电池流向SOC低的电池，设定均衡电流为2A
        elif socs[i] < socs[i + 1]:  # 如果第i个电池SOC小于第i+1个电池
            u[i] = -2  # 从SOC低的电池流向SOC高的电池
        else:
            u[i] = 0  # 如果SOC相等，不发生能量传递
    return u


def step_balancing(battery_pack, u, sample_time=30):
    """
    执行均衡过程，通过SPM模型的step函数对电池充放电
    """
    for i in range(battery_pack.num_cells - 1):
        # 如果需要从电池i到电池i+1传递能量
        if u[i] > 0:
            battery_pack.cells[i].step(-u[i], st=sample_time)  # 放电
            battery_pack.cells[i + 1].step(u[i], st=sample_time)  # 充电
        elif u[i] < 0:
            battery_pack.cells[i].step(abs(u[i]), st=sample_time)  # 充电
            battery_pack.cells[i + 1].step(-abs(u[i]), st=sample_time)  # 放电

    # 更新电池的SOC
    battery_pack.update_socs()


def is_balanced(socs, threshold=0.01):
    """
    判断电池组是否达到均衡状态。
    """
    return np.max(socs) - np.min(socs) <= threshold


# 初始化电池组
battery_pack = BatteryPack(num_cells=3)
print(f"初始SOC: {battery_pack.socs}")

# 均衡过程模拟
iteration = 0
max_iterations = 100  # 最大迭代次数，避免无限循环
while not is_balanced(battery_pack.socs) and iteration < max_iterations:
    print(f"第{iteration + 1}次迭代:")

    # 计算能量传递方向和均衡电流
    u = calculate_energy_transfer(battery_pack.socs)
    print(f"均衡电流: {u}")

    # 执行均衡步骤
    step_balancing(battery_pack, u, sample_time=30)

    # 打印更新后的SOC
    print(f"更新后的SOC: {battery_pack.socs}")

    # 迭代计数
    iteration += 1

# 输出最终结果
if is_balanced(battery_pack.socs):
    print(f"均衡完成！最终SOC: {battery_pack.socs}")
else:
    print(f"达到最大迭代次数，均衡未完成。最终SOC: {battery_pack.socs}")