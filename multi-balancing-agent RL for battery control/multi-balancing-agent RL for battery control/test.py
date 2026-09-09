import matplotlib.pyplot as plt

# 示例数据
x = [0, 1, 2, 3, 4, 5]
y1 = [0.4, 0.6, 0.8, 0.9, 1.0, 1.1]
y2 = [0.5, 0.7, 0.85, 0.95, 1.05, 1.1]
y3 = [0.3, 0.5, 0.7, 0.85, 0.9, 1.0]
y4 = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

# 绘制不同 marker 的曲线
plt.plot(x, y1, marker='*', linewidth=3, label='Line 1: *')  # 星型
plt.plot(x, y2, marker='d', label='Line 2: d')  # 小菱形
plt.plot(x, y3, marker='o', label='Line 3: o')  # 圆形
plt.plot(x, y4, marker='x', label='Line 4: x')  # 叉号

# 添加图例和显示
plt.legend()
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('SOC')
plt.show()