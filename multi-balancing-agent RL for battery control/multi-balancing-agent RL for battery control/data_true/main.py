import pybamm
import numpy as np
import matplotlib.pyplot as plt

# 读取实测数据
U_1C = np.loadtxt("image/data_1C_age")
T_1C = np.loadtxt("image/temp_data 1C begin")
Time_1C = np.arange(0, U_1C.size)

# 设置模型参数网格进行仿真
# pybamm.set_logging_level("INFO")
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"}, name="lumped thermal model")
geometry = model.default_geometry
# 选去相关参数
param = pybamm.ParameterValues("Chen2020")
param.update(
    # 设置为输入
    {"Current function [A]": "[input]",
     "Positive electrode diffusivity [m2.s-1]": 5.81e-15,
     "Negative electrode diffusivity [m2.s-1]": 4.32e-14,
     "Positive particle radius [m]": 3.98e-6,
     "Negative particle radius [m]": 4.18e-6,
     'Negative electrode active material volume fraction': 0.560,
     'Positive electrode active material volume fraction': 0.526,

     'Thermodynamic factor': 0.3,
     "Negative current collector specific heat capacity [J.kg-1.K-1]": 1200.0,
     "Positive current collector specific heat capacity [J.kg-1.K-1]": 1500.0,
     "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
     "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
     'Negative electrode density [kg.m-3]': 2657.0,
     'Positive electrode density [kg.m-3]': 3262.0,
     'Negative electrode specific heat capacity [J.kg-1.K-1]': 3200.0,
     'Positive electrode specific heat capacity [J.kg-1.K-1]': 2200.0,
     'Separator specific heat capacity [J.kg-1.K-1]': 2500.0,
     "Total heat transfer coefficient [W.m-2.K-1]": 25.0,

     "Ambient temperature [K]": "[input]",
     "Initial temperature [K]": "[input]",
     }
)
param.process_geometry(geometry)
param.process_model(model)
# 网格设置
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
# solve model
t_eval = np.linspace(0, 7000, 7001)

solution = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3).solve(model, t_eval,
                                                                        inputs={"Current function [A]": 5 * 1,
                                                                                "Ambient temperature [K]": 298.15,
                                                                                "Initial temperature [K]": 298.15})
volt_1C = solution["Voltage [V]"].entries
time_1C = solution["Time [s]"].entries
temp_1C = solution["Cell temperature [C]"].entries

temp_1C = np.mean(temp_1C, 0)
# number = min(len(temp_1C), len(T_1C))

# temp_test = T_1C[1:number-30]
# temp_test2 = temp_1C[1:number-30]
# loss = np.sqrt(np.mean((temp_test - temp_test2) ** 2))
# print(loss)

#
number = min(len(volt_1C), len(U_1C))
time_1C = time_1C[:number+30]
volt_1C = volt_1C[:number+30]
Time_1C = Time_1C[:number]
U_1C = U_1C[:number]
#
plt.figure(figsize=(12, 9))
plt.plot(time_1C, volt_1C, linewidth=7, color="y", label="Battery3-model")
plt.plot(Time_1C[1:Time_1C.size:60], U_1C[1:Time_1C.size:60], 'o', markersize=13.5, color="purple", label="Battery3-measured")

plt.xlabel("Time/s", fontsize=34, fontweight="bold")
plt.ylabel("Voltage/V", fontsize=35, fontweight="bold")
plt.xticks(fontsize=35, fontweight='bold')
plt.yticks(fontsize=36, fontweight='bold')

plt.legend(loc='lower left', fontsize=36)
ax = plt.gca()
ax.spines['top'].set_linewidth(5)  # 调整上边框线宽
ax.spines['right'].set_linewidth(5)  # 调整右边框线宽
ax.spines['left'].set_linewidth(5)  # 调整左边框线宽
ax.spines['bottom'].set_linewidth(5)

plt.savefig("Battery3_volt", dpi=300)
plt.show()

# plt.figure(figsize=(12, 9))
# plt.plot(time_1C, temp_1C, linewidth=7, label="Battery1-model")
# plt.plot(Time_1C[1:Time_1C.size:60], T_1C[1:Time_1C.size:60],  'o', markersize=15.5, label="Battery1-measured")
# #
# plt.xlabel("Time/s", fontsize=34, fontweight="bold")
# plt.ylabel("Voltage/V", fontsize=35, fontweight="bold")
# plt.xticks(fontsize=35, fontweight='bold')
# plt.yticks(fontsize=36, fontweight='bold')
#
# plt.legend(loc='lower right', fontsize=36)
# ax = plt.gca()
# ax.spines['top'].set_linewidth(5)  # 调整上边框线宽
# ax.spines['right'].set_linewidth(5)  # 调整右边框线宽
# ax.spines['left'].set_linewidth(5)  # 调整左边框线宽
# ax.spines['bottom'].set_linewidth(5)
#
# plt.savefig("Battery1_volt", dpi=300)
# plt.show()
