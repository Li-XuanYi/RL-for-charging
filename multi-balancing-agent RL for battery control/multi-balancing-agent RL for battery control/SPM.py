import pybamm
import numpy as np
import math


def cal_soc(c):
    return (c - 873.0) / (30171.3 - 873.0)

def normalize_outputs(soc, voltage, temperature):
    norm_soc = (soc - 0.5)*2
    norm_voltage = (voltage - 3.5) / 1
    norm_temperature = (temperature - 308) / 11
    norm_output = np.array([norm_soc, norm_voltage, norm_temperature])

    return norm_output

class SPM:
    def __init__(self, init_v=3.2, init_t=298, init_soc=0.1, param='Chen2020'):
        # 传递参数
        self.reward = None
        self.param = param
        self.sett = {'sample_time': 30*3,
                     'periodic_test': 20,
                     'number_of_training_episodes': 1000,
                     'number_of_training': 3,
                     'episodes_number_test': 10,
                     'constraints temperature max': 309,
                     'constraints voltage max': 4.2}
        # 设置一下日志信息
        # pybamm.set_logging_level("DEBUG")
        # 模型初始化
        options = {
            "thermal": "lumped",
        }

        model = pybamm.lithium_ion.SPMe(options=options)
        param = pybamm.ParameterValues(self.param)
        param["Upper voltage cut-off [V]"] = 4.2

        # 根据所给的电压初始化参数
        param.set_initial_stoichiometries("{} V".format(init_v))
        # param.set_initial_stoichiometries(init_soc)
        # 改变电流函数为输入模型
        param["Current function [A]"] = "[input]"

        c_initial = init_soc * (30171.3 - 873.0) + 873.0
        param["Initial concentration in negative electrode [mol.m-3]"] = c_initial

        self.model = model
        self.param = param
        self.temp = init_t
        self.voltage = init_v
        self.soc = init_soc
        self.soc_d = None
        self.temp_d = None
        self.voltage_d = None
        self.sol = None
        self.info = None
        self.done = False

    def step(self, action, st=None):
        # 连续求解状态替换
        if self.sol is not None:
            self.model.set_initial_conditions_from(self.sol)
        # 仿真设置
        simulation = pybamm.Simulation(self.model, parameter_values=self.param)
        # 时间间隔设置
        if st is not None:
            t_eval = np.linspace(0, st, 2)
        else:
            t_eval = np.linspace(0, self.sett['sample_time'], 2)
        sol = simulation.solve(t_eval, inputs={"Current function [A]": -action})
        self.voltage = sol["Voltage [V]"].entries[-1]
        self.temp = sol["X-averaged cell temperature [K]"].entries[-1]
        c = sol["R-averaged negative particle concentration [mol.m-3]"].entries[-1][-1]
        self.soc = cal_soc(c)
        self.info = sol.termination
        # 数据的更新
        self.sol = sol

    def reset(self, init_v=3.2, init_t=298, init_soc=0.1):
        self.__init__(init_v, init_t, init_soc)

        return

class MultiSPM:

    def __init__(self, num_of_agent, state_shape, obs_shape, n_actions, episode_limit, action_space):
        self.initial_conditions = {}
        # 初始化四个 SPM 电池，每个电池有不同的初始参数
        self.num_of_agent = num_of_agent
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.episode_limit = episode_limit
        self.action_space = action_space

        self.spm1 = SPM(init_v=2.8, init_t=298, init_soc=0.1)
        self.spm2 = SPM(init_v=3.2, init_t=300, init_soc=0.2)
        self.spm3 = SPM(init_v=3.6, init_t=302, init_soc=0.3)

    def get_env_info(self):
        env_info = {
            "n_agents": self.num_of_agent,
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit
        }

        return env_info

    def reset(self):

        self.initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        self.initial_conditions['init_t'] = np.random.uniform(low=298, high=303)
        self.spm1.reset(init_v=self.initial_conditions['init_v'], init_t=self.initial_conditions['init_t'], init_soc=0.3)

        self.initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        self.initial_conditions['init_t'] = np.random.uniform(low=298, high=303)
        self.spm2.reset(init_v=self.initial_conditions['init_v'], init_t=self.initial_conditions['init_t'], init_soc=0.5)
        # self.spm2.param.update({
        #     "Initial inner SEI thickness [m]": 3.5e-09,
        #     "Initial outer SEI thickness [m]": 3.7e-09,
        #     "Negative electrode active material volume fraction": 0.55,
        #     "Positive electrode active material volume fraction": 0.40,
        #     "Typical plated lithium concentration [mol.m-3]": 1500.0,
        # })

        self.initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        self.initial_conditions['init_t'] = np.random.uniform(low=298, high=303)
        self.spm3.reset(init_v=self.initial_conditions['init_v'], init_t=self.initial_conditions['init_t'], init_soc=0.7)
        # self.spm3.param.update({
        #     "Initial inner SEI thickness [m]": 5e-09,
        #     "Initial outer SEI thickness [m]": 5.5e-09,
        #     "Negative electrode active material volume fraction": 0.5,
        #     "Positive electrode active material volume fraction": 0.36,
        #     "Typical plated lithium concentration [mol.m-3]": 2000.0,
        # })

    def get_obs(self):
        norm_out1 = normalize_outputs(self.spm1.soc, self.spm1.voltage, self.spm1.temp)
        norm_out2 = normalize_outputs(self.spm2.soc, self.spm2.voltage, self.spm2.temp)
        norm_out3 = normalize_outputs(self.spm3.soc, self.spm3.voltage, self.spm3.temp)

        multi_obs = [norm_out1, norm_out2, norm_out3]
        return multi_obs

    def get_state(self, obs):
        state = np.concatenate(obs)

        return state

    def get_avail_agent_actions(self, agent_id):
        if agent_id == 0:
            current_soc = self.spm1.soc  # 获取电池 SOC
            current_voltage = self.spm1.voltage  # 获取电池电压
            current_temperature = self.spm1.temp  # 获取电池温度

            avail_actions = np.zeros(len(self.action_space))

            for i, action in enumerate(self.action_space):
                # SOC 限制：如果 SOC > 0.95，只允许放电（action < 0）或不充电（action = 0）
                if current_soc > 0.95:
                    if action <= self.action_space[4]:
                        avail_actions[i] = 1
                    continue  # 直接跳过，避免进一步检查

                # SOC 限制：如果 SOC < 0.05，只允许充电（action > 0）
                if current_soc < 0.05:
                    if action > self.action_space[4]:
                        avail_actions[i] = 1
                    continue  # 直接跳过，避免进一步检查

                # 电压和温度限制：如果电压或温度超过最大值，只允许小电流充放电
                if current_voltage >= self.spm1.sett['constraints voltage max'] or current_temperature >= self.spm1.sett['constraints temperature max']:
                    # 限制充放电电流到一个小范围，比如 -0.5C 到 0.5C
                    if self.action_space[2] <= action <= self.action_space[6]:
                        avail_actions[i] = 1
                else:
                    # 正常情况下，优先选择大电流充电
                    if action >= self.action_space[4]:  # 正电流（充电）
                        avail_actions[i] = 1

            return avail_actions

        if agent_id == 1:
            current_soc = self.spm2.soc  # 获取电池 SOC
            current_voltage = self.spm2.voltage  # 获取电池电压
            current_temperature = self.spm2.temp  # 获取电池温度

            avail_actions = np.zeros(len(self.action_space))

            for i, action in enumerate(self.action_space):
                # SOC 限制：如果 SOC > 0.95，只允许放电（action < 0）或不充电（action = 0）
                if current_soc > 0.95:
                    if action <= self.action_space[4]:
                        avail_actions[i] = 1
                    continue  # 直接跳过，避免进一步检查

                # SOC 限制：如果 SOC < 0.05，只允许充电（action > 0）
                if current_soc < 0.05:
                    if action > self.action_space[4]:
                        avail_actions[i] = 1
                    continue  # 直接跳过，避免进一步检查

                # 电压和温度限制：如果电压或温度超过最大值，只允许小电流充放电
                if current_voltage >= self.spm2.sett['constraints voltage max'] or current_temperature >= self.spm2.sett['constraints temperature max']:
                    # 限制充放电电流到一个小范围，比如 -2C 到 2C
                    if self.action_space[2] <= action <= self.action_space[6]:
                        avail_actions[i] = 1
                else:
                    # 正常情况下，优先选择大电流充电
                    if action >= self.action_space[4]:  # 正电流（充电）
                        avail_actions[i] = 1

            return avail_actions

        if agent_id == 2:
            current_soc = self.spm3.soc  # 获取电池 SOC
            current_voltage = self.spm3.voltage  # 获取电池电压
            current_temperature = self.spm3.temp  # 获取电池温度

            avail_actions = np.zeros(len(self.action_space))

            for i, action in enumerate(self.action_space):
                # SOC 限制：如果 SOC > 0.95，只允许放电（action < 0）或不充电（action = 0）
                if current_soc > 0.95:
                    if action <= self.action_space[4]:
                        avail_actions[i] = 1
                    continue  # 直接跳过，避免进一步检查

                # SOC 限制：如果 SOC < 0.05，只允许充电（action > 0）
                if current_soc < 0.05:
                    if action > self.action_space[4]:
                        avail_actions[i] = 1
                    continue  # 直接跳过，避免进一步检查

                # 电压和温度限制：如果电压或温度超过最大值，只允许小电流充放电
                if current_voltage >= self.spm3.sett['constraints voltage max'] or current_temperature >= self.spm3.sett['constraints temperature max']:
                    # 限制充放电电流到一个小范围，比如 -0.5C 到 0.5C
                    if self.action_space[2] <= action <= self.action_space[6]:
                        avail_actions[i] = 1
                else:
                    # 正常情况下，优先选择大电流充电
                    if action >= self.action_space[4]:  # 正电流（充电）
                        avail_actions[i] = 1

            return avail_actions

    def multi_step(self, actions, beta):

        reward_bal = 0
        terminated = False

        action1 = self.action_space[actions[0]]
        action2 = self.action_space[actions[1]]
        action3 = self.action_space[actions[2]]

        single_obs1, reward1, _ = self.spm1.step(action1)
        single_obs2, reward2, _ = self.spm2.step(action2)
        single_obs3, reward3, _ = self.spm3.step(action3)

        mean_soc = (self.spm1.soc + self.spm2.soc + self.spm3.soc) / 3
        unbal_all_1 = (self.spm1.soc - mean_soc) ** 2
        unbal_all_2 = (self.spm2.soc - mean_soc) ** 2
        unbal_all_3 = (self.spm3.soc - mean_soc) ** 2
        unbal = math.sqrt((unbal_all_1 + unbal_all_2 + unbal_all_3) / 3)

        if unbal > beta:
            reward_bal = -50 * (unbal - beta)
        if unbal <= beta and self.spm1.soc > 0.9 and self.spm2.soc > 0.9 and self.spm3.soc > 0.9:
            terminated = True

        reward = reward1 + reward2 + reward3 + reward_bal

        return reward, terminated

