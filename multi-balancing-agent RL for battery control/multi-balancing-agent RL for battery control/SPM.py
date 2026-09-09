import pybamm
import numpy as np

from reward import pack_reward, safety_reward, terminal_status


def cal_soc(c):
    return (c - 873.0) / (30171.3 - 873.0)

def normalize_outputs(soc, voltage, temperature):
    norm_soc = (soc - 0.5)*2
    norm_voltage = (voltage - 3.5) / 1
    norm_temperature = (temperature - 308) / 11
    norm_output = np.array([norm_soc, norm_voltage, norm_temperature])

    return norm_output

class SPM:
    def __init__(
        self,
        init_v=3.2,
        init_t=298,
        init_soc=0.1,
        param='Chen2020',
        sample_time=90,
        max_voltage=4.2,
        max_temperature=309.0,
        initialization_mode="soc_consistent",
        parameter_overrides=None,
    ):
        # 传递参数
        self.reward = None
        self.param = param
        self.initialization_mode = initialization_mode
        self.sett = {'sample_time': sample_time,
                     'periodic_test': 20,
                     'number_of_training_episodes': 1000,
                     'number_of_training': 3,
                     'episodes_number_test': 10,
                     'constraints temperature max': max_temperature,
                     'constraints voltage max': max_voltage}
        # 设置一下日志信息
        # pybamm.set_logging_level("DEBUG")
        # 模型初始化
        options = {
            "thermal": "lumped",
        }

        model = pybamm.lithium_ion.SPMe(options=options)
        param = pybamm.ParameterValues(self.param)
        if parameter_overrides:
            param.update(parameter_overrides, check_already_exists=True)
        param["Upper voltage cut-off [V]"] = max_voltage
        param["Initial temperature [K]"] = init_t
        # Reset a previously input-driven current before PyBaMM computes the
        # voltage-based initial stoichiometries during an episode reset.
        param["Current function [A]"] = 0.0

        if initialization_mode == "soc_consistent":
            param.set_initial_state(float(init_soc))
        elif initialization_mode == "legacy_mixed":
            param.set_initial_state("{} V".format(init_v))
        else:
            raise ValueError(f"Unknown initialization mode: {initialization_mode}")
        # 改变电流函数为输入模型
        param["Current function [A]"] = "[input]"

        if initialization_mode == "legacy_mixed":
            c_initial = init_soc * (30171.3 - 873.0) + 873.0
            param["Initial concentration in negative electrode [mol.m-3]"] = c_initial

        self.model = model
        self.param = param
        self.simulation = pybamm.Simulation(
            self.model, parameter_values=self.param
        )
        self.temp = init_t
        self.voltage = init_v
        self.soc = init_soc
        self.soc_d = None
        self.temp_d = None
        self.voltage_d = None
        self.sol = None
        self.info = None
        self.done = False
        if initialization_mode == "soc_consistent":
            inputs = {"Current function [A]": 0.0}
            self.sol = self.simulation.solve(
                np.array([0.0, 1e-6]), inputs=inputs
            )
            self.voltage = self.sol["Voltage [V]"].entries[0]
            self.temp = self.sol[
                "X-averaged cell temperature [K]"
            ].entries[0]
            concentration = self.sol[
                "R-averaged negative particle concentration [mol.m-3]"
            ].entries[0][-1]
            self.soc = cal_soc(concentration)

    def step(self, action, st=None):
        duration = float(st if st is not None else self.sett['sample_time'])
        model_input = -float(action)
        inputs = {"Current function [A]": model_input}
        if self.sol is None:
            sol = self.simulation.solve(
                np.linspace(0.0, duration, 2), inputs=inputs
            )
        else:
            sol = self.simulation.step(
                duration, npts=2, save=False, inputs=inputs
            )
        self.voltage = sol["Voltage [V]"].entries[-1]
        self.temp = sol["X-averaged cell temperature [K]"].entries[-1]
        c = sol["R-averaged negative particle concentration [mol.m-3]"].entries[-1][-1]
        self.soc = cal_soc(c)
        self.info = sol.termination
        # 数据的更新
        self.sol = sol

        observation = normalize_outputs(self.soc, self.voltage, self.temp)
        reward, voltage_reward, temperature_reward = safety_reward(
            [self.voltage],
            [self.temp],
            self.sett['constraints voltage max'],
            self.sett['constraints temperature max'],
        )
        info = {
            "termination": self.info,
            "voltage_reward": voltage_reward,
            "temperature_reward": temperature_reward,
        }
        return observation, reward, info

    def reset(self, init_v=3.2, init_t=298, init_soc=0.1):
        self.__init__(
            init_v,
            init_t,
            init_soc,
            param=self.param,
            sample_time=self.sett['sample_time'],
            max_voltage=self.sett['constraints voltage max'],
            max_temperature=self.sett['constraints temperature max'],
            initialization_mode=self.initialization_mode,
        )

        return

class MultiSPM:

    def __init__(
        self,
        num_of_agent,
        state_shape,
        obs_shape,
        n_actions,
        episode_limit,
        action_space,
        *,
        beta=0.02,
        soc_ref=0.90,
        time_penalty=-0.75,
        balance_penalty_scale=-50.0,
        voltage_penalty_scale=-20.0,
        temperature_penalty_scale=-2.0,
        max_voltage=4.2,
        max_temperature=309.0,
        sample_time=90,
        initial_socs=(0.3, 0.5, 0.7),
        initialization_mode="soc_consistent",
        cell_parameter_overrides=None,
    ):
        if num_of_agent != 3:
            raise NotImplementedError(
                "The released identified-cell environment currently supports three cells. "
                "The larger-n experiment requires an explicit virtual-cell generation rule."
            )
        self.initial_conditions = {}
        # 初始化四个 SPM 电池，每个电池有不同的初始参数
        self.num_of_agent = num_of_agent
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.episode_limit = episode_limit
        self.action_space = action_space
        self.beta = float(beta)
        self.soc_ref = float(soc_ref)
        self.time_penalty = float(time_penalty)
        self.balance_penalty_scale = float(balance_penalty_scale)
        self.voltage_penalty_scale = float(voltage_penalty_scale)
        self.temperature_penalty_scale = float(temperature_penalty_scale)
        self.max_voltage = float(max_voltage)
        self.max_temperature = float(max_temperature)
        self.sample_time = int(sample_time)
        self.initial_socs = tuple(float(value) for value in initial_socs)
        self.initialization_mode = initialization_mode
        if cell_parameter_overrides is None:
            cell_parameter_overrides = ({}, {}, {})
        if len(cell_parameter_overrides) != 3:
            raise ValueError(
                "cell_parameter_overrides must contain three parameter mappings"
            )
        self.cell_parameter_overrides = tuple(cell_parameter_overrides)
        self.last_transition = None

        self.spm1 = SPM(init_v=2.8, init_t=298, init_soc=self.initial_socs[0], sample_time=sample_time, max_voltage=max_voltage, max_temperature=max_temperature, initialization_mode=initialization_mode, parameter_overrides=self.cell_parameter_overrides[0])
        self.spm2 = SPM(init_v=3.2, init_t=300, init_soc=self.initial_socs[1], sample_time=sample_time, max_voltage=max_voltage, max_temperature=max_temperature, initialization_mode=initialization_mode, parameter_overrides=self.cell_parameter_overrides[1])
        self.spm3 = SPM(init_v=3.6, init_t=302, init_soc=self.initial_socs[2], sample_time=sample_time, max_voltage=max_voltage, max_temperature=max_temperature, initialization_mode=initialization_mode, parameter_overrides=self.cell_parameter_overrides[2])

    def get_env_info(self):
        env_info = {
            "n_agents": self.num_of_agent,
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit
        }

        return env_info

    def reset(self, initial_socs=None):

        if initial_socs is not None:
            if len(initial_socs) != 3:
                raise ValueError("initial_socs must contain exactly three SOC values")
            self.initial_socs = tuple(float(value) for value in initial_socs)

        self.initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        self.initial_conditions['init_t'] = np.random.uniform(low=298, high=303)
        self.spm1.reset(init_v=self.initial_conditions['init_v'], init_t=self.initial_conditions['init_t'], init_soc=self.initial_socs[0])

        self.initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        self.initial_conditions['init_t'] = np.random.uniform(low=298, high=303)
        self.spm2.reset(init_v=self.initial_conditions['init_v'], init_t=self.initial_conditions['init_t'], init_soc=self.initial_socs[1])
        # self.spm2.param.update({
        #     "Initial inner SEI thickness [m]": 3.5e-09,
        #     "Initial outer SEI thickness [m]": 3.7e-09,
        #     "Negative electrode active material volume fraction": 0.55,
        #     "Positive electrode active material volume fraction": 0.40,
        #     "Typical plated lithium concentration [mol.m-3]": 1500.0,
        # })

        self.initial_conditions['init_v'] = np.random.uniform(low=2.8, high=3.2)
        self.initial_conditions['init_t'] = np.random.uniform(low=298, high=303)
        self.spm3.reset(init_v=self.initial_conditions['init_v'], init_t=self.initial_conditions['init_t'], init_soc=self.initial_socs[2])
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

    def multi_step(self, actions, beta=None):

        beta = self.beta if beta is None else float(beta)

        action1 = self.action_space[actions[0]]
        action2 = self.action_space[actions[1]]
        action3 = self.action_space[actions[2]]

        self.spm1.step(action1)
        self.spm2.step(action2)
        self.spm3.step(action3)

        socs = [self.spm1.soc, self.spm2.soc, self.spm3.soc]
        voltages = [self.spm1.voltage, self.spm2.voltage, self.spm3.voltage]
        temperatures = [self.spm1.temp, self.spm2.temp, self.spm3.temp]
        reward, components = pack_reward(
            socs,
            voltages,
            temperatures,
            beta=beta,
            time_penalty=self.time_penalty,
            balance_scale=self.balance_penalty_scale,
            max_voltage=self.max_voltage,
            max_temperature=self.max_temperature,
            voltage_scale=self.voltage_penalty_scale,
            temperature_scale=self.temperature_penalty_scale,
        )
        terminated, _ = terminal_status(socs, beta=beta, soc_ref=self.soc_ref)
        self.last_transition = {
            "actions_a": [float(action1), float(action2), float(action3)],
            "socs": [float(value) for value in socs],
            "voltages_v": [float(value) for value in voltages],
            "temperatures_k": [float(value) for value in temperatures],
            "reward_components": components,
            "terminated": terminated,
        }

        return reward, terminated

