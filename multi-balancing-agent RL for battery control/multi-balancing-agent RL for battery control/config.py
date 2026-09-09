import torch

class Config:
    def __init__(self):
        self.train = True
        self.seed = 7
        self.cuda = True

        # train setting
        self.last_action = True  # 使用最新动作选择动作
        self.reuse_network = True  # 对所有智能体使用同一个网络
        self.n_epochs = 800
        self.evaluate_epoch = 100
        self.evaluate_number = 1
        self.batch_size = 128
        self.buffer_size = int(800)
        self.gamma = 0.99
        self.grad_norm_clip = 10  # prevent gradient explosion
        self.update_target_params = 200  # 200

        # test setting
        self.load_model = False

        if self.cuda:
            self.device = torch.device("cuda: 3" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # model structure
        # drqn net
        self.drqn_hidden_dim = 128
        self.qmix_hidden_dim = 256
        self.two_hyper_layers = True
        self.hyper_hidden_dim = 64
        self.model_dir = './models/'
        self.optimizer = "RMS"
        self.learning_rate = 2e-4

        # epsilon greedy
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        # self.anneal_steps = 50000  # 50000
        # self.anneal_epsilon = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        # self.epsilon_anneal_scale = 'step'

    def set_env_info(self, env_info):
        self.n_actions = env_info["n_actions"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.episode_limit = env_info["episode_limit"]
