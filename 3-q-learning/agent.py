import numpy as np


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim  # 动作的维度
        self.lr = cfg.lr  # 学习率
        self.gamma = cfg.gamma # 衰减系数
        self.epsilon = 0  # 按一定概率随机选取动作
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        ####################### 智能体的决策函数，需要完成Q表格方法（需要完成）#######################
        self.sample_count += 1
        action = np.random.choice(self.action_dim)  #随机探索选取一个动作
        return action

    def update(self, state, action, reward, next_state, done):
        ############################ Q表格的更新方法（需要完成）##################################
        pass

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
