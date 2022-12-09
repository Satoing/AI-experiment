import numpy as np
import math


class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim=action_dim #行动的维度
        self.lr=cfg.lr #学习率
        self.gamma=cfg.gamma
        self.epsilon=0
        self.sample_count=0
        self.epsilon_start=cfg.epsilon_start
        self.epsilon_end=cfg.epsilon_end
        self.epsilon_decay=cfg.epsilon_decay
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    # 训练过程中智能体选择动作的方法
    def choose_action(self, state):
        ####################### 智能体的决策函数，需要完成Q表格方法（需要完成）#######################
        self.sample_count += 1
        
        # e-greedy策略
        # epsilon是会递减的，这里选择指数递减
        self.epsilon = self.epsilon_end+(self.epsilon_start-self.epsilon_end)\
            *math.exp(-1.*self.sample_count/self.epsilon_decay)
        
        if np.random.uniform(0,1) > self.epsilon:
            action = np.argmax(self.Q_table[state])  #选择Q(s,a)最大对应的动作
        else:
            action = np.random.choice(self.action_dim)  #随机选择动作
        return action

    # 学习过程中更新Q表格
    # Qlearning传入的参数为s、a、r、s，比sarsa少了下一个状态的动作
    def update(self, state, action, reward, next_state, done):
        ############################## Q表格的更新方法（需要完成）##################################
        Q_predict = self.Q_table[state][action]
        if done: Q_target=reward
        else: Q_target=reward+self.gamma*np.max(self.Q_table[next_state])
        
        self.Q_table[state][action]+=self.lr*(Q_target-Q_predict)
        
    def predict(self,state):
        action=np.argmax(self.Q_table[state])
        return action

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
