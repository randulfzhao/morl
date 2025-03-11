"""
多目标E.coli演化环境比较脚本

比较三种策略在抗生素选择问题上的表现：
1. 随机策略 vs DQN
2. 随机策略 vs MORL
3. DQN vs MORL

MORL权重设置为每次只关注一个基因型（该基因型权重为1，其他为0）
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics

import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.envelope.envelope import Envelope


# 1. 环境创建函数
def make_env():
    """为DQN创建标准Gym环境"""
    env = gym.make("microbio-v0")
    env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env)
    return env

def make_mo_env():
    """为MORL创建多目标Gym环境"""
    env = mo_gym.make("mo-microbio-v0")  
    env = FlattenObservation(env)
    env = MORecordEpisodeStatistics(env, gamma=0.98)
    return env


# 2. 定义DQN模型和Agent
class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)


class DQNet(nn.Module):
    """DQN网络模型"""
    def __init__(self, input_shape, num_actions):
        super(DQNet, self).__init__()
        features = input_shape[0]
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, 28)
        self.fc3 = nn.Linear(28, num_actions)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """DQN代理"""
    def __init__(self,
                 input_shape,
                 num_actions,
                 lr=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.005,
                 epsilon_decay=0.995,
                 update_target_every=300,
                 device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_every = update_target_every
        self.device = device

        self.q_net = DQNet(input_shape, num_actions).to(self.device)
        self.target_net = DQNet(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.train_step_count = 0

    def select_action(self, state):
        """选择动作（epsilon-贪心策略）"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_epsilon(self):
        """更新epsilon值"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def train_step(self, batch_size=64):
        """训练一步"""
        if self.replay_buffer.size() < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # 当前Q值
        current_q = self.q_net(states_t).gather(1, actions_t)

        # 下一个状态的最大Q值
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

        # Q值目标
        target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        # 计算损失并优化
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_step_count += 1
        if self.train_step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# 3. 定义药物适应度数据
def define_mira_landscapes():
    """
    定义 Mira et al. (2015) 抗生素适应度数据
    返回一个二维列表 (num_drugs x 16)，表示每种抗生素对每种基因型的适应度值
    值越大表示基因型对抗生素的抗性越强
    """
    drugs = []
    drugs.append([1.851, 2.082, 1.948, 2.434, 2.024, 2.198, 2.033, 0.034, 1.57, 2.165, 0.051, 0.083, 2.186, 2.322, 0.088, 2.821])    #AMP
    drugs.append([1.778, 1.782, 2.042, 1.752, 1.448, 1.544, 1.184, 0.063, 1.72, 2.008, 1.799, 2.005, 1.557, 2.247, 1.768, 2.047])    #AM
    drugs.append([2.258, 1.996, 2.151, 2.648, 2.396, 1.846, 2.23, 0.214, 0.234, 0.172, 2.242, 0.093, 2.15, 0.095, 2.64, 0.516])      #CEC
    drugs.append([0.16, 0.085, 1.936, 2.348, 1.653, 0.138, 2.295, 2.269, 0.185, 0.14, 1.969, 0.203, 0.225, 0.092, 0.119, 2.412])     #CTX
    drugs.append([0.993, 0.805, 2.069, 2.683, 1.698, 2.01, 2.138, 2.688, 1.106, 1.171, 1.894, 0.681, 1.116, 1.105, 1.103, 2.591])    #ZOX
    drugs.append([1.748, 1.7, 2.07, 1.938, 2.94, 2.173, 2.918, 3.272, 0.423, 1.578, 1.911, 2.754, 2.024, 1.678, 1.591, 2.923])       #CXM
    drugs.append([1.092, 0.287, 2.554, 3.042, 2.88, 0.656, 2.732, 0.436, 0.83, 0.54, 3.173, 1.153, 1.407, 0.751, 2.74, 3.227])       #CRO
    drugs.append([1.435, 1.573, 1.061, 1.457, 1.672, 1.625, 0.073, 0.068, 1.417, 1.351, 1.538, 1.59, 1.377, 1.914, 1.307, 1.728])    #AMC
    drugs.append([2.134, 2.656, 2.618, 2.688, 2.042, 2.756, 2.924, 0.251, 0.288, 0.576, 1.604, 1.378, 2.63, 2.677, 2.893, 2.563])    #CAZ
    drugs.append([2.125, 1.922, 2.804, 0.588, 3.291, 2.888, 3.082, 3.508, 3.238, 2.966, 2.883, 0.89, 0.546, 3.181, 3.193, 2.543])    #CTT
    drugs.append([1.879, 2.533, 0.133, 0.094, 2.456, 2.437, 0.083, 0.094, 2.198, 2.57, 2.308, 2.886, 2.504, 3.002, 2.528, 3.453])    #SAM
    drugs.append([1.743, 1.662, 1.763, 1.785, 2.018, 2.05, 2.042, 0.218, 1.553, 0.256, 0.165, 0.221, 0.223, 0.239, 1.811, 0.288])    #CPR
    drugs.append([0.595, 0.245, 2.604, 3.043, 1.761, 1.471, 2.91, 3.096, 0.432, 0.388, 2.651, 1.103, 0.638, 0.986, 0.963, 3.268])    #CPD
    drugs.append([2.679, 2.906, 2.427, 0.141, 3.038, 3.309, 2.528, 0.143, 2.709, 2.5, 0.172, 0.093, 2.453, 2.739, 0.609, 0.171])     #TZP
    drugs.append([2.59, 2.572, 2.393, 2.832, 2.44, 2.808, 2.652, 0.611, 2.067, 2.446, 2.957, 2.633, 2.735, 2.863, 2.796, 3.203])     #FEP
    return drugs


# 4. 训练DQN Agent函数
def train_dqn_agent(episodes=500, learning_rate=0.001, gamma=0.99, 
                   epsilon_min=0.005, batch_size=64, update_target_every=310):
    """训练DQN代理"""
    print("开始训练DQN Agent...")
    
    env = make_env()
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DQNAgent(
        input_shape=obs_shape,
        num_actions=num_actions,
        lr=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=epsilon_min,
        epsilon_decay=0.999,
        update_target_every=update_target_every,
        device="cpu"
    )

    for episode in tqdm(range(episodes)):
        obs, info = env.reset()
        obs = obs.flatten()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            next_obs = next_obs.flatten() 
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.train_step(batch_size)
            
            obs = next_obs
            episode_reward += reward
            
            if done or truncated:
                break

        agent.update_epsilon()
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")

    print("DQN Agent训练完成！")
    return agent


# 5. 训练MORL Agent函数
def train_morl_agent(total_timesteps=1000000):
    """训练MORL代理"""
    print("开始训练MORL Agent...")
    
    env = make_mo_env()
    eval_env = make_mo_env()

    # 设置算法参数
    learning_rate = 1e-4
    initial_epsilon = 0.2
    final_epsilon = 0.01
    epsilon_decay_steps = 50000
    tau = 1.0
    target_net_update_freq = 200
    buffer_size = int(2e6)
    net_arch = [256, 256, 256, 256, 256, 256]
    batch_size = 64
    learning_starts = 1000
    gradient_updates = 1
    gamma = 0.99
    envelope = True
    num_sample_w = 4
    per = False
    per_alpha = 0.6
    initial_homotopy_lambda = 0.0
    final_homotopy_lambda = 1.0
    homotopy_decay_steps = 10000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = Envelope(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        epsilon_decay_steps=epsilon_decay_steps,
        tau=tau,
        target_net_update_freq=target_net_update_freq,
        buffer_size=buffer_size,
        net_arch=net_arch,
        batch_size=batch_size,
        learning_starts=learning_starts,
        gradient_updates=gradient_updates,
        gamma=gamma,
        envelope=envelope,
        num_sample_w=num_sample_w,
        per=per,
        per_alpha=per_alpha,
        initial_homotopy_lambda=initial_homotopy_lambda,
        final_homotopy_lambda=final_homotopy_lambda,
        homotopy_decay_steps=homotopy_decay_steps,
        device=device,
        log=False,
        seed=42
    )

    # 评估参数
    eval_freq = 500
    ref_point = np.zeros(16)  # 16个基因型的参考点
    known_pareto_front = None

    # 开始训练
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        known_pareto_front=known_pareto_front,
        eval_freq=eval_freq,
        num_eval_weights_for_front=10,
        num_eval_episodes_for_front=3,
        num_eval_weights_for_eval=10,
        verbose=True
    )

    print("MORL Agent训练完成！")
    return agent


# 6. 测试与比较函数

def compare_random_vs_dqn(agent_dqn, eval_env, drugs, num_episodes=100):
    """比较随机策略和DQN策略的表现差异"""
    print("比较随机策略 vs DQN...")
    
    fitness_diff_genotypes = [[] for _ in range(16)]
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = eval_env.reset()
        random_env = deepcopy(eval_env)
        done = False

        while not done:
            # 选择动作
            action_dqn = agent_dqn.select_action(obs)
            action_rand = random.randint(0, 14)
            
            # 执行动作
            next_obs, reward, done, truncated, info = eval_env.step(action_dqn)
            next_obs_rand, _, _, _, _ = random_env.step(action_rand)

            # 计算各基因型的适应度
            fitness_dqn = drugs[action_dqn]
            fitness_rand = drugs[action_rand]
            
            # 记录差异（随机-DQN）
            fitness_diff = np.array(fitness_rand) - np.array(fitness_dqn)
            
            # 更新状态
            obs = next_obs
            done = done or truncated
            
        # 保存每个基因型的差异
        for i in range(16):
            fitness_diff_genotypes[i].append(fitness_diff[i])
    
    return fitness_diff_genotypes


def compare_random_vs_morl(agent_morl, eval_env, drugs, num_episodes=100):
    """比较随机策略和MORL策略的表现差异"""
    print("比较随机策略 vs MORL...")
    
    fitness_diff_genotypes = [[] for _ in range(16)]
    
    # 逐个基因型进行比较
    for genotype_idx in range(16):
        # 设置权重向量，目标基因型为1，其他为0
        weights = np.zeros(16)
        weights[genotype_idx] = 1
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            random_env = deepcopy(eval_env)
            done = False
            
            while not done:
                # 选择动作
                action_morl = agent_morl.eval(obs, weights)
                action_rand = random.randint(0, 14)
                
                # 执行动作
                next_obs, reward, done, truncated, info = eval_env.step(action_morl)
                next_obs_rand, _, _, _, _ = random_env.step(action_rand)
                
                # 计算该基因型的适应度
                fitness_morl = drugs[action_morl][genotype_idx]
                fitness_rand = drugs[action_rand][genotype_idx]
                
                # 更新状态
                obs = next_obs
                done = done or truncated
            
            # 记录差异（随机-MORL）
            fitness_diff_genotypes[genotype_idx].append(fitness_rand - fitness_morl)
    
    return fitness_diff_genotypes


def compare_dqn_vs_morl(agent_dqn, agent_morl, eval_env, drugs, num_episodes=100):
    """比较DQN策略和MORL策略的表现差异"""
    print("比较DQN vs MORL...")
    
    fitness_diff_genotypes = [[] for _ in range(16)]
    
    # 逐个基因型进行比较
    for genotype_idx in range(16):
        # 设置权重向量，目标基因型为1，其他为0
        weights = np.zeros(16)
        weights[genotype_idx] = 1
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            dqn_env = deepcopy(eval_env)
            done = False
            
            while not done:
                # 选择动作
                action_morl = agent_morl.eval(obs, weights)
                action_dqn = agent_dqn.select_action(obs)
                
                # 执行动作
                next_obs, reward, done, truncated, info = eval_env.step(action_morl)
                next_obs_dqn, _, _, _, _ = dqn_env.step(action_dqn)
                
                # 计算该基因型的适应度
                fitness_morl = drugs[action_morl][genotype_idx]
                fitness_dqn = drugs[action_dqn][genotype_idx]
                
                # 更新状态
                obs = next_obs
                done = done or truncated
            
            # 记录差异（DQN-MORL）
            fitness_diff_genotypes[genotype_idx].append(fitness_dqn - fitness_morl)
    
    return fitness_diff_genotypes


# 7. 可视化比较结果
def plot_comparison(fitness_diff_genotypes, title):
    """绘制比较结果"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for i, ax in enumerate(axes.flatten()):
        ax.scatter(range(len(fitness_diff_genotypes[i])), fitness_diff_genotypes[i], alpha=0.7)
        ax.set_title(f"Genotype {i + 1}\n{title}", fontsize=10, pad=10)
        ax.set_xlabel("Index", fontsize=8)
        ax.set_ylabel("Fitness Diff", fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout(pad=3.0)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


def collect_fitness_data(agent_dqn, agent_morl, eval_env, drugs, num_episodes=100):
    """收集三种策略(随机、DQN、MORL)的原始适应度数据"""
    print("收集三种策略的适应度数据...")
    
    # 创建数据结构存储三种策略的适应度结果
    random_fitness = [[] for _ in range(16)]
    dqn_fitness = [[] for _ in range(16)]
    morl_fitness = [[] for _ in range(16)]
    
    # 逐个基因型进行比较
    for genotype_idx in range(16):
        # 设置MORL权重向量，目标基因型为1，其他为0
        weights = np.zeros(16)
        weights[genotype_idx] = 1
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            dqn_env = deepcopy(eval_env)
            random_env = deepcopy(eval_env)
            done = False
            
            while not done:
                # 选择动作
                action_morl = agent_morl.eval(obs, weights)
                action_dqn = agent_dqn.select_action(obs)
                action_rand = random.randint(0, 14)
                
                # 执行动作
                next_obs, reward, done, truncated, info = eval_env.step(action_morl)
                next_obs_dqn, _, dqn_done, _, _ = dqn_env.step(action_dqn)
                next_obs_rand, _, rand_done, _, _ = random_env.step(action_rand)
                
                # 记录各策略对当前基因型的适应度值
                morl_fitness[genotype_idx].append(drugs[action_morl][genotype_idx])
                dqn_fitness[genotype_idx].append(drugs[action_dqn][genotype_idx])
                random_fitness[genotype_idx].append(drugs[action_rand][genotype_idx])
                
                # 更新状态
                obs = next_obs
                done = done or truncated
    
    return random_fitness, dqn_fitness, morl_fitness

def plot_violin_comparison(random_fitness, dqn_fitness, morl_fitness):
    """使用小提琴图绘制三种策略的原始适应度结果"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    # 设置小提琴图的颜色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 蓝、橙、绿
    labels = ["Random", "DQN", "MORL"]
    
    for i, ax in enumerate(axes.flatten()):
        # 准备数据
        data_to_plot = [
            random_fitness[i],
            dqn_fitness[i],
            morl_fitness[i]
        ]
        
        # 绘制小提琴图
        parts = ax.violinplot(data_to_plot, showmeans=False, showmedians=True)
        
        # 设置颜色
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # 添加标题和标签
        ax.set_title(f"Genotype {i + 1}", fontsize=24, pad=10)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, rotation=45, fontsize=20)
        ax.tick_params(axis='y', labelsize=8)
        
        # 计算并标注中位数
        medians = [np.median(data) for data in data_to_plot]
        for j, median in enumerate(medians):
            ax.text(j+1, median + 0.1, f"{median:.2f}", 
                   horizontalalignment='center', size='x-small', 
                   color='black', weight='semibold')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置y轴标签
        if i % 4 == 0:  # 只在每行第一个子图上添加y轴标签
            ax.set_ylabel("Fitness Value", fontsize=10)

    # 添加整体标题
    plt.suptitle("Comparison of Fitness Values Across Different Strategies", fontsize=16)
    
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])  # rect参数为整体标题留出空间
    plt.savefig("strategy_fitness_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# 8. 主函数
def main():
    # 定义药物适应度数据
    drugs = define_mira_landscapes()
    
    # 创建评估环境
    eval_env = make_env()
    mo_eval_env = make_mo_env()
    
    # 训练DQN代理
    dqn_agent = train_dqn_agent(episodes=500)
    
    # 训练MORL代理
    morl_agent = train_morl_agent(total_timesteps=200000)
    
    # 比较随机策略和DQN
    random_fitness, dqn_fitness, morl_fitness = collect_fitness_data(
        dqn_agent, morl_agent, mo_eval_env, drugs, num_episodes=100
    )
    plot_violin_comparison(random_fitness, dqn_fitness, morl_fitness)
    print("随机策略 vs DQN 比较完成！")


if __name__ == "__main__":
    main()