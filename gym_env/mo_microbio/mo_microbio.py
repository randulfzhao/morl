"""
多目标E.coli演化环境

这个环境模拟了细菌对抗生素的演化过程，作为多目标强化学习问题。
代理根据当前种群状态选择抗生素，目标是针对每种基因型最小化其适应度。

观测空间: 16维向量，表示16种基因型的相对丰度
动作空间: 离散空间(15)，表示可选择的15种抗生素
奖励: 16维向量，对每个基因型都返回-fitness值
"""

import itertools
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle


class MOEvolvingEnv(Env, EzPickle):
    """
    多目标 E. coli 环境，每个基因型都有各自的适应度目标。
    
    观测空间: 16维向量，表示16种基因型的相对丰度
    动作空间: Discrete(15) (选择哪种抗生素)
    奖励空间: 16维向量，对每个基因型都返回 -fitness 值
    """
    def __init__(self, 
                 N: int = 4,                 # 基因型长度(4位二进制 -> 16种可能的基因型)
                 num_drugs: int = 15,        # 可选抗生素数量
                 pop_size: int = 10000,      # 种群总数量
                 gen_per_step: int = 20,     # 每步骤内的演化代数
                 mutation_rate: float = 1e-5, # 突变率
                 hgt_rate: float = 0,        # 水平基因转移率
                 drug_landscapes: list = None): # 抗生素适应度景观
        super().__init__()

        # 环境参数
        self.N = N
        self.num_drugs = num_drugs
        self.pop_size = pop_size
        self.gen_per_step = gen_per_step
        self.mutation_rate = mutation_rate
        self.hgt_rate = hgt_rate

        # 如果没有给定抗生素适应度数据，就使用默认的 Mira et al. (2015) 数据
        if drug_landscapes is None:
            self.drug_landscapes = self.define_mira_landscapes()
        else:
            self.drug_landscapes = drug_landscapes

        # 检测超级基因型（对大多数抗生素都有高抗性的基因型）
        self.super_genotype_indices = self.detect_super_genotypes()
        
        # 所有可能的 16 个基因型
        self.genotypes = [''.join(seq) for seq in itertools.product("01", repeat=self.N)]

        # 定义观测空间: 各基因型在种群中的相对丰度
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(len(self.genotypes),),  # 16维向量
            dtype=np.float32
        )
        
        # 定义动作空间: 15种抗生素选择
        self.action_space = Discrete(self.num_drugs)
        
        # 定义多目标奖励空间: 每个基因型都有自己的奖励
        self.reward_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.genotypes),),  # 16维向量
            dtype=np.float32
        )

        # 初始化环境
        self.reset()

    def define_mira_landscapes(self):
        """
        定义自带的 Mira et al. (2015) 抗生素适应度数据
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

    def detect_super_genotypes(self):
        """
        检测'超级基因型'——对大多数抗生素都有较高抗性的基因型
        
        返回：
            超级基因型的索引列表
        """
        extreme_threshold = 3.0  # 极高抗性阈值
        mean_threshold = 2.0     # 平均抗性阈值

        df = pd.DataFrame(self.drug_landscapes).T
        # 计算每个基因型对多少种药物有极高抗性
        extreme_counts = df.apply(lambda row: (row >= extreme_threshold).sum(), axis=1)
        # 计算每个基因型的平均抗性
        avg_values = df.mean(axis=1)

        # 筛选同时满足两个条件的基因型：至少有一种药物达到极高抗性，且平均抗性高于阈值
        superbugs = df[(extreme_counts > 0) & (avg_values >= mean_threshold)]
        return superbugs.index.tolist()

    def reset(self, seed=None, **kwargs):
        """
        重置环境状态
        
        返回：
            obs: 初始观测
            info: 附加信息
        """
        super().reset(seed=seed)
        
        # 初始化种群，只给第一个基因型全部数量，其它为0
        self.pop = {g: (self.pop_size if i == 0 else 0) for i, g in enumerate(self.genotypes)}

        # 重置环境参数
        self.current_drug_index = 0
        self.steps = 0
        self.done = False

        # 返回初始观测和信息
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        执行一步环境交互
        
        参数：
            action: 选择的抗生素索引
            
        返回：
            observation: 新的观测
            reward: 16维奖励向量，对应每个基因型的-fitness
            done: 是否结束
            truncated: 是否因为步数限制而截断
            info: 附加信息
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset to start a new episode.")

        # 选择当前抗生素
        self.current_drug_index = action
        current_drug = self.drug_landscapes[self.current_drug_index]

        # 按照选择的抗生素执行多代演化
        for _ in range(self.gen_per_step):
            self._mutation_step()   # 突变
            self._hgt_step()        # 水平基因转移
            self._offspring_step(current_drug)  # 繁殖下一代

        # 计算每个基因型的适应度和对应的多目标奖励
        fitnesses = np.array([current_drug[i] for i, g in enumerate(self.genotypes)], dtype=np.float32)
        mo_reward = -fitnesses  # 多目标奖励: 适应度的负值 (shape: 16,)

        # 计算平均适应度（用于info）
        abundance = np.array([self.pop[g] / self.pop_size for g in self.genotypes])
        avg_fitness = float(np.sum(abundance * fitnesses))

        # 更新状态
        self.steps += 1
        # 结束条件：超过最大步数或达到终止条件
        self.done = (self.steps >= 500) or self._check_termination()

        # 附加信息
        info = {
            "super_genotypes_present": self._check_super_genotypes(),
            "avg_fitness": avg_fitness
        }

        return self._get_obs(), mo_reward, self.done, False, info

    def _get_obs(self):
        """
        获取当前观测（16种基因型的相对丰度）
        """
        obs = np.array([self.pop[g] / self.pop_size for g in self.genotypes], dtype=np.float32)
        return obs

    def _check_super_genotypes(self):
        """
        检查超级基因型是否出现在当前种群中
        
        返回：
            bool: 是否有超级基因型
        """
        for idx in self.super_genotype_indices:
            genotype = self.genotypes[idx]
            if self.pop[genotype] > 0:
                return True
        return False

    def _mutation_step(self):
        """
        执行一步随机突变过程
        """
        # 根据泊松分布确定突变数量
        mutation_count = np.random.poisson(self.mutation_rate * self.pop_size * self.N)
        for _ in range(mutation_count):
            self._mutation_event()

    def _mutation_event(self):
        """
        单次突变事件
        """
        haplotype = self._get_random_haplotype()
        # 确保种群数量足够发生突变
        if self.pop[haplotype] > 1:
            self.pop[haplotype] -= 1
            # 生成突变后的基因型
            new_haplotype = self._mutate_haplotype(haplotype)
            # 更新种群计数
            self.pop[new_haplotype] = self.pop.get(new_haplotype, 0) + 1

    def _mutate_haplotype(self, haplotype):
        """
        对单个基因型进行随机突变
        
        参数：
            haplotype: 原基因型
            
        返回：
            突变后的基因型
        """
        # 随机选择一个位点进行突变
        site = np.random.randint(0, self.N)
        # 0变1，1变0
        new_char = "1" if haplotype[site] == "0" else "0"
        return haplotype[:site] + new_char + haplotype[site+1:]

    def _hgt_step(self):
        """
        执行一步水平基因转移过程
        """
        # 根据泊松分布确定水平基因转移数量
        hgt_count = np.random.poisson(self.hgt_rate * self.pop_size * self.N)
        for _ in range(hgt_count):
            self._hgt_event()

    def _hgt_event(self):
        """
        单次水平基因转移事件
        """
        # 随机选择供体和受体
        donor = self._get_random_haplotype()
        receiver = self._get_random_haplotype()
        
        # 确保受体在种群中存在
        if self.pop[receiver] > 0:
            # 基因转移：供体的'1'可以转移给受体（如果受体在相应位置是'0'）
            new_receiver = "".join(
                "1" if (d == "1" and r == "0") else r
                for d, r in zip(donor, receiver)
            )
            self.pop[receiver] -= 1
            self.pop[new_receiver] = self.pop.get(new_receiver, 0) + 1

    def _offspring_step(self, drug):
        """
        根据适应度生成下一代种群
        
        参数：
            drug: 当前选择的抗生素适应度值
        """
        # 计算每个基因型的适应度
        fitnesses = np.array([drug[i] for i, g in enumerate(self.genotypes)])
        
        # 计算基于适应度的选择概率
        total_fitness = np.sum(fitnesses)
        if total_fitness > 0:
            weights = fitnesses / total_fitness
        else:
            # 如果总适应度为0，则均匀分布
            weights = np.ones_like(fitnesses) / len(fitnesses)

        # 根据多项式分布生成下一代
        new_counts = np.random.multinomial(self.pop_size, weights)
        self.pop = {g: new_counts[i] for i, g in enumerate(self.genotypes)}

    def _get_random_haplotype(self):
        """
        根据当前种群分布随机选择一个基因型
        
        返回：
            随机选择的基因型
        """
        haplotypes = list(self.pop.keys())
        probabilities = [self.pop[h] / self.pop_size for h in haplotypes]
        return np.random.choice(haplotypes, p=probabilities)

    def _check_termination(self):
        """
        检查是否达到终止条件（某个基因型占据种群的90%以上）
        
        返回：
            bool: 是否达到终止条件
        """
        return any(self.pop[g] > 0.9 * self.pop_size for g in self.genotypes)

    def render(self):
        """
        打印当前种群状态
        """
        print(f"Step {self.steps}:")
        for genotype, count in self.pop.items():
            if count > 0:
                print(f"  {genotype}: {count}")