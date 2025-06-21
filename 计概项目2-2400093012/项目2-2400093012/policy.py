from tensorboardX import SummaryWriter
from tqdm import tqdm
import gym
import numpy as np
from random import random
from game import action_mask, NSTATES, NACTIONSPACE, reset, step, create_env
import pygame
# ======================= 你需要补全的函数 ===============================

def sample_actions(map:list[list], state:int)->int:
    '''
    根据所在位置等概率采样所有可能动作(需要补全)
    '''
    a = action_mask(map,state)
    # print(f"random_a:{np.random.choice(a)}")
    return np.random.choice(a)

def compute_qpi_MC(pi:list[int], map:list[list], gamma:float, epsilon:float, num_episodes=1000):
    """
    使用蒙特卡洛方法来估计动作价值函数Q_pi。
    参数：
    pi -- 在环境中使用的确定性策略，是一个大小为状态数的numpy数组，输入状态，输出动作。
    map -- 冰湖环境list。
    gamma -- 折扣因子，一个0到1之间的浮点数。
    epsilon -- 选择动作的概率。
    num_episodes -- 进行采样的回合数。

    返回值：
    Q -- 动作价值函数Q_pi的估计，是一个字典，键是状态-动作对，值是该状态-动作对的估计值。
    """
    Q = np.zeros((NSTATES, NACTIONSPACE), dtype=np.float32)
    N = np.zeros((NSTATES, NACTIONSPACE), dtype=np.int64)
    for i in range(num_episodes):
        # 生成新的回合
        state, map = reset(map)
        episode = []

        # 对于该回合中的每个时间步
        while True:
            # 根据策略选择动作,采用epsilon-greedy方式采样(需要补全)
            # ====================================================
            action = 0

            if np.random.rand() < epsilon:
                action = sample_actions(map,state)
            else:
                action = pi[state]
            # print(f"action:{action}")
            # ===================================================

            # 执行动作，获得新状态和回报值 #附加要求（gold=是否找到宝藏的flag）
            next_state, reward, done, gold= step(map, state, action)

            # 记录状态、动作、回报值 # 、宝藏flag
            episode.append((state, action, reward, gold))
            # 如果回合结束，记录最后一个状态、动作和奖励 # +宝藏flag ，退出
            if done == True:
                episode.append((next_state, None, reward, gold))
                break
            # 转换到下一个状态
            state = next_state

        # 对于该回合中的每个状态-动作对
        visited = set() # 这个状态是否经过
        G = 0 # 累计回报


        # 计算回报值的累积,并更新动作价值函数 # 还有宝藏的flag
        for _, (state, action, reward, gold) in enumerate(reversed(episode)):
            # G = gamma * G + reward
            # 需要补全G的更新方式
            # ==================================
            G = gamma * G + reward
            # print(f"G:{G}")
            # =================================

            if action == None:
                continue
            sa = (state, action)
            # 如果该状态-动作对没有被访问过，更新N和Q
            if sa not in visited:
                visited.add(sa)
                state = int(state)
                action = int(action)
                # =========== 补全Q的更新方式，并且添加访问 =================
                N[state][action] += 1
                Q[state][action] = Q[state][action] + ((G - Q[state][action]) / N[state][action])
                # print(f"N:{N},Q:{Q}")
                # ===============================================

    return Q

# ======================= 尽力不要修改的函数 ===============================

def policy_iteration_MC(map:list[list], gamma:float, eps0:float=0.5, decay:float=0.1, num_episodes:int=100000,\
                         diff_p:bool=False, decay_f=None, writer=None, name='LGD!!')->np.array:
    """
    使用蒙特卡洛方法来实现策略迭代。
    参数：
    env -- OpenAI Gym环境对象。
    gamma -- 折扣因子，一个0到1之间的浮点数。
    eps0 -- 初始的探索概率。
    decay – 衰减速率。
    num_episodes -- 进行采样的回合数。

    返回值：
    Q -- 动作价值函数Q_pi的估计，是一个字典，键是状态-动作对，值是该状态-动作对的估计值。
    """

    pi = np.zeros(NSTATES)
    iteration = 1
    while True:

        if diff_p:
            epsilon = decay_f(eps0, decay, iteration)
        else:
            epsilon = eps0/(1+decay*iteration)

        Q = compute_qpi_MC(pi, map, gamma, epsilon, num_episodes)
        new_pi = Q.argmax(axis=1)
        if (pi != new_pi).sum() == 0:# 策略不再改变，作为收敛判定条件
                print(iteration)
                result = test_pi(map, new_pi)
                writer.add_scalar(f'/MC3/{name}', result, global_step=iteration)
                return new_pi

        pi = new_pi
        iteration = iteration + 1
        if iteration % 1 == 0:
            result = test_pi(map, new_pi)
            writer.add_scalar(f'/MC3/{name}', result, global_step=iteration)
        

def test_pi(map:list[list], pi, num_episodes=1000):
    """
    测试策略。
    参数：
        env -- OpenAI Gym环境对象。
        pi -- 需要测试的策略。
        num_episodes -- 进行测试的回合数。

    返回值：
        成功到达终点的频率。
    """

    count = 0

    for e in range(num_episodes):
        ob, map = reset(map)

        while True:
            a = pi[ob]
            # 附加要求（gold=是否找到宝藏的flag）
            ob, rew, done, gold = step(map, ob, a)

            if done:
                count += 1 if rew == 1 else 0
                break

    return count / num_episodes


# ======================= 可以修改以用于调试 ===============================

if __name__ == '__main__':
    pygame.mixer.music.stop()

    map = create_env() # 创建地图

    # 为模型训练添加tensorboard日志，训练可视化
    writer = SummaryWriter('./result')

    # 开始策略迭代
    pi = policy_iteration_MC(map, 0.95, decay=0.1, writer=writer, name=f'basic')

    # 输出最终评测结果
    print(f'Successful Rate: {test_pi(map, pi)}')

    print('finish')
