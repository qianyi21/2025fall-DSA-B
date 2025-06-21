# 这一部分为游戏实现，请根据你的理解实现终端形式的游戏，使得游戏能够显示当前地图，你的位置，以及你认为有必要的信息
# 此外，该游戏需要支持单人操作、随时退出游戏、游戏存档的保存和加载
# 最后，你还需要补全代码，以便于小北在后续的代码中调用你的游戏

# python 3.12.6

import gym
import numpy as np
import random
import json

# ================音效===============

import pygame
import os

# 获取当前文件所在的目录
dir = os.path.dirname(os.path.abspath(__file__))

# 音效文件路径
MOVE = os.path.join(dir, 'sounds', 'move.wav')
GOAL = os.path.join(dir, 'sounds', 'goal.wav')
FALL = os.path.join(dir, 'sounds', 'fall.wav')
TRES = os.path.join(dir, 'sounds', 'treasure.wav')
DING = os.path.join(dir, 'sounds', 'ding~.wav')
ERROR = os.path.join(dir, 'sounds', 'error.wav')
BGMS = os.path.join(dir, 'sounds', 'backgroundmusic.wav')
QUIT = os.path.join(dir, 'sounds', 'quit.wav')

# 初始化
pygame.mixer.init()

# 加载音效
pygame.mixer.music.load(BGMS)
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(-1)

# 改用拼接后的路径
move_sound = pygame.mixer.Sound(MOVE)
fall_sound = pygame.mixer.Sound(FALL)
goal_sound = pygame.mixer.Sound(GOAL)
treasure_sound = pygame.mixer.Sound(TRES)
ding_sound = pygame.mixer.Sound(DING)
error_sound = pygame.mixer.Sound(ERROR)
background = pygame.mixer.Sound(BGMS)
quit_sound = pygame.mixer.Sound(QUIT)

# ====================================

ACTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)] # 动作集合
NACTIONSPACE = 4
NSTATES = 16
MAPSHAPE = (4, 4)

END_STATE = 'G'
START_STATE = 'S'
ICE_HOLE_STATE = 'H'
SAFE_STATE = 'F'
# 附加要求
GOLD_STATE = 'g'

# 基本函数
# 不要修改这些函数，但是可以调用以获取地图list形式

def state_int2tuple(state:int)->tuple[int, int]:
    '''
    将state从int类型转换为tuple类型

    对应顺序为:
    [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11],
     [12, 13, 14, 15]]
    '''
    return (state // 4, state % 4)


def state_tuple2int(state:tuple[int, int])->int:
    '''
    将state从tuple类型转换为int类型
    '''
    return state[0] * 4 + state[1]



def create_env()->list:
    '''
    创建游戏环境，返回当前游戏的状态
    :return: 返回当前游戏的整张地图的list形式
    例如：
    [['F', 'F', 'F', 'F'],
     ['F', 'S', 'H', 'F'],
     ['H', 'F', 'F', 'F'],
     ['F', 'F', 'F', 'G']]
    其形状一定是4x4矩阵。
    '''
    env = gym.make("FrozenLake-v1")  # 创建环境
    env.reset()
    state = env.render('ansi')
    state = list(filter(lambda x : x == 'F' or x == 'S' or x == 'H' or x == 'G', state))
    state = [state[i * 4: (i + 1) * 4] for i in range(4)]
    # 附加要求（在重置地图时加上宝藏位置）
    map = [(r,c) for r in range(4) for c in range(4) if state[r][c] == SAFE_STATE]
    gold =random.choice(map)
    state[gold[0]][gold[1]] = GOLD_STATE

    return state


def reset(map:list[list])->list[int, list[str]]:
    '''
    根据地图信息，重新初始化游戏，返回当前游戏的位置和新的地图
    其中S表示起始位置，G表示终点位置，H表示冰洞位置，F表示安全位置
    '''
    safe = []

    for i in range(16):
        pos = state_int2tuple(i)
        if map[pos[0]][pos[1]] == 'S':
            map[pos[0]][pos[1]] = 'F'
            safe.append(i)
        elif map[pos[0]][pos[1]] == 'F':
            safe.append(i)
        else:
            continue
    new_s = safe[random.randint(0, len(safe) - 1)]
    new_pos = state_int2tuple(new_s)
    map[new_pos[0]][new_pos[1]] = 'S'
    return [new_s, map]

        

# ========================梦开始的地方========================



def action_mask(map:list[list], state:int)->list[int]:
    '''
    根据所在地图位置，返回合法的动作集合
    '''

    row,col=state_int2tuple(state)
    a=[]

    for i,(row_,col_) in enumerate(ACTIONS):
        new_row,new_col=row + row_,col + col_
        if 0 <= new_row < MAPSHAPE[0] and 0 <= new_col < MAPSHAPE[1]:
            a.append(i)
    return a


def step(map:list[list], state:int, action:int)->tuple[int, int, bool, bool]:
    '''
    根据当前地图位置和动作，返回下一步的地图位置和Reward值和是否结束游戏的Flag
    只有到达终点Reward为1，其它时候Reward都是0
    '''

    row,col=state_int2tuple(state)
    row_,col_=ACTIONS[int(action)]
    new_row,new_col=row + row_,col + col_

    if not (0 <= new_row < MAPSHAPE[0] and 0 <= new_col < MAPSHAPE[1]):
        return state, 0, False, False

    new_state=state_tuple2int((new_row,new_col))
    place=map[new_row][new_col]

    # print(new_state)
    # print(place)

    if place == ICE_HOLE_STATE:
        return new_state, 0, True, False
    elif place == END_STATE:
        return new_state, 1, True, False
    # 附加要求（玩家到达宝藏位置的结果）
    elif place == GOLD_STATE:
        if not map[new_row][new_col] == SAFE_STATE:
            map[new_row][new_col] = SAFE_STATE # 将宝藏位置改为安全位置
            return new_state, 100, False, True # 这里的reward在if __name__ == '__main__':中另外进行判断了~
    else:
        return new_state, 0, False, False


'''
定义你自己的函数，完成所需功能吧~
'''

# 附加要求（加上当前总奖励的进度）
def save_game(state: int, map: list[list], reward: int):
    with open('savefile.json','w') as f:
        json.dump({'state':state,'map':map, 'reward': reward},f)

def load_game():
    try:
        with open('savefile.json','r') as f:
            saved=json.load(f)
            return saved['state'],saved['map'],saved['reward']
    except FileNotFoundError:
        return None

def player_map(map:list[list],state:int):
    row,col=state_int2tuple(state)
    map_=[row.copy() for row in map]
    map_[row][col] = 'P'
    # print(map_)
    for i in map_:
        print(' '.join(i))


if __name__ == '__main__':
    '''
    python game.py 运行你的代码，使其能够在控制台游玩该游戏
    '''

    pygame.mixer.music.load(BGMS)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)

    map=create_env()
    state,map=reset(map)
    # 附加要求
    reward_total = 0  # 当前总奖励
    print("欢迎来到‘小北的未名湖挑战！’")
    print("游戏操作：\nw:上 s:下 a:左 d:右 quit:退出 restart:重新开始 save:存档 load:读档")

    while True:
        print("\n地图:")
        player_map(map,state)
        print(f"小北位置:{state_int2tuple(state)}")
        # 附加要求
        print(f"当前总奖励: {reward_total}")

        a = input("请输入动作：").strip().lower()

        if a == 'quit':
            quit_sound.play()
            while pygame.mixer.get_busy():
                pygame.time.Clock().tick(10)
            print("再见！小北会想你的~")
            break
        elif a == 'restart':
            print("正在重新开始新游戏...")
            map=create_env()
            state,map=reset(map)
            # 附加要求（重置reward）
            reward_total=0
            ding_sound.play()
            print("新一轮游戏开始啦！")
            continue
        elif a == 'save':
            # 附加要求（存档和读档时将之前的reward也一并保存或读取）
            save_game(state,map,reward_total)
            ding_sound.play()
            print("不管有没有boss，存个档再说！")
            continue
        elif a == 'load':
            file=load_game()
            if file:
                state,map,reward_total=file
                ding_sound.play()
                print("读档成功！")
            else:
                error_sound.play()
                print("没有存档！")
            continue

        action={'w':1,'s':0,'a':3,'d':2}
        if a in action:
            move_sound.play()
            action_ = action[a]
            if action_ in action_mask(map,state):
                state,reward,end, gold = step(map,state,action_)
                # 附加要求（每移动一步reward-1）
                reward_total += -1  # 更新总奖励
                # 附加要求（到达宝藏位置reward+100）
                if gold:
                    treasure_sound.play()
                    reward_total=reward_total + 101 # 到宝箱位置不需要-1，所以额外加上上方reward_total中额外减的-1
                    print("天地的馈赠！奖励+100")

                if end:
                    pygame.mixer.music.stop()
                    if reward == 1:
                        goal_sound.play()
                        while pygame.mixer.get_busy():
                            pygame.time.Clock().tick(10)
                        print("\n恭喜你到达终点！\n太厉害啦，简直是未名湖湖神！")
                    else:
                        fall_sound.play()
                        while pygame.mixer.get_busy():
                            pygame.time.Clock().tick(10)
                        print("\n啊哦！掉进洞里了！\n小北：咕噜咕噜咕噜...咳咳")
                    break
            else:
                error_sound.play()
                print("前面的区域，以后再来探索吧！")
        else:
            error_sound.play()
            print("记得认真看游戏操作喔！")
