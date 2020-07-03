import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from tqdm import tqdm

class Environment():
    
    def __init__(self):
    # 보드는 0으로 초기화된 9개의 배열로 준비
    # 게임종료 : done = True
        self.board_a = np.zeros(9)
        self.done = False
        self.reward = 0
        self.winner = 0
        self.print = False

    def move(self, p1, p2, player):
    # 각 플레이어가 선택한 행동을 표시 하고 게임 상태(진행 또는 종료)를 판단
    # p1 = 1, p2 = -1로 정의
    # 각 플레이어는 행동을 선택하는 select_action 메서드를 가짐
        if player == 1:
            pos = p1.select_action(env, player)
        else:
            pos = p2.select_action(env, player)
        
        # 보드에 플레이어의 선택을 표시
        self.board_a[pos] = player
        if self.print:
            print(player)
            self.print_board()
        # 게임이 종료상태인지 아닌지를 판단
        self.end_check(player)
        
        return  self.reward, self.done
 
    # 현재 보드 상태에서 가능한 행동(둘 수 있는 장소)을 탐색하고 리스트로 반환
    def get_action(self):
        observation = []
        for i in range(9):
            if self.board_a[i] == 0:
                observation.append(i)
        return observation
    
    # 게임이 종료(승패 또는 비김)됐는지 판단
    def end_check(self,player):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # 승패 조건은 가로, 세로, 대각선 이 -1 이나 1 로 동일할 때 
        end_condition = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for line in end_condition:
            if self.board_a[line[0]] == self.board_a[line[1]] \
                and self.board_a[line[1]] == self.board_a[line[2]] \
                and self.board_a[line[0]] != 0:
                # 종료됐다면 누가 이겼는지 표시
                self.done = True
                self.reward = player
                return
        # 비긴 상태는 더는 보드에 빈 공간이 없을때
        observation = self.get_action()
        if (len(observation)) == 0:
            self.done = True
            self.reward = 0            
        return
        
    # 현재 보드의 상태를 표시 p1 = O, p2 = X    
    def print_board(self):
        print("+----+----+----+")
        for i in range(3):
            for j in range(3):
                if self.board_a[3*i+j] == 1:
                    print("|  O",end=" ")
                elif self.board_a[3*i+j] == -1:
                    print("|  X",end=" ")
                else:
                    print("|   ",end=" ")
            print("|")
            print("+----+----+----+")
            

class Human_player():
    
    def __init__(self):
        self.name = "Human player"
        
    def select_action(self, env, player):
        while True:
            # 가능한 행동을 조사한 후 표시
            available_action = env.get_action()
            print("possible actions = {}".format(available_action))

            # 상태 번호 표시
            print("+----+----+----+")
            print("+  0 +  1 +  2 +")
            print("+----+----+----+")
            print("+  3 +  4 +  5 +")
            print("+----+----+----+")
            print("+  6 +  7 +  8 +")
            print("+----+----+----+")
                        
            # 키보드로 가능한 행동을 입력 받음
            action = input("Select action(human) : ")
            action = int(action)
            
            # 입력받은 행동이 가능한 행동이면 반복문을 탈출
            if action in available_action:
                return action
            # 아니면 행동 입력을 반복
            else:
                print("You selected wrong action")
        return

       
    
class Q_learning_player():
    
    def __init__(self):
        self.name = "Q_player"
        # Q-table을 딕셔너리로 정의
        self.qtable = {}
        # e-greedy 계수 정의
        self.epsilon = 1
        # 학습률 정의
        self.learning_rate = 0.1
        self.gamma=0.9
        self.print = False

    # policy에 따라 상태에 맞는 행동을 선택
    def select_action(self, env, player):
        # policy에 따라 행동을 선택
        action = self.policy(env)
    
        return action 
        
    def policy(self, env):
        # 행동 가능한 상태를 저장
        available_action = env.get_action()
        # 행동 가능한 상태의 Q-value를 저장
        qvalues = np.zeros(len(available_action))

        if self.print:
            print("{} : available_action".format(available_action))

        # 행동 가능한 상태의 Q-value를 조사
        for i, act in enumerate(available_action):

            key = (tuple(env.board_a),act)

            # 현재 상태를 경험한 적이 없다면(딕셔너리에 없다면) 딕셔너리에 추가(Q-value = 0)
            if self.qtable.get(key) ==  None:                
                self.qtable[key] = 0
            # 행동 가능한 상태의 Q-value 저장
            qvalues[i] = self.qtable.get(key)
     


        # e-greedy
        # 가능한 행동들 중에서 Q-value가 가장 큰 행동을 저장
        greedy_action = np.argmax(qvalues)                    
        
        pr = np.zeros(len(available_action))
      

        # max Q-value와 같은 값이 여러개 있는지 확인한 후 double_check에 상태를 저장
        double_check = (np.where(qvalues == np.max(qvalues),1,0))

        #  여러개 있다면 중복된 상태중에서 다시 무작위로 선택    
        if np.sum(double_check) > 1:
         
            double_check = double_check/np.sum(double_check)
            greedy_action =  np.random.choice(range(0,len(double_check)), p=double_check)
            
        # e-greedy로 행동들의 선택 확률을 계산
        pr = np.zeros(len(available_action))

        for i in range(len(available_action)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon/len(available_action)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))
            else:
                pr[i] = self.epsilon / len(available_action)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))

        action = np.random.choice(range(0,len(available_action)), p=pr)

      
        return available_action[action]    
    
    def learn_qtable(self,board_bakup, action_backup, env, reward):
        # 현재 상태와 행동을 키로 저장
        key = (board_bakup,action_backup)
     
        # Q-table 학습
        if env.done == True:
            # 게임이 끝났을 경우 학습

            self.qtable[key] += self.learning_rate*(reward - self.qtable[key])
            
         
        else:
            # 게임이 진행중일 경우 학습
            # 다음 상태의 max Q 값 계산
            available_action = env.get_action()        
            qvalues = np.zeros(len(available_action))

            for i, act in enumerate(available_action):
                next_key = (tuple(env.board_a),act)
                # 다음 상태를 경험한 적이 없다면(딕셔너리에 없다면) 딕셔너리에 추가(Q-value = 0)
                if self.qtable.get(next_key) ==  None:                
                    self.qtable[next_key] = 0
                qvalues[i] = self.qtable.get(next_key)

            # maxQ 조사
            maxQ = np.max(qvalues)  
            
            # 게임이 진행중일 때 학습
            self.qtable[key] += self.learning_rate*(reward + self.gamma * maxQ - self.qtable[key])
        
                
        if self.print:
            print("-----------   learn_qtable end -------------")


# 큐함수 훈련시키는 코드 


p1_Qplayer = Q_learning_player()
p2_Qplayer = Q_learning_player()

# 입실론은 0.5로 설정
p1_Qplayer.epsilon = 0.5
p2_Qplayer.epsilon = 0.5

p1_score = 0
p2_score = 0
draw_score = 0

# printer = True
print()
max_learn = 100000

for j in tqdm(range(max_learn)):
    np.random.seed(j)
    env = Environment()
    
    for i in range(10000):
        
        # p1 행동 선택
        player = 1
        pos = p1_Qplayer.policy(env)
        # 현재 상태 s, 행동 a를 저장
        p1_board_backup = tuple(env.board_a)
        p1_action_backup = pos
        env.board_a[pos] = player
        env.end_check(player)
        
        # 게임이 종료상태라면 각 플레이어의 Q-table을 학습
        if env.done == True:
            # 비겼으면 보수 0
            if env.reward == 0:
                p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, 0)
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, 0)
                draw_score += 1
                break
            # p1이 이겼으므로 보상 +1
            # p2이 졌으므로 보상 -1
            else:
                p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, 1)
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, -1)
                p1_score += 1
                break
            
        # 게임이 끌나지 않았다면 p2의 Q-talble을 학습 (게임 시작직후에는 p2 는 학습할수 없음)
        if i != 0:
            p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, -0.01)
#         print("p1_board_backup = {} p1_action_backup = {}".format(p1_board_backup, p1_action_backup))
        
        # p2 행동 선택
        player = -1
        pos = p2_Qplayer.policy(env)
        p2_board_backup = tuple(env.board_a)
        p2_action_backup = pos
        env.board_a[pos] = player
        env.end_check(player)
    
#         print("p1_board_backup = {} p1_action_backup = {}".format(p1_board_backup, p1_action_backup))
        
        if env.done == True:
            # 비겼으면 보수 0
            if env.reward == 0:
                p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, 0)
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, 0)
                draw_score += 1
                break
            # p2이 이겼으므로 보상 +1
            # p1이 졌으므로 보상 -1
            else:
                p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, -1)
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, 1)
                p2_score += 1
                break
    
        # 게임이 끌나지 않았다면 p1의  Q-talble 학습
        p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, -0.01)    
    
print("p1 = {} p2 = {} draw = {}".format(p1_score,p2_score,draw_score))
print("end train")


# 게임 진횅 코드 

np.random.seed(0)

p1 = Human_player()
# p2 = Human_player()

# p1 = Random_player()
# p2 = Random_player()

# p1 = Monte_Carlo_player()
# p1.num_playout = 100
#p2 = Q_learning_player()
#p2.num_playout = 1000

#p1 = p1_Qplayer
#p1.epsilon = 0

p2 = p2_Qplayer
p2.epsilon = 0

# p1 = p1_DQN
# p1.epsilon = 0

# 지정된 게임 수를 자동으로 두게 할 것인지 한게임씩 두게 할 것인지 결정
# auto = True : 지정된 판수(games)를 자동으로 진행 
# auto = False : 한판씩 진행

auto = False

# auto 모드의 게임수
games = 100

print("pl player : {}".format(p1.name))
print("p2 player : {}".format(p2.name))

# 각 플레이어의 승리 횟수를 저장
p1_score = 0
p2_score = 0
draw_score = 0


if auto: 
    # 자동 모드 실행
    for j in tqdm(range(games)):
        
        np.random.seed(j)
        env = Environment()
        
        for i in range(10000):
            # p1 과 p2가 번갈아 가면서 게임을 진행
            # p1(1) -> p2(-1) -> p1(1) -> p2(-1) ...
            reward, done = env.move(p1,p2,(-1)**i)
            # 게임 종료 체크
            if done == True:
                if reward == 1:
                    p1_score += 1
                elif reward == -1:
                    p2_score += 1
                else:
                    draw_score += 1
                break

else:                
    # 한 게임씩 진행하는 수동 모드
    np.random.seed(1)
    while True:
        
        env = Environment()
        env.print = False
        for i in range(10000):
            reward, done = env.move(p1,p2,(-1)**i)
            env.print_board()
            if done == True:
                if reward == 1:
                    print("winner is p1({})".format(p1.name))
                    p1_score += 1
                elif reward == -1:
                    print("winner is p2({})".format(p2.name))
                    p2_score += 1
                else:
                    print("draw")
                    draw_score += 1
                break
        
        # 최종 결과 출력        
        print("final result")
        env.print_board()

        # 한게임 더?최종 결과 출력 
        answer = input("More Game? (y/n)")

        if answer == 'n':
            break           

print("p1({}) = {} p2({}) = {} draw = {}".format(p1.name, p1_score,p2.name, p2_score,draw_score))

















    