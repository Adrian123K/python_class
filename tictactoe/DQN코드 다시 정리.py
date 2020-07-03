import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from tqdm import tqdm

class Environment():
    
    def __init__(self):
    # ����� 0���� �ʱ�ȭ�� 9���� �迭�� �غ�
    # �������� : done = True
        self.board_a = np.zeros(9)
        self.done = False
        self.reward = 0
        self.winner = 0
        self.print = False

    def move(self, p1, p2, player):
    # �� �÷��̾ ������ �ൿ�� ǥ�� �ϰ� ���� ����(���� �Ǵ� ����)�� �Ǵ�
    # p1 = 1, p2 = -1�� ����
    # �� �÷��̾�� �ൿ�� �����ϴ� select_action �޼��带 ����
        if player == 1:
            pos = p1.select_action(env, player)
        else:
            pos = p2.select_action(env, player)
        
        # ���忡 �÷��̾��� ������ ǥ��
        self.board_a[pos] = player
        if self.print:
            print(player)
            self.print_board()
        # ������ ����������� �ƴ����� �Ǵ�
        self.end_check(player)
        
        return  self.reward, self.done
 
    # ���� ���� ���¿��� ������ �ൿ(�� �� �ִ� ���)�� Ž���ϰ� ����Ʈ�� ��ȯ
    def get_action(self):
        observation = []
        for i in range(9):
            if self.board_a[i] == 0:
                observation.append(i)
        return observation
    
    # ������ ����(���� �Ǵ� ���)�ƴ��� �Ǵ�
    def end_check(self,player):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # ���� ������ ����, ����, �밢�� �� -1 �̳� 1 �� ������ �� 
        end_condition = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for line in end_condition:
            if self.board_a[line[0]] == self.board_a[line[1]] \
                and self.board_a[line[1]] == self.board_a[line[2]] \
                and self.board_a[line[0]] != 0:
                # ����ƴٸ� ���� �̰���� ǥ��
                self.done = True
                self.reward = player
                return
        # ��� ���´� ���� ���忡 �� ������ ������
        observation = self.get_action()
        if (len(observation)) == 0:
            self.done = True
            self.reward = 0            
        return
        
    # ���� ������ ���¸� ǥ�� p1 = O, p2 = X    
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
            # ������ �ൿ�� ������ �� ǥ��
            available_action = env.get_action()
            print("possible actions = {}".format(available_action))

            # ���� ��ȣ ǥ��
            print("+----+----+----+")
            print("+  0 +  1 +  2 +")
            print("+----+----+----+")
            print("+  3 +  4 +  5 +")
            print("+----+----+----+")
            print("+  6 +  7 +  8 +")
            print("+----+----+----+")
                        
            # Ű����� ������ �ൿ�� �Է� ����
            action = input("Select action(human) : ")
            action = int(action)
            
            # �Է¹��� �ൿ�� ������ �ൿ�̸� �ݺ����� Ż��
            if action in available_action:
                return action
            # �ƴϸ� �ൿ �Է��� �ݺ�
            else:
                print("You selected wrong action")
        return

class Random_player():
    
    def __init__(self):
        self.name = "Random player"
        self.print = False
        
    def select_action(self, env, player):
        # ������ �ൿ ����
        available_action = env.get_action()
        # ������ �ൿ �� �ϳ��� �������� ����
        action = np.random.randint(len(available_action))
#         print("Select action(random) = {}".format(available_action[action]))
        return available_action[action]       
    
class Q_learning_player():
    
    def __init__(self):
        self.name = "Q_player"
        # Q-table�� ��ųʸ��� ����
        self.qtable = {}
        # e-greedy ��� ����
        self.epsilon = 1
        # �н��� ����
        self.learning_rate = 0.1
        self.gamma=0.9
        self.print = False

    # policy�� ���� ���¿� �´� �ൿ�� ����
    def select_action(self, env, player):
        # policy�� ���� �ൿ�� ����
        action = self.policy(env)
    
        return action 
        
    def policy(self, env):
        # �ൿ ������ ���¸� ����
        available_action = env.get_action()
        # �ൿ ������ ������ Q-value�� ����
        qvalues = np.zeros(len(available_action))

        if self.print:
            print("{} : available_action".format(available_action))

        # �ൿ ������ ������ Q-value�� ����
        for i, act in enumerate(available_action):

            key = (tuple(env.board_a),act)

            # ���� ���¸� ������ ���� ���ٸ�(��ųʸ��� ���ٸ�) ��ųʸ��� �߰�(Q-value = 0)
            if self.qtable.get(key) ==  None:                
                self.qtable[key] = 0
            # �ൿ ������ ������ Q-value ����
            qvalues[i] = self.qtable.get(key)
     


        # e-greedy
        # ������ �ൿ�� �߿��� Q-value�� ���� ū �ൿ�� ����
        greedy_action = np.argmax(qvalues)                    
        
        pr = np.zeros(len(available_action))
      

        # max Q-value�� ���� ���� ������ �ִ��� Ȯ���� �� double_check�� ���¸� ����
        double_check = (np.where(qvalues == np.max(qvalues),1,0))

        #  ������ �ִٸ� �ߺ��� �����߿��� �ٽ� �������� ����    
        if np.sum(double_check) > 1:
         
            double_check = double_check/np.sum(double_check)
            greedy_action =  np.random.choice(range(0,len(double_check)), p=double_check)
            
        # e-greedy�� �ൿ���� ���� Ȯ���� ���
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
        # ���� ���¿� �ൿ�� Ű�� ����
        key = (board_bakup,action_backup)
     
        # Q-table �н�
        if env.done == True:
            # ������ ������ ��� �н�

            self.qtable[key] += self.learning_rate*(reward - self.qtable[key])
            
         
        else:
            # ������ �������� ��� �н�
            # ���� ������ max Q �� ���
            available_action = env.get_action()        
            qvalues = np.zeros(len(available_action))

            for i, act in enumerate(available_action):
                next_key = (tuple(env.board_a),act)
                # ���� ���¸� ������ ���� ���ٸ�(��ųʸ��� ���ٸ�) ��ųʸ��� �߰�(Q-value = 0)
                if self.qtable.get(next_key) ==  None:                
                    self.qtable[next_key] = 0
                qvalues[i] = self.qtable.get(next_key)

            # maxQ ����
            maxQ = np.max(qvalues)  
            
            # ������ �������� �� �н�
            self.qtable[key] += self.learning_rate*(reward + self.gamma * maxQ - self.qtable[key])
        
                
        if self.print:
            print("-----------   learn_qtable end -------------")

class Monte_Carlo_player():
    
    def __init__(self):
        self.name = "MC player"
        self.num_playout = 1000
        
    def select_action(self, env, player):
        # ������ �ൿ ����
        available_action = env.get_action()
        V = np.zeros(len(available_action))
        
        for i in range(len(available_action)):
            # �÷��̾ƿ��� 100�� �ݺ�
            for j in range(self.num_playout):
                # ���� ���¸� �����ؼ� �÷��� �ƿ��� ���
                temp_env = copy.deepcopy(env)
                # �÷��̾ƿ��� ����� �¸� �÷��̾��� ������ ��ȯ
                # p1 �� �̱�� 1, p2 �� �̱�� -1
                self.playout(temp_env, available_action[i], player)
                if player == temp_env.reward:
                    V[i] += 1
   
        return available_action[np.argmax(V)]    

    # �÷��̾ƿ� ����Լ�
    # ������ ������� (�� �Ǵ� �� �Ǵ� ���) �� �ɶ����� �ൿ�� ���Ƿ� �����ϴ� ���� �ݺ�
    # �÷��̾�� ��� �ٲ�� ������ (-)�� ���ؼ� -1, 1, -1 �� �ǰ���    
    def playout(self, temp_env, action, player):
        
        temp_env.board_a[action] = player
        temp_env.end_check(player)
        # ���� ���� üũ
        if temp_env.done == True:
            return 
        else:
            # �÷��̾� ��ü
            player = -player
            # ������ �ൿ ����
            available_action = temp_env.get_action()
            # �������� �ൿ�� ����
            action = np.random.randint(len(available_action))
            self.playout(temp_env, available_action[action], player)

from keras.models import Sequential
from keras.optimizers import SGD
from keras import metrics
from keras.layers import Dense, Flatten, Conv2D
from keras.models import load_model
import time

class DQN_player():
    
    def __init__(self):
        self.name = "DQN_player"
        self.epsilon = 1
        self.learning_rate = 0.1
        self.gamma=0.9
        
        # �ΰ��� �Ű���� ����
        self.main_network = self.make_network()
        self.target_network = self.make_network()
        # ���� �Ű���� ����ġ�� Ÿ�� �Ű���� ����ġ�� ����
        self.copy_network()
        
        self.print = False
        self.print1 = False
        self.count = np.zeros(9)
        self.win = np.zeros(9)
        self.begin = 0
        self.e_trend = []
        
    # �Ű�� ����
    def make_network(self):

        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(3,3,2)))
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='tanh'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dense(9))
        print(self.model.summary())
             
        self.model.compile(optimizer = SGD(lr=0.01), loss = 'mean_squared_error', metrics=['mse'])
        
        return self.model
    
    # �Ű�� ����
    def copy_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
        
    def save_network(self, name):
        filename = name + '_main_network.h5'
        self.main_network.save(filename)
        print("end save model")

        
    # 1���� �迭�� ������¸� 2�������� ��ȯ
    def state_convert(self, board_a):
        d_state = np.full((3,3,2),0.1)
        for i in range(9):
            if board_a[i] == 1:
                d_state[i//3,i%3,0] = 1
            elif board_a[i] == -1:
                d_state[i//3,i%3,1] = 1
            else:
                pass
        return d_state
    
    
    def select_action(self, env, player):
        
        action = self.policy(env)

            
        return action 
        
    def policy(self, env):
        
        if self.print:
            print("-----------   policy start -------------")
        
        # �ൿ ������ ���¸� ����
        available_state = env.get_action()
        
        state_2d = self.state_convert(env.board_a)
        x = np.array([state_2d],dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0,:]
        
        if self.print:
            print("{} : policy state".format(available_state))
            print("{} : qvalues".format(np.round(qvalues,3)))
        
        # �ൿ ������ ������ Q-value�� ����
        available_state_qvalues = qvalues[available_state]

        if self.print:
            print("{} : available_state_qvalues".format(np.round(available_state_qvalues,3)))
        
        # max Q-value�� Ž���� �� ����
        greedy_action = np.argmax(available_state_qvalues)
        if self.print:
            print("{} : self.epsilon".format(self.epsilon))
            print("{} : greedy_action".format(greedy_action))
            print("{} : qvalue = {}".format(available_state_qvalues[greedy_action]))
        
        # max Q-value�� ���� ���� ������ �ִ��� Ȯ���� �� double_check�� ���¸� ����
        double_check = (np.where(qvalues == np.max(available_state[greedy_action]),1,0))
        
        #  ������ �ִٸ� �ߺ��� �����߿��� �ٽ� �������� ����    
        if np.sum(double_check) > 1:
            if self.print:
                print("{} : double_check".format(np.round(double_check,2)))
            double_check = double_check/np.sum(double_check)
            greedy_action =  np.random.choice(range(0,len(double_check)), p=double_check)
            if self.print:
                print("{} : greedy_action".format(greedy_action))
                print("{} : double_check".format(np.round(double_check,2)))
                print("{} : selected state".format(available_state[greedy_action]))
        
        # ��-greedy
        pr = np.zeros(len(available_state))

        for i in range(len(available_state)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon/len(available_state)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))
            else:
                pr[i] = self.epsilon / len(available_state)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))

        action = np.random.choice(range(0,len(available_state)), p=pr)        
        
        if self.print:
            print("{} : pr".format(np.round(pr,2)))
            print("{} : action".format(action))
            print("{} : state[action]".format(available_state[action]))
            print("-----------   policy end -------------")

        if len(available_state) == 9:
            self.count[action] +=1
            self.begin = action
            
        return available_state[action]        
        
    def learn_dqn(self,board_bakup, action_backup, env, reward):
        
        # �Է��� 2�������� ��ȯ�� ��, ���� �Ű������ q-value�� ���
        new_state = self.state_convert(board_bakup)
        x = np.array([new_state],dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0,:]
        before_action_value = copy.deepcopy(qvalues)
        delta = 0
        
        if self.print:
            print("-----------   learn_qtable start -------------")
            print("{} : board_bakup".format(board_bakup))
            print("{} : action_backup".format(action_backup))
            print("{} : reward = {}".format(reward))
            
        if env.done == True:
            if reward == 1:
                self.win[self.begin] += 1
#                 print("winnn")
#                 print("{}".format(self.win))
            if self.print:
                print("{} : delta".format(delta))
                print("{} : before update : actions[action_backup]".format(np.round(qvalues[action_backup],3)))
                print("1  : new_qvalue")
            
            # ������ ��������� �Ű���� �н��� ���� ���� �����͸� ����
            qvalues[action_backup] = reward
            y=np.array([qvalues],dtype=np.float32).astype(np.float32)
            # ������ ���� �����ͷ� ���� �Ű���� �н�
            self.main_network.fit(x, y, epochs=10, verbose=0)
            

        else:
            # ������ �������϶�  �Ű���� �н��� ���� ���� �����͸� ����
            # ���� ���¿��� �ְ� Q ���� ���
            new_state = self.state_convert(env.board_a)
            next_x = np.array([new_state],dtype=np.float32).astype(np.float32)
            next_qvalues = self.target_network.predict(next_x)[0,:]
            available_state = env.get_action()
            maxQ = np.max(next_qvalues[available_state])            
            
            if self.print:
                print("{} : old_qvalue".format(np.round(before_action_value[action_backup],3)))
                print("{} : next_qvalue".format(np.round(next_qvalues,3)))
                print("{} : available_state".format(np.round(available_state,3)))
                print("{} : maxQ".format(np.round(maxQ,3)))
            
            delta = self.learning_rate*(reward + self.gamma * maxQ - qvalues[action_backup])
            
            if self.print:
                print("{} : delta".format(np.round(delta,3)))
                print("{} : before_update_qvalues".format(np.round(qvalues,3)))
                print("{} : before_update_qvalue".format(np.round(qvalues[action_backup],3)))
                
            qvalues[action_backup] += delta
            
            if self.print:
                print("{} : after_update_qvalue".format(np.round(qvalues[action_backup],3)))            
                print("{} : before_update_qvalues".format(np.round(qvalues,3)))
                target_action_value = copy.deepcopy(qvalues)
                print("{} : new_qvalues".format(np.round(qvalues,3)))            
                print("{} : target_action_value id = {}".format(np.round(target_action_value,3),target_action_value))            
            # ������ ���� �����ͷ� ���� �Ű���� �н�
            y=np.array([qvalues],dtype=np.float32).astype(np.float32)
            self.main_network.fit(x, y, epochs = 10, verbose=0)
            
            if self.print:
                after_action_value = copy.deepcopy(self.main_network.predict(x)[0,:])
                delta = after_action_value - before_action_value
                print("{} : before_action_value id = {}".format(np.round(before_action_value,3),id(before_action_value)))
                print("{} : target_action_value id = {}".format(np.round(target_action_value,3),id(target_action_value)))
                print("{} : after_action_value id = {}".format(np.round(after_action_value,3),id(after_action_value)))
                print("{} : delta action value".format(np.round(delta,3)))
                state = ((0,0,0,0,0,0,0,0,0))
                state_2d = self.state_convert(state)
                x = np.array([state_2d],dtype=np.float32).astype(np.float32)
                qvalues = self.main_network.predict(x)[0,:]
                print("{} : initial state qvalues".format(np.round(qvalues,3)))

            
        if self.print:
            print("-----------   learn_qtable end -------------")
            
            
np.random.seed(0)

p1_DQN = DQN_player()

print_opt = False
p1_DQN.print = print_opt
p1_DQN.print1 = print_opt

p1_score = 0
p2_score = 0
draw_score = 0

max_learn = 20000

trend = []

for k in range(3):
    if k == 0:
        p2 = Random_player()
    elif k == 1:
        p2 = Monte_Carlo_player()
        p2.num_playout = 100
    elif k == 2:
        p2 = p2_Qplayer
        p2.epsilon = 0.5
        p2.print = False
        p2.print1 = False
        
    print("p2 player is {}".format(p2.name))

    for j in tqdm(range(max_learn)):
        np.random.seed(j)
        env = Environment()
        
        # ������ �� ���� �Ű���� ����ġ�� Ÿ�� �Ű���� ����ġ�� ����
        p1_DQN.epsilon = 0.7
        p1_DQN.copy_network()

        for i in range(10000):
            # p1 �ൿ�� ����
            player = 1
            pos = p1_DQN.policy(env)

            p1_board_backup = tuple(env.board_a)
            p1_action_backup = pos

            env.board_a[pos] = player
            env.end_check(player)

            # ���� ������
            if env.done == True:
                # p1�� �¸��̹Ƿ� ������ �ൿ�� ���� +1
                # p2�� ������ �ൿ�� ���� -1
                # p1 �ൿ�� ����� �̱�ų� ���ų�
                if env.reward == 0:
                    p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0)
                    draw_score += 1
                    break
                else:
                    p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 1)
                    p1_score += 1
                    break

            # p2 �ൿ�� ����
            player = -1
            pos = p2.select_action(env, player)
            env.board_a[pos] = player
            env.end_check(player)

            if env.done == True:
                # p2�¸� = p1 �й� ������ �ൿ�� ���� -1
                # ���� ���� : 0
                if env.reward == 0:
                    p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0)
                    draw_score += 1
                    break
                else:
                    # ���� ���� : -1
                    p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, -1)
                    p2_score += 1
                    break

            # ������ ������ �ʾҴٸ� p1�� Q-talble �н�
            p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0)

        # 5���Ӹ��� ���� �Ű���� ����ġ�� Ÿ�� �Ű���� ����ġ�� ����
        if j%5 == 0:
            p1_DQN.copy_network()

    print("p1 = {} p2 = {} draw = {}".format(p1_score,p2_score,draw_score))
print("end learn")

p1_DQN.save_network("p1_DQN_0708")            





    