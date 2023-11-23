# import pygame.display
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from gensim.models.word2vec import Word2Vec
from NMT import Translate, kor_model, eng_model, eng_list
import csv
import pandas as pd

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 1e-3

# TEST SENTENCES
# '그들은 내가 잘하는 것을 바탕으로 별명을 사용하고 있기 때문에 나는 사람들이 치타라고 불러주면 기분이 좋아.'
# '나는 친구에게 그 철학자의 책을 선물해 주겠다고 말했습니다.'
# ENGLISH
# 'I feel happy when people call me cheetah because they are using a nickname based on something that I am good at.'
# 'I told my friends that I will give you the philosopher's book as a gift.'

TEST_DATA = [['그들은 내가 잘하는 것을 바탕으로 별명을 사용하고 있기 때문에 나는 사람들이 치타라고 불러주면 기분이 좋아.',
              'I feel happy when people call me cheetah because they are using a nickname based on something that I am good at.'],
             ['나는 친구에게 그 철학자의 책을 선물해 주겠다고 말했습니다.',
              'I told my friends that I will give you the philosopher\'s book as a gift.']]

################################################
# 큰 데이터 사용
# f = open("data/data.csv",'r',encoding='utf-8')
# reader = csv.reader(f)
# inputList = []
# for line in reader:
#     inputList.append(line)
# f.close()
#
# dataFrame = pd.DataFrame(inputList[1:],columns=inputList[0])
# DATA = []
#
# for i in range(len(dataFrame['원문'])):
#     DATA.append([dataFrame['원문'][i], dataFrame['번역문'][i]])
#####################################################
# 작은 데이터 사용
f = open("data/low_eng.csv",'r',encoding='utf-8-sig')
reader = csv.reader(f)
inputList = []
for line in reader:
    inputList.append(line)
f.close()

dataFrame = pd.DataFrame(inputList[1:],columns=inputList[0])
DATA = []

for i in range(len(dataFrame['원문'])):
    DATA.append([dataFrame['원문'][i], dataFrame['번역문'][i]])


# TODO:
# tokenize
# 단어 임베딩
# 문장을 임베딩된 벡터의 배열로 바꿈

# kor_model = Word2Vec.load('model/korsentences.model')
# eng_model = Word2Vec.load('model/engsentences.model')
# kor_wv = kor_model.wv
# eng_wv = eng_model.wv
# kor_list = kor_wv.index_to_key
# eng_list = eng_wv.index_to_key

# pretrained = torch.load("model/low_model.pth")

class Agent:

    def __init__(self,model=Linear_QNet(400,256,len(eng_list)),ngames=0,record=0,epsilon=80):
        self.record = record
        self.n_games = ngames
        self.epsilon = epsilon # randomness
        self.gamma = 1 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, translate):
        sentence = translate.now_translation[0]
        sent_text = sentence.split(" ")

        # sum of vectors
        sum_sentence = np.zeros(200)
        sum_trans = np.zeros(200)

        for s in sent_text:
            sum_sentence += kor_model.wv[s]

        if translate.sentence != "":
            trans_sentence = translate.sentence
            trans_text = trans_sentence.split(" ")
            for t in trans_text:
                if t != '':
                    sum_trans += eng_model.wv[t]
            sum_trans /= len(trans_text)

        # mean vectors
        sum_sentence /= len(sent_text)


        state = np.concatenate((sum_sentence,sum_trans), axis=0) # input sentence;번역중인 문장

        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon -= 1/3000
        # if self.epsilon < 30 and self.record == 0:
        #     self.epsilon = 50
        final_move = np.zeros(len(eng_list))
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, len(eng_list) - 1)
            final_move[move] = 1
            if move == len(eng_list):
                print("rand stop")
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train(data = TEST_DATA):
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    record = -100
    agent = Agent()
    game = Translate(data)

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # text = font.render("Game: " + str(agent.n_games), True, (255,255,255))
            # game.display.blit(text, [game.w -60, game.h])
            agent.train_long_memory()
            # pygame.display.flip()

            if reward > record or agent.n_games%100 == 0:
                if reward > record:
                    record = reward
                    agent.record = record
                agent.model.save()
                f = open('ngames.txt','w')
                f.write(str(agent.n_games) + '\n')
                f.write(str(agent.epsilon))
                f.close()

            print('Game', agent.n_games, 'Record:', record, 'Reward: ', reward, 'Epsilon', agent.epsilon)

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train(DATA)