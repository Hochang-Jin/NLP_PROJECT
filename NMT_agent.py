import pygame.display
import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from NMT import Translate

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 1e-3

# TODO:
# tokenize
# 단어 임베딩
# 문장을 임베딩된 벡터의 배열로 바꿈

class Agent:

    def __init__(self,model=Linear_QNet(13,16,4),ngames=0,record=0,epsilon=60):
        self.record = record
        self.n_games = ngames
        self.epsilon = epsilon # randomness
        self.gamma = 1 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, translate):
        state = [
            # input sentence
            translate.now_translation[0],
            # 번역중인 문장
            translate.sentence
        ]

        return np.array(state)
