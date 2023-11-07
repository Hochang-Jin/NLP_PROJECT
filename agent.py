import pygame.display
import torch
import random
import numpy as np
from collections import deque
from game import GameAI, Point, BLOCK_SIZE, font
from model import Linear_QNet, QTrainer
from map import mapMaker, minDistance
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 1e-3

MAP_X = 30
MAP_Y = 20
N_WALLS = 0


class Agent:

    def __init__(self,model=Linear_QNet(13,16,4),ngames=0,record=0,epsilon=60):
        self.record = record
        self.n_games = ngames
        self.epsilon = epsilon # randomness
        self.gamma = 1 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.player
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        state = [
            # dangerUp
            game.is_collision(point_u),
            # dangerRight
            game.is_collision(point_r),
            # dangerDown
            game.is_collision(point_d),
            # dangerLeft
            game.is_collision(point_l),

            game.score,
            # goal
            game.is_goal(point_u), # up is goal
            game.is_goal(point_r), # right is goal
            game.is_goal(point_d), # down is goal
            game.is_goal(point_l), # left is goal

            # goal location
            game.player.y > game.goal.y, # goal is in the upward
            game.player.x < game.goal.x, # goal is in the rightward
            game.player.y < game.goal.y,
            game.player.x > game.goal.x
        ]

        return np.array(state, dtype=int)

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
        self.epsilon -= 1/((MAP_X * MAP_Y + N_WALLS) / 10)
        if self.epsilon < 30 and self.record == 0:
            self.epsilon = 50
        final_move = [0, 0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def setMap(type='new'):
    if type == 'new':
        Map = mapMaker(MAP_X, MAP_Y, N_WALLS)
        f = open('map.txt','w')
        f.write(np.array2string(Map,precision=2, separator=',', suppress_small=True))
        f.close()

        agent = Agent()
        game = GameAI(map=Map)
        return agent, game
    elif type == 'old':
        f = open('map.txt', 'r')
        fmap = f.read()
        fmap = eval('np.array(' + fmap + ')')
        f.close()
        f = open('ngames.txt','r')
        ngames = f.readline()
        epsilon = f.readline()
        ngames = int(ngames)
        epsilon = float(epsilon)
        f.close()

        model = Linear_QNet(13, 16, 4)
        model.load_state_dict(torch.load('./model/model.pth'))
        model.eval()

        agent = Agent(model=model,ngames=ngames,epsilon=epsilon)
        game = GameAI(map=fmap)
        return agent, game

def train():
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    record = -100

    # Setting Map
    agent, game = setMap('new')

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            text = font.render("Game: " + str(agent.n_games), True, (255,255,255))
            game.display.blit(text, [game.w -60, game.h])
            agent.train_long_memory()
            pygame.display.flip()

            if score > record or agent.n_games%100 == 0:
                if score > record:
                    record = score
                    agent.record = record
                agent.model.save()
                f = open('ngames.txt','w')
                f.write(str(agent.n_games) + '\n')
                f.write(str(agent.epsilon))
                f.close()

            print('Game', agent.n_games, 'Record:', record, 'Reward: ', reward, 'Epsilon', agent.epsilon)

            if agent.n_games%300 == 0 :
                break

            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    while True:
        train(  )