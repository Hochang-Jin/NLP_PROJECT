import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from map import minDistance

pygame.init()

font = pygame.font.SysFont('arial', 25)

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BUTTON1 = (200, 100, 255)


BLOCK_SIZE = 20
global SPEED
SPEED = 60

# default map
MAP  = np.array([[1,0,0,0,0],
        [0,0,0,0,2],
        [0,0,0,0,0]]) # 0: nothing, 1: player, 2: goal, 3: wall

# array to map setting
def getPoint(x, y):
    return Point(x*BLOCK_SIZE, y*BLOCK_SIZE)

class GameAI:
    def __init__(self, map=MAP):
        self.map = map
        self.w = BLOCK_SIZE * len(self.map[0])
        self.h = BLOCK_SIZE * len(self.map)
        self.walls = []
        # self.upButton = pygame.Rect(self.w, 0, 40, 40)
        self.upImage = pygame.image.load("IMG_4516.PNG")
        self.upButton = self.upImage.get_rect(center=(self.w+20, 20))
        self.downImage = pygame.image.load("IMG_4515.PNG")
        self.downButton = self.downImage.get_rect(center=(self.w + 20, 60))

        # self.downButton = pygame.Rect(self.w,40,40,40)
        # init display
        self.display = pygame.display.set_mode((self.w + 80, self.h + 40))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # map reading

        player_point = np.argwhere(self.map == 1)[0]
        goal_point = np.argwhere(self.map == 2)[0]
        walls_point = np.argwhere(self.map == 3)

        self.player = getPoint(player_point[1], player_point[0])
        self.goal = getPoint(goal_point[1], goal_point[0])
        for wall in walls_point:
            self.walls.append(getPoint(wall[1], wall[0]))

        self.score = 300
        self.frame_iteration = 0

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.upButton.collidepoint(event.pos):
                    up()
                elif self.downButton.collidepoint(event.pos):
                    down()

        # 2. move
        self._move(action)  # update the head

        # 3. check if game over
        reward = 0
        game_over = False

        # action save for collision
        tmp = action.index(1)
        taction = action.copy()
        taction[(tmp+2)%4] = 1
        taction[tmp] = 0

        if self.is_collision():
            game_over = True
            self._move(taction)
            reward = self.score / 100 - 10
            tmap = self.map.copy()
            player_point = np.argwhere(tmap==1)[0]
            tmap[player_point[0]][player_point[1]] = 0
            tmap[int(self.player.y/BLOCK_SIZE)][int(self.player.x/BLOCK_SIZE)] = 1
            reward -= minDistance(tmap) / 10
            self.score = 0
            return reward, game_over, self.score

        if self.score < 0:
            game_over = True
            reward = -10
            tmap = self.map.copy()
            player_point = np.argwhere(tmap == 1)[0]
            tmap[player_point[0]][player_point[1]] = 0
            tmap[int(self.player.y / BLOCK_SIZE)][int(self.player.x / BLOCK_SIZE)] = 1
            reward -= minDistance(tmap) / 10
            self.score = 0
            return reward, game_over, self.score

        # 4. player reaches goal or not
        if self.player == self.goal:
            reward = 100000000
            game_over = True
            self.score = 300 - self.score
            return reward, game_over, self.score
        else:
            self.score -= 1

        # 5. update ui and clock
        global SPEED
        if SPEED < 0:
            SPEED = 1
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.player
        # hits boundary
        # if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
        #     return True
        # hits wall
        if pt in self.walls:
            return True

        return False

    def is_goal(self, pt=None):
        if pt is None:
            pt = self.player
        if pt == self.goal:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # player: Blue
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.player.x, self.player.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(self.player.x + 4, self.player.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))
        # goal : Red
        pygame.draw.rect(self.display, RED, pygame.Rect(self.goal.x, self.goal.y, BLOCK_SIZE, BLOCK_SIZE))
        # wall : White
        for wall in self.walls:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(wall[0],wall[1], BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, self.h])
        # pygame.draw.rect(self.display, (200,150,30), pygame.Rect(self.w, 0, 40, 40))
        # pygame.draw.rect(self.display, (150,200,30), pygame.Rect(self.w,40,40,40))
        self.display.blit(self.upImage, self.upButton)
        self.display.blit(self.downImage, self.downButton)

        text = font.render("Speed: " + str(SPEED), True, WHITE)
        self.display.blit(text, [self.w/2, self.h])
        pygame.display.flip()

    def _move(self, action):

       x = self.player.x
       y = self.player.y
       # [up, right, down left]
       if np.array_equal(action, [1, 0, 0, 0]):
           y -= BLOCK_SIZE
       elif np.array_equal(action, [0, 1, 0, 0]):
           x += BLOCK_SIZE
       elif np.array_equal(action, [0, 0, 1, 0]):
           y += BLOCK_SIZE
       else:
           x -= BLOCK_SIZE

       self.player = Point(x, y)
       
 # Button
def up():
    global SPEED
    SPEED += 10

def down():
    global SPEED
    SPEED -= 10

if __name__ == '__main__':
    GameAI()