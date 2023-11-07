import pygame
import numpy as np

pygame.init()
font = pygame.font.Font('./font/MaruBuri-SemiBold.otf', 24)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

class Translate:
    def __init__(self,translation):
        self.display = pygame.display.set_mode((1080, 720))
        pygame.display.set_caption('Translation')
        self.clock = pygame.time.Clock()
        self.translation = translation
        self.now_translation = translation[0]
        self.index = 0
        self.sentence = ""
        self.word = ""
        print(self.translation[0])

    def Update_Ui(self):
        pygame.display.flip()

    def start(self):
        self.display.fill(BLACK)
        while True:
            event = pygame.event.poll()  # 이벤트 처리
            if event.type == pygame.QUIT:
                break

            # input, target 문장 표시
            text_kor = self.now_translation[0]
            text_eng = self.now_translation[1]

            text_kor = font.render('input 문장 : ' + text_kor, True, WHITE)
            text_eng = font.render('target 문장 : ' + text_eng, True, WHITE)

            self.display.blit(text_kor,[0,0])
            self.display.blit(text_eng,[540,0])

            # 현재 번역 중인 문장 표시
            text_sentence = self.sentence
            text_sentence = font.render('번역 중인 문장 : ' + text_sentence, True, WHITE)
            self.display.blit(text_sentence, [340,200])

            # 단어 입력 창
            text_word = self.word
            text_word = font.render('입력 단어 : ' + text_word, True, WHITE)
            self.display.blit(text_word, [200,500])

            self.Update_Ui()  # 모든 화면 그리기 업데이트
            self.clock.tick(30)  # 30 FPS (초당 프레임 수) 를 위한 딜레이 추가, 딜레이 시간이 아닌 목표로 하는 FPS 값


Translate([["안녕하세요","hello"]]).start()