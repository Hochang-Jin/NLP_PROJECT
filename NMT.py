# import pygame
import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math
import sys

# pygame.init()
# font = pygame.font.Font('./font/MaruBuri-SemiBold.otf', 18)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

kor_model = Word2Vec.load('model/kor.model')
eng_model = Word2Vec.load('model/eng.model')
kor_wv = kor_model.wv
eng_wv = eng_model.wv
kor_list = kor_wv.index_to_key
eng_list = eng_wv.index_to_key


class Translate:
    def __init__(self,translation):
        # self.display = pygame.display.set_mode((2160, 720))
        # pygame.display.set_caption('Translation')
        # self.clock = pygame.time.Clock()
        self.translation = translation
        self.now_translation = translation[0]
        self.index = 0
        self.sentence = ""
        self.word = ""
        self.score = 0
        self.bleu = 0
        self.flag = False

    def reset(self):
        self.index += 1
        self.index %= len(self.translation)
        self.now_translation = self.translation[self.index]
        self.sentence = ""
        self.word = ""
        self.score = 0
        self.bleu = 0
        self.flag = False

    def Update_Ui(self):
        self.display.fill(BLACK)

        # input, target 문장 표시
        text_kor = self.now_translation[0]
        text_eng = self.now_translation[1]

        text_kor = font.render(text_kor, True, WHITE)
        text_eng = font.render(text_eng, True, WHITE)

        self.display.blit(font.render('번역할 문장 : ', True, WHITE), [0, 0])
        self.display.blit(font.render('타겟 문장 : ', True, WHITE), [0, 80])


        self.display.blit(text_kor,[0,40])
        self.display.blit(text_eng,[0,120])

        # 현재 번역 중인 문장 표시
        text_sentence = font.render('번역 중인 문장 : ', True, WHITE)
        self.display.blit(text_sentence, [0,200])
        self.display.blit(font.render(self.sentence,True,WHITE), [0,240])

        # 단어 입력 창
        text_word = self.word
        text_word = font.render('입력 단어 : ' + text_word, True, WHITE)
        self.display.blit(text_word, [300,500])

        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS (초당 프레임 수) 를 위한 딜레이 추가, 딜레이 시간이 아닌 목표로 하는 FPS 값

    def play_step(self, action):
        # 1. collect user input
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()

        # 2. 단어 추가
        # if action.argmax() == len(eng_list): # action[-1] 은 <end> 로 사용
        #     pass
        # else:
        try:
            self.word = eng_list[action.argmax()]
            if self.word == self.sentence.split(" ")[-2]:
                self.flag = True

        except:
            pass
        self.sentence += self.word
        self.sentence += " "

        sentence_split = self.sentence.split(" ")
        target_split = self.now_translation[1].split(" ")[0:-1]

        now_bleu = sentence_bleu(target_split,sentence_split,weights=[1,0,0,0])
        self.score = now_bleu - self.bleu
        self.bleu = now_bleu

        # 3. check if game over
        reward = self.score
        if self.flag:
            reward -= 1
        game_over = False

        # 끝을 의미하는 action 혹은 문장이 10단어가 넘어가면 종료
        if action.argmax() == 0 or len(self.sentence.split(" ")) > 10:

            game_over = True
            # sent_len = self.sentence.split(" ")
            # sent_len.remove('')
            # rm_dupl = len(set(sent_len)) # 중복제거
            # tgt_rm_dupl = len(set(self.now_translation[1].split(" ")))
            # len_dupl = abs(rm_dupl - tgt_rm_dupl)  # 중복 제거 후 남은 갯수 비교
            # len_dupl = math.exp(-len_dupl)
            #
            # rm_dupl -= len(sent_len)
            #
            sentence_split = self.sentence.split(" ")[0:-1]
            target_split = self.now_translation[1].split(" ")
            len_gen = len(sentence_split)
            len_tgt = len(target_split)

            # sent_len = len(sent_len)
            # TODO:
            # REWARD는 BLEU 스코어로 계산
            # reward += sentence_bleu(target_split, sentence_split,smoothing_function=SmoothingFunction().method2) #* len_dupl
            # reward += rm_dupl
            reward = (self.bleu - (abs(len_gen-len_tgt) / 10))

            print()
            print("last word : {}".format(self.word))
            print("target : " + self.now_translation[1])
            print("generate : " + self.sentence)
            print("bleu : {}".format(sentence_bleu(target_split,sentence_split,weights=[1,0,0,0])))
            print("target_split : {}".format(target_split))
            print("sentence_split : {}".format(sentence_split))
            # print("len_dupl : {}".format(len_dupl))

            return reward, game_over

        # 4. update ui and clock
        # self.Update_Ui()

        # 5. return game over and score
        return reward, game_over
if __name__ == '__main__':
    Translate([["그들은 내가 잘하는 것을 바탕으로 별명을 사용하고 있기 때문에 나는 사람들이 치타라고 불러주면 기분이 좋아.","I feel happy when people call me cheetah because they are using a nickname based on something that I am good at."]]).Update_Ui()