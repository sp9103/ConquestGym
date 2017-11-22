# 장애물 회피 게임 즉, 자율주행차:-D 게임을 구현합니다.
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Game:
    def __init__(self, screen_width, screen_height, show_game=True):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # 도로의 크기는 스크린의 반으로 정하며, 도로의 좌측 우측의 여백을 계산해둡니다.
        self.road_width = int(4)
        self.road_left = int(2)
        self.road_right = int(self.road_left + self.road_width - 1)

        # 자동차와 장애물의 초기 위치와, 장애물 각각의 속도를 정합니다.
        self.car_size = 0.8
        self.car = {"col": 1, "row": 2, "width": self.car_size, "height": self.car_size}
        self.block = [
            {"col": 0, "row": 0, "speed": 1, "width": self.car_size, "height": self.car_size},
            {"col": 0, "row": 0, "speed": 2, "width": self.car_size, "height": self.car_size},
            {"col": 0, "row": 0, "speed": 1, "width": self.car_size, "height": self.car_size},
            {"col": -1, "row": -1, "speed": 1, "width": self.car_size, "height": self.car_size}
        ]

        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.show_game = show_game

        if show_game:
            self.fig, self.axis = self._prepare_display()

    def _prepare_display(self):
        """게임을 화면에 보여주기 위해 matplotlib 으로 출력할 화면을 설정합니다."""
        fig, axis = plt.subplots(figsize=(4, 6))
        fig.set_size_inches(4, 6)
        # 화면을 닫으면 프로그램을 종료합니다.
        fig.canvas.mpl_connect('close_event', exit)
        plt.axis((0, self.screen_width, 0, self.screen_height))
        plt.tick_params(top='off', right='off',
                        left='off', labelleft='off',
                        bottom='off', labelbottom='off')

        plt.draw()
        # 게임을 진행하며 화면을 업데이트 할 수 있도록 interactive 모드로 설정합니다.
        plt.ion()
        plt.show()

        return fig, axis

    def _get_state(self):
        """게임의 상태를 가져옵니다.
        게임의 상태는 screen_width x screen_height 크기로 각 위치에 대한 상태값을 가지고 있으며,
        빈 공간인 경우에는 0, 사물이 있는 경우에는 1이 들어있는 1차원 배열입니다.
        계산의 편의성을 위해 2차원 -> 1차원으로 변환하여 사용합니다.
        """
        state = np.zeros((self.screen_width, self.screen_height))

        state[self.car["col"], self.car["row"]] = 1

        if self.block[0]["row"] < self.screen_height:
            state[self.block[0]["col"], self.block[0]["row"]] = 2

        if self.block[1]["row"] < self.screen_height:
            state[self.block[1]["col"], self.block[1]["row"]] = 3

        if self.block[2]["row"] < self.screen_height:
            state[self.block[2]["col"], self.block[2]["row"]] = 4
        if self.block[2]["row"]+1 < self.screen_height:
            state[self.block[2]["col"], self.block[2]["row"]+1] = 4

        if self.block[3]["row"] != -1 and self.block[3]["col"] != -1:
            if self.block[3]["row"] < self.screen_height:
                state[self.block[3]["col"], self.block[3]["row"]] = 2

        return state

    def _draw_screen(self):
        title = " Avg. Reward: %d Reward: %d Total Game: %d" % (
                        self.total_reward / self.total_game,
                        self.current_reward,
                        self.total_game)

        self.axis.clear()
        self.axis.set_title(title, fontsize=12)

        road = patches.Rectangle((self.road_left - 1, 0),
                                 self.road_width + 1, self.screen_height,
                                 linewidth=0, facecolor="#333333")

        # 중앙선
        xcoords = [2.5, 3.5, 4.5]
        for xc in xcoords:
            plt.axvline(x=xc, color='y', linestyle='--')

        # 가드레일
        xcoords = [1.3, 5.8]
        for xc in xcoords:
            plt.axvline(x=xc, color='gray', linewidth = 4)

        # 자동차, 장애물들을 1x1 크기의 정사각형으로 그리도록하며, 좌표를 기준으로 중앙에 위치시킵니다.
        # 자동차의 경우에는 장애물과 충돌시 확인이 가능하도록 0.5만큼 아래쪽으로 이동하여 그립니다.
        car = patches.Rectangle((self.car["col"] - 0.4, self.car["row"] - 0.4),
                                self.car_size, self.car_size,
                                linewidth=0, facecolor="#00FF00")
        block1 = patches.Rectangle((self.block[0]["col"] - 0.4, self.block[0]["row"]),
                                   self.car_size, self.car_size,
                                   linewidth=0, facecolor="#0000FF")
        block2 = patches.Rectangle((self.block[1]["col"] - 0.4, self.block[1]["row"]),
                                   self.car_size, self.car_size,
                                   linewidth=0, facecolor="#FF0000")
        block3 = patches.Rectangle((self.block[2]["col"] - 0.4, self.block[2]["row"]),
                                   self.car_size, self.car_size,
                                   linewidth=0, facecolor="black")
        block3_tail = patches.Rectangle((self.block[2]["col"] - 0.4, self.block[2]["row"] + 1 - 0.2),
                                   self.car_size, 1,
                                   linewidth=0, facecolor="black")

        block4 = patches.Rectangle((self.block[3]["col"] - 0.4, self.block[3]["row"]),
                                   self.car_size, self.car_size,
                                   linewidth=0, facecolor="#0000FF")

        self.axis.add_patch(road)
        self.axis.add_patch(car)
        self.axis.add_patch(block1)
        self.axis.add_patch(block2)
        self.axis.add_patch(block3)
        self.axis.add_patch(block3_tail)
        self.axis.add_patch(block4)

        self.fig.canvas.draw()
        # 게임의 다음 단계 진행을 위해 matplot 의 이벤트 루프를 잠시 멈춥니다.
        plt.pause(0.0001)

    def reset(self):
        """자동차, 장애물의 위치와 보상값들을 초기화합니다."""
        self.current_reward = 0
        self.total_game += 1

        self.car["col"] = int(self.screen_width / 2)

        self.block[0]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[0]["row"] = 0
        self.block[1]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[1]["row"] = 0
        self.block[2]["col"] = random.randrange(self.road_left, self.road_right + 1)
        self.block[2]["row"] = 0
        self.block[3]["col"] = -1
        self.block[3]["row"] = -1

        self._update_block()

        return self._get_state()

    def _update_car(self, move):
        """액션에 따라 자동차를 이동시킵니다.
        자동차 위치 제한을 도로가 아니라 화면의 좌우측 끝으로 하고,
        도로를 넘어가면 패널티를 주도록 학습해서 도로를 넘지 않게 만들면 더욱 좋을 것 같습니다.
        """

        """# 자동차의 위치가 도로의 좌측을 넘지 않도록 합니다: max(0, move) > 0
        self.car["col"] = max(self.road_left, self.car["col"] + move)
        # 자동차의 위치가 도로의 우측을 넘지 않도록 합니다.: min(max, screen_width) < screen_width
        self.car["col"] = min(self.car["col"], self.road_right)"""

        self.car["col"] = self.car["col"] + move

    def _update_block(self):
        """장애물을 이동시킵니다.
        장애물이 화면 내에 있는 경우는 각각의 속도에 따라 위치 변경을,
        화면을 벗어난 경우에는 다시 방해를 시작하도록 재설정을 합니다.
        """
        reward = 0

        if self.block[0]["row"] > 0:
            self.block[0]["row"] -= self.block[0]["speed"]
        else:
            while 1:
                rand_val = random.randrange(self.road_left, self.road_right + 1)
                if rand_val != self.block[1]["col"] and rand_val != self.block[2]["col"]:
                    break

            self.block[0]["col"] = rand_val
            self.block[0]["row"] = self.screen_height
            reward += 1

        if self.block[1]["row"] > 0:
            self.block[1]["row"] -= self.block[1]["speed"]
        else:
            while 1:
                rand_val = random.randrange(self.road_left, self.road_right + 1)
                if rand_val != self.block[0]["col"] and rand_val != self.block[2]["col"]\
                        and rand_val != self.block[3]["col"]:
                    break

            self.block[1]["col"] = rand_val
            self.block[1]["row"] = self.screen_height
            reward += 1

        if self.block[2]["row"] > 0:
            self.block[2]["row"] -= self.block[2]["speed"]
        else:
            while 1:
                rand_val = random.randrange(self.road_left, self.road_right + 1)
                if rand_val != self.block[0]["col"] and rand_val != self.block[1]["col"]:
                    break

            self.block[2]["col"] = rand_val
            self.block[2]["row"] = self.screen_height - 1
            reward += 1

        if self.block[3]["row"] > 0 and self.block[3]["col"] != -1:
            self.block[3]["row"] -= self.block[3]["speed"]
        else:
            self.block[3]["col"] = self.block[3]["row"] = -1
            rand_val = random.randrange(self.road_left, self.road_right + 1)
            if rand_val == self.block[0]["col"] and self.block[0]["row"] != (self.screen_height - 1):
                self.block[3]["col"] = rand_val
                self.block[3]["row"] = self.screen_height - 1
                reward += 1
            elif rand_val == self.block[2]["col"] and self.block[2]["row"] < (self.screen_height - 2):
                self.block[3]["col"] = rand_val
                self.block[3]["row"] = self.screen_height - 1
                reward += 1

        return reward

    def _is_car_smash(self, block):
        if self.car["col"] == block["col"]:
            """block_row_pos = [block["row"] - block["height"] / 2, block["row"] + block["height"] / 2]
            car_row_pos = [self.car["row"] - self.car["height"] / 2, self.car["row"] + self.car["height"] / 2]
            if block_row_pos[0] <= car_row_pos[0] and car_row_pos[0] <= block_row_pos[1]:
                return True

            if block_row_pos[0] <= car_row_pos[1] and car_row_pos[1] <= block_row_pos[1]:
                return True"""

            if block["row"] == self.car["row"] or block["row"] + 1 == self.car["row"]:
                return True

        return False

    def _is_gameover(self):
        # 장애물과 자동차가 충돌했는지를 파악합니다.
        # 사각형 박스의 충돌을 체크하는 것이 아니라 좌표를 체크하는 것이어서 화면에는 약간 다르게 보일 수 있습니다.

        # 그냥 승용차
        if self._is_car_smash(self.block[0]) or self._is_car_smash(self.block[3]):
            self.total_reward += self.current_reward

            return -2
        # 외제차
        if self._is_car_smash(self.block[1]):
            self.total_reward += self.current_reward

            return -4
        
        # 트럭
        if self._is_car_smash(self.block[2]):
            self.total_reward += self.current_reward

            return -4
        tail_block = {"col": self.block[2]["col"], "row": self.block[2]["row"]+1, "speed": 1, "width": self.car_size, "height": self.car_size}
        if self._is_car_smash(tail_block):
            self.total_reward += self.current_reward

            return -4
        
        # 가드레일을 박았을 때도 패널티를 줌
        if self.car["col"] < self.road_left or self.car["col"] > self.road_right:

            self.total_reward += self.current_reward

            return -1

        return 0

    def step(self, action):

        reward = 0
        isGameover = False
        # action: 0: 좌, 1: 유지, 2: 우
        # action - 1 을 하여, 좌표를 액션이 0 일 경우 -1 만큼, 2 일 경우 1 만큼 옮깁니다.
        self._update_car(action - 1)
        # 장애물을 이동시킵니다. 장애물이 자동차에 충돌하지 않고 화면을 모두 지나가면 보상을 얻습니다.
        escape_reward = self._update_block()
        # 움직임이 적을 경우에도 보상을 줘서 안정적으로 이동하는 것 처럼 보이게 만듭니다.
        stable_reward = 1. / self.screen_height if action == 1 else 0
        # 게임이 종료됐는지를 판단합니다. 자동차와 장애물이 충돌했는지를 파악합니다.
        gameover = self._is_gameover()

        if gameover != 0:
            # 장애물에 충돌한 경우 -2점을 보상으로 줍니다. 장애물이 두 개이기 때문입니다.
            # 장애물을 회피했을 때 보상을 주지 않고, 충돌한 경우에만 -1점을 주어도 됩니다.
            reward += gameover
            isGameover = True
        else:
            reward = escape_reward + stable_reward
            # reward = stable_reward
            self.current_reward += reward
            isGameover = False

        if self.show_game:
            self._draw_screen()

        return self._get_state(), reward, isGameover