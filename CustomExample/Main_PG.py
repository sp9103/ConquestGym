# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.
import tensorflow as tf
import numpy as np
import random
import time

from game2 import Game
from ddpg import DDPG

tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

# 최대 학습 횟수
MAX_EPISODE = 60000
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 1000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 100

REPLAY_MEMORY = 10000  # Replay buffer size

# action: 0: 좌, 1: 유지, 2: 우
NUM_ACTION = 3

SCREEN_WIDTH = 7
SCREEN_HEIGHT = 10

def train():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
    ddpg = DDPG(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION, REPLAY_MEMORY)

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    epsilon = 0.0001
    var = 1
    total_reward_list = []
    time_step = 0

    ddpg.copy_main_to_target_net()

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        ddpg.InitState(state)

        while not terminal:
            a = ddpg.get_action(state)
            a = np.random.normal(a, var)

            min_a = np.min(a)
            a = (a - min_a)
            sum_a = np.sum(a)
            a = a / sum_a
            choosed_action = np.random.choice(NUM_ACTION, 1, p=a)

            state, reward, terminal = game.step(choosed_action)
            total_reward += reward

            ddpg.remember(state, a, reward, terminal)
            mem_count = ddpg.memory.count()
            if mem_count > REPLAY_MEMORY:
                var *= .9995  # decay the action randomness
                if time_step % TRAIN_INTERVAL == 0:
                    ddpg.learn()

            time_step += 1

        print('게임횟수: %d 점수: %.1f' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, episode + 1)
            total_reward_list = []

def main(_):
    #if FLAGS.train:
    train()

if __name__ == '__main__':
    tf.app.run()
