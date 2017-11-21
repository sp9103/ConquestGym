import tensorflow as tf
import numpy as np
import random
import time
import gym

from Model import DQN

#tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
tf.app.flags.DEFINE_string("train", "",
                           """method to train, """
                           """1. DQN, """
                           """2. Double DQN, """
                           """ex) --train=DQN.""")
FLAGS = tf.app.flags.FLAGS

# 최대 학습 횟수
MAX_EPISODE = 400
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 1000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다.
OBSERVE = 100

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def train():
    print('let\'s start train')
    sess = tf.Session()

    game = gym.make('CartPole-v0')
    NUM_ACTION = game.action_space

    net = DQN(sess, 4, 2)          #example has 4 state

    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    epsilon = 1.0
    time_step = 0
    cost = 0.0
    total_reward_list = []

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        net.init_state(state)

        while not terminal:
            if np.random.rand() < epsilon:
                action = random.randrange(2)
            else:
                action = net.get_action()

            # 일정 시간이 지난 뒤 부터 입실론 값을 줄입니다.
            # 초반에는 학습이 전혀 안되어 있기 때문입니다.
            if episode > OBSERVE:
                epsilon -= 1 / 1000

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal, info = game.step(action)
            total_reward += reward

            # 현재 상태를 Brain에 기억시킵니다.
            # 기억한 상태를 이용해 학습하고, 다음 상태에서 취할 행동을 결정합니다.
            net.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                if FLAGS.train == "DQN":
                    cost = net.train()
                elif FLAGS.train == "DDQN":
                    cost = net.train_DDQN()
                #elif FLAGS.train == "DuelDQN":

            if time_step % TARGET_UPDATE_INTERVAL == 0:
                net.update_target_network()

            time_step += 1

            #game.render()

        print('게임횟수: %d 점수: %.1f cost : %f' % (episode + 1, total_reward, cost))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)

    game.close()

def replay():
    print('let\'s start train')
    sess = tf.Session()


    game = gym.make('CartPole-v0')
    NUM_ACTION = game.action_space
    net = DQN(sess, 4, 2)  # example has 4 state

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False

        state = game.reset()
        net.init_state(state)

        while not terminal:
            action = net.get_action()
            state, reward, terminal, info = game.step(action)
            net.remember(state, action, reward, terminal)

            time.sleep(0.1)
            game.render()

    game.close()

def main(_):
    """reward = 0
    done = False

    env = gym.make('CartPole-v0')

    agent = RandomAgent(env.action_space)

    for i in range(MAX_EPISODE):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(0)
            if done:
                break

            env.render()
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

            # Close the env and write monitor result info to disk

    env.close()"""

    if FLAGS.train:
        if FLAGS.train == "DQN":
            print("DQN train start")
            train()
        elif FLAGS.train == "DDQN":
            print("Double DQN train start")
            train()
        elif FLAGS.train == "DuelDQN":
            print("Duel DQN train start")
            train()
        else:
            print("Invalid option\n"
                  "DQN or Double DQN or Duel DQN")
    else:
        replay()

if __name__ == '__main__':
    tf.app.run()