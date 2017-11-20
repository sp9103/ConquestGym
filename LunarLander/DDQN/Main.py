import tensorflow as tf
import numpy as np
import random
import time
import gym

from Model import DQN

tf.app.flags.DEFINE_string("train", "",
                           """method to train, """
                           """1. DQN, """
                           """2. Double DQN, """
                           """ex) --train=DQN.""")
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 40000
OBSERVE = 100
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 1000

def train():
    print('let\'s start train')

    sess = tf.Session()

    game = gym.make('LunarLander-v2')
    NUM_ACTION = 4
    DIM_STATE = game.observation_space.shape[0]

    net = DQN(sess, DIM_STATE, NUM_ACTION)  # example has 4 state

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
                action = random.randrange(NUM_ACTION)
            else:
                action = net.get_action()

            if episode > OBSERVE:
                epsilon -= 1 / 1000

            state, reward, terminal, info = game.step(action)
            total_reward += reward

            net.remember(state, action, reward, terminal)

            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
               if FLAGS.train == "DQN":
                   cost = net.train()
               elif FLAGS.train == "DDQN":
                   cost = net.train()
                ##

            if time_step % TARGET_UPDATE_INTERVAL == 0:
                net.update_target_network()

            time_step += 1

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
    #sess = tf.Session()

    game = gym.make('LunarLander-v2')
    #NUM_ACTION = game.action_space.size.count()   <-action space의 사이즈 어떻게 받는지 알면 알려주셈
    NUM_ACTION = 4
    NUM_STATE = game.observation_space.shape[0]
    #net = DQN(sess, NUM_STATE, NUM_ACTION)  # example has 4 state

    #saver = tf.train.Saver()
    #ckpt = tf.train.get_checkpoint_state('model')
    #saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False

        state = game.reset()
        #net.init_state(state)

        while not terminal:
            #action = net.get_action()
            action = random.randrange(NUM_ACTION)
            state, reward, terminal, info = game.step(action)
            #net.remember(state, action, reward, terminal)

            time.sleep(0.1)
            game.render()

    game.close()

def main(_):
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