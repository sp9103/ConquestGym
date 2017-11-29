import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from agent import RandomAgent
from ddpg import *
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise
import util

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('target', nargs='?', default='SpaceInvaders-v0')
    parser.add_argument('target', nargs='?', default='CartPole-v0')
    parser.add_argument('--num_game', default=10000)
    args = parser.parse_args()

    env = gym.make(args.target)

    reward = 0
    bRender = True
    bTrain = True
    batch_size = 64
    gamma = .99

    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    sess = tf.Session()

    actor = ActorNetwork(sess, dim_state, dim_action)
    critic = CriticNetwork(sess, dim_state, dim_action, actor.base_ind)

    sess.run(tf.global_variables_initializer())

    actor.update_target_network(True)
    critic.update_target_network(True)
    
    R = ReplayBuffer()
    N = OUNoise(dim_action, sigma=.3)

    for _ in range(args.num_game):
        print('new game has begun')

        N.reset()
        s_t = env.reset()
        r_t = 0
        done = False
        cnt = 0
        reward = 0
        while not done:
            if bRender and cnt % 3 == 0:
                env.render()

            a_t = actor.act([s_t]) + N.noise()
            a = sess.run(tf.argmax(a_t[0]))
            s_new, r_t, done, _ = env.step(a)
            reward = reward + r_t

            if bTrain:
                R.store(s_t, a_t[0], r_t, s_new, done)

                batch = R.sample(batch_size)
                
                s_s = np.array([_[0] for _ in batch])
                a_s = np.array([_[1] for _ in batch])
                y_s = []

                for it in batch:
                    yi = it[2]
                    if it[-1]:
                        yi = yi + gamma * critic.target_Q([it[3]], actor.target_act([it[3]]))
                    y_s.append(yi)
                y_s = np.array(y_s)

                critic.update_critic(s_s, a_s, y_s)
                ag_s = critic.dQda(s_s, a_s)

                actor.update_policy(s_s, ag_s[0])

                # update the actor policy using the sampled gradient
                actor.update_target_network()
                critic.update_target_network()

            s_t = s_new
            cnt = cnt + 1
        print('reward:' + str(reward))

    env.close()
