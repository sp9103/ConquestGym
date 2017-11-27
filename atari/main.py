import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import RandomAgent
from ddpg import *
from replay_buffer import ReplayBuffer
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

    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    sess = tf.Session()

    actor = ActorNetwork(sess, dim_state, dim_action)
    critic = CriticNetwork(sess, dim_state, dim_action, actor.base_ind)

    sess.run(tf.global_variables_initializer())

    actor.update_target_network(True)
    critic.update_target_network(True)
    
    R = ReplayBuffer()

    for _ in range(args.num_game):
        print('new game has begun')

        s_t = env.reset()
        r_t = 0
        done = False
        while not done:
            if bRender:
                env.render()

            a_t = actor.act([s_t])
            s_new, r_t, done, _ = env.step(a_t)

            if bTrain:
                R.store(s_t, a_t, r_t, s_new)

                batch = R.sample(batch_size)
                # update critic by minimizing loss
                # update the actor policy using the sampled gradient
                actor.update_target_network()
                critic.update_target_network()

            s_t = s_new

    env.close()
