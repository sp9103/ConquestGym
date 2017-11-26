import gym
import matplotlib.pyplot as plt

from agent import RandomAgent
from replay_buffer import ReplayBuffer
import util

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?', default='SpaceInvaders-v0')
    parser.add_argument('--num_game', default=100)
    args = parser.parse_args()

    env = gym.make(args.target)
    agent = RandomAgent(env.action_space)

    reward = 0
    bRender = True
    bTrain = True

    batch_size = 5

    hasTrainedData = False
    
    if hasTrainedData:
        # load trained data
        a = 0
    else:
        # randomly initialize critic network Q(s,a|\theta^Q) and actor \mu(s|\theta^\mu) with \theta^Q and \theta^\mu
        a = 1
        
    # initialize target network Q' and \mu' with weights \theta^{Q'} <- \theta^Q, \theta^{\mu'} <- \theta^\mu
    
    R = ReplayBuffer()

    for _ in range(args.num_game):
        print('new game has begun')
        ob = env.reset()
        s_t = util.phi(ob)
        # plt.imshow(s_t, cmap = plt.get_cmap('gray'))
        # plt.show()

        r_t = 0
        done = False
        while not done:
            if bRender:
                env.render()
            a_t = agent.act(s_t, r_t, done)
            s_new, r_t, done, _ = env.step(a_t)

            if bTrain:
                R.store(s_t, a_t, r_t, s_new)
                batch = R.sample(batch_size)
                # update critic by minimizing loss
                # update the actor policy using the sampled gradient
                # update the target networks

    env.close()
