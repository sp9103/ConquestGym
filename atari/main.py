import gym
import dqn
from agent import RandomAgent
import util
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs='?', default='SpaceInvaders-v0')
    parser.add_argument('--num_game', default=100)
    args = parser.parse_args()

    env = gym.make(args.target)
    agent = RandomAgent(env.action_space)
    
    Q = dqn.build_dqn()

    reward = 0
    render = True

    for _ in range(args.num_game):
        print('new game has begun')
        ob = env.reset()
        ob_ = util.phi(ob)
        # plt.imshow(ob_, cmap = plt.get_cmap('gray'))
        # plt.show()

        done = False
        while not done:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if render:
                env.render()

    env.close()
