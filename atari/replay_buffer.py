from collections import deque
import random
from datetime import datetime

class ReplayBuffer(object):
    def __init__(self, max_len=100000):
        random.seed(datetime.now())
        self.buffer = deque([], max_len)
        self.max_len = max_len

    def size(self):
        return len(self.buffer)

    def store(self, s, a, r, s2, done):
        transition = (s, a, r, s2, done)
        self.buffer.append(transition)

    def clear(self):
        self.buffer.clear()

    def sample(self, counts):
        num_smpl = min(counts, self.size())
        batch = random.sample(self.buffer, num_smpl)
        return batch

