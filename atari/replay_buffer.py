from collections import deque
import random
from datetime import datetime

class ReplayBuffer(object):
    MAX_LEN=100000

    def __init__(self):
        self.buffer = deque([], self.MAX_LEN)
        random.seed(datetime.now())

    def size(self):
        return len(self.buffer)

    def store(self, s, a, r, s2):
        transition = (s, a, r, s2)
        self.buffer.append(transition)

    def clear(self):
        self.buffer.clear()

    def sample(self, counts):
        num_smpl = min(counts, self.size())
        batch = random.sample(self.buffer, num_smpl)
        return batch

