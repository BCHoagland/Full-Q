import random
import torch
from collections import deque

class Storage:
    def __init__(self):
        self.buffer = deque(maxlen=int(1e5))

    '''store a single transition'''
    def store(self, data):
        with torch.no_grad():
            # make all data 1-dimensional tensors
            def fix(x):
                if len(x.shape) == 0: x = x.unsqueeze(-1)
                return x

            data = tuple(fix(d) for d in data)
            self.buffer.append(data)


    '''return a random sample from the stored transitions'''
    def sample(self, batch_size):
        with torch.no_grad():
            batch_size = min(len(self.buffer), batch_size)
            batch = random.sample(self.buffer, batch_size)

            n = len(self.buffer[0])
            data = (torch.stack([arr[i] for arr in batch]) for i in range(n))
            return data


    '''clear stored transitions'''
    def clear(self):
        self.buffer.clear()