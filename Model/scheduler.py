from abc import abstractmethod, ABC

class Scheduler(ABC):
    def __init__(self, initial_lr=0.1, drop_factor=0.5, drop_every=2):
        self.lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every = drop_every
        self.epoch_count = 0

    def get_lr(self):
        return self.lr

    @abstractmethod
    def step(self):
        pass

class LinearScheduler(Scheduler):
    def step(self):
        self.epoch_count += 1
        if self.epoch_count % self.drop_every == 0:
            self.lr *= self.drop_factor