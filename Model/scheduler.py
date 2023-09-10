from abc import abstractmethod, ABC

class Scheduler(ABC):
    """
    Abstract base class for learning rate schedulers.

    Attributes:
        initial_lr (float): Initial learning rate.
        drop_factor (float): Factor by which the learning rate is dropped.
        drop_every (int): Number of epochs before dropping the learning rate.
        epoch_count (int): Count of epochs.

    Methods:
        get_lr(self): Get the current learning rate.
        step(self): Update the scheduler's internal state.

    """
    def __init__(self, initial_lr=0.1, drop_factor=0.5, drop_every=2):
        """
        Initialize a learning rate scheduler.

        Args:
            initial_lr (float): Initial learning rate.
            drop_factor (float): Factor by which the learning rate is dropped.
            drop_every (int): Number of epochs before dropping the learning rate.
        """
        self.lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every = drop_every
        self.epoch_count = 0

    def get_lr(self):
        """
        Get the current learning rate.

        Returns:
            float: Current learning rate.
        """
        return self.lr

    @abstractmethod
    def step(self):
        """
        Update the scheduler's internal state.

        This method should be called at the end of each epoch to update the learning rate.

        """
        pass


class LinearScheduler(Scheduler):
    """
    Linear learning rate scheduler.

    Methods:
        step(self): Update the learning rate based on the linear schedule.

    Attributes:
        None
    """

    def step(self):
        """
        Update the learning rate based on the linear schedule.

        The learning rate is dropped by the specified factor every 'drop_every' epochs.

        """
        self.epoch_count += 1
        if self.epoch_count % self.drop_every == 0:
            self.lr *= self.drop_factor
