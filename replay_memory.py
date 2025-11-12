import random
from collections import deque, namedtuple

# Define the structure of an experience transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """A finite-sized replay memory to store past experiences."""

    def __init__(self, capacity):
        """
        Args:
            capacity (int): The maximum number of transitions to store.
        """
        # Use a deque (double-ended queue) for efficient storage
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly samples a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of the memory."""
        return len(self.memory)