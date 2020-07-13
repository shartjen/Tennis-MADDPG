from collections import namedtuple, deque
import random
import numpy as np
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            self.batch_size (int): size of each training batch
        """
        self.buffer_size = config['buffer_size']
        self.memory = deque(maxlen=self.buffer_size)  # internal memory (deque)
        self.action_size = config['action_size']
        self.batch_size = config['batch_size']
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # self.seed = random.seed(config['seed'])
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print('Adding experiences to memory : ')
        # print('Adding Types : ',type(state), type(action), type(reward), type(next_state), type(done))
        # print('adding shape: ',state.shape, action.shape, next_state.shape)
        # print(rewards,dones)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # print('Sample : state : ',experiences[0].state.shape)
        # print('Sample : next_state : ',experiences[0].next_state.shape)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # print('Sample : states : ',states.shape)
        # print('Sample : next_states : ',next_states.shape)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def save_buffer(self):
        states = np.vstack([e.state for e in self.memory if e is not None])
        actions = np.vstack([e.action for e in self.memory if e is not None])
        rewards = np.vstack([e.reward for e in self.memory if e is not None])
        next_states = np.vstack([e.next_state for e in self.memory if e is not None])
        dones = np.vstack([e.done for e in self.memory if e is not None]).astype(np.uint8)
        # print('save buffer : ',states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        
        return states, actions, rewards, next_states, dones
                    
    def load_buffer(self, states, actions, rewards, next_states, dones):
        for i in range(states.shape[0]):
            self.add(states[:1,:], actions[:1,:], rewards[:1,:], next_states[:1,:], dones[:1,:])
        return