from dqn_agent import *
from model import DUELQNetwork

ALPHA = 0.1   # importance sampling exponent
BETA = 0.5    # weight update normalization constant
FRAC = 0.0002 # fractional to reach a beta of 1 in 2000 episodes

class preplay_agent(Agent):
    '''
    Modified version of DQN that incorporates Double Networks, Dueling Networks and Prioritized Replay
    '''

    def __init__(self, state_size, action_size, seed, alpha=ALPHA, beta=BETA, frac=FRAC):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        super(preplay_agent, self).__init__(state_size, action_size, seed)

        # Q-Network
        self.qnetwork_local = DUELQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DUELQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.alpha = alpha
        self.beta = beta
        self.frac = frac

    def get_beta(self, frac):
        """
        Update beta to increase the effect of the weight update normalization constant.
        """
        self.beta = self.beta + frac * (1. - self.beta)
        return self.beta

    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get arg_max_action
        argmax_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions)


        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculate beta and importance sampling weights
        beta = self.get_beta
        weights = self.memory.get_weights(beta)

        # Calculate TD-error and update the importance sampling
        td_error = Q_expected - Q_targets
        memory.update_probabilities(td_error)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)      


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)