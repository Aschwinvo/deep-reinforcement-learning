from dqn_agent import *

class ddqn_agent(Agent):
    '''
    Modified version of DQN that incorporates DDQN
    '''

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        super(ddqn_agent, self).__init__(state_size, action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

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

        # print(self.qnetwork_local(next_states))
        # print(self.qnetwork_local(next_states).detach())
        # print(self.qnetwork_local(next_states).detach().max(1))
        # print(self.qnetwork_local(next_states).detach().max(1)[1])
        # print(self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1))

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).gather(1, argmax_actions)


        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)      