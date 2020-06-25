# Report

In this repository we provide a solution to the Banana environment using deep Q learning (DQN), double deep Q learning (DDQN) and the use of a dueling network. 

## Learning algorithm

Our reinforcement learning algorithm needs at least 4 things to function properly. 

1. The environment and a connection to python (provided by Banana.exe and unity-agents)
2. The our jupyter notebook that plays episodes, collects rewards and supplies information to our reinforcement learning algorithm.
3. Our deep neural network that estimates the value of an action at a parcticular state.
4. Our DQN (or other) learning algorithm


We will describe our learning algorithm and model in the next section

### Deep neural network

In previous exercises we have always kept a Q-table with Q-values for every state in the table. This was easy to perform when our statespace was very low and discrete. By using a neural network we can instead use an approximation of our Q-values instead of the actual Q-value. Using a neural network can drastically reduce the amount of paramaters that need to be stored in memory and can handle continous values.

For our neural network we have chosen 4 layers of 48 units each. Each layer except the final layer uses the Relu non-linearity function to apply non-linearity to our network. Our final layer has 4 output units, each estimating the Q-value for each of the 4 actions.

I have chosen an input layer of 48 units so that each state can potentially be a direct mapping of a single dimension of the state. I feel that this does not squeeze our dimensions down and does not lose any information in the input layer. I have chosen a deeper network, which experimentally seemed to work better. The amount of units the same throughout the network is kept identical, because there does not seem to be a benefit to reducing the layer size and computationally would not yield a speedup anyway.

The loss function is Mean Squared Error and we use the Adam optimizer that features momentum and RMSPROP

### DQN Learning algorithm

The learning algorithm that is used consists of 3 things.

1. Experience replay.
2. Fixed Q-targets.
3. Soft update the network.
4. Hyperparameters.

#### 1. Experience replay

Our model uses a replay buffer. This is a buffer of BUFFER_SIZE episodes that stores episodes in our RAM. Whenever we invoke a learning update (and our buffer is full), we will sample episodes from our buffer equal to our BATCH_SIZE. This breaks having continious episodes where we keep repeating the same actions because it yielded good rewards. After every episode our buffer gets reduced to our BUFFER_SIZE using deque().

#### 2. Fixed Q-targets

For DQN we use the TD error to estimate the update to our networks parameters. However, when we use our model to predict the target Q value and and simultaneously estimate the old Q value we allow the network to try and hit a continiously moving target that we will simple never reach. To prevent this, instead we use a second network with its own parameters that will instead predict our Q target. The second network is updated is updated less frequently and this way we can make sure that our gradient updates are at least able to converge to our target network.

#### 3. Soft update to our network.

Instead of periodically updating our target network with our local network, we can instead provide a smooth transition by using a soft update rule: 

θ_target = τ*θ_local + (1 - τ)*θ_target.

#### 4. Hyperparameters

BUFFER_SIZE = int(1e4)  # replay buffer size

Chosen to be sufficiently large to break symmetry, but not too large to avoid slow down learning to a crawl.

BATCH_SIZE = 48         # minibatch size

Performing mini batch gradient descent can decrease randomness, while simultaneously increase learning by not having to calculate the gradient over the entire data.

GAMMA = 0.95            # discount factor

I have chosen a discount factor experimentally, where I tried to find a balance between going to a cluster of bananas that is far away, yet always going for a close by banana even when surrounded by mostly blue bananas.

TAU = 1e-3              # for soft update of target parameters

Smooth transition of the local network to the target network.

LR = 5e-4               # learning rate 

Learning rate chosen to converge our parameters to a local minimum.

UPDATE_EVERY = 4        # how often to update the network

We want to make sure we play a few episodes before improving the network to make sure that we learn information from our new parameters first.



